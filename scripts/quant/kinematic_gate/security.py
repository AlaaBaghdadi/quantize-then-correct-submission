"""Path, input, and config validation for kinematic gate (zero-trust)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

ALLOWED_ROLES = frozenset({"fp16", "quant"})
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
MAX_KEYPOINTS_JSON_BYTES = 200 * 1024 * 1024  # 200 MiB
MAX_CLIP_ID_LEN = 128
MAX_SUBPROCESS_TIMEOUT_SEC = 4 * 3600  # 4 hours per video extraction


def sanitize_clip_id(clip_id: str) -> str:
    """Return a cache-safe clip identifier; reject path traversal."""
    if not isinstance(clip_id, str):
        raise TypeError("clip_id must be a string")
    clip_id = clip_id.strip()
    if not clip_id:
        raise ValueError("clip_id must be non-empty")
    if len(clip_id) > MAX_CLIP_ID_LEN:
        raise ValueError(f"clip_id exceeds max length {MAX_CLIP_ID_LEN}")
    if ".." in clip_id or "/" in clip_id or "\\" in clip_id:
        raise ValueError(f"clip_id must not contain path separators: {clip_id!r}")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_id)
    if not safe:
        raise ValueError("clip_id has no safe characters after sanitization")
    return safe


def validate_role(role: str) -> str:
    if role not in ALLOWED_ROLES:
        raise ValueError(f"role must be one of {tuple(sorted(ALLOWED_ROLES))}, got {role!r}")
    return role


def safe_child_path(base: Path, *parts: str) -> Path:
    """Resolve path and ensure it stays under base (prevents path traversal)."""
    base_resolved = Path(base).resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Resolved path escapes base directory: {candidate} is not under {base_resolved}"
        ) from exc
    return candidate


def validate_video_path(path: Path) -> Path:
    """Ensure path is an existing regular video file with allowed suffix."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Video not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video extension {suffix!r}; allowed: {sorted(ALLOWED_VIDEO_SUFFIXES)}"
        )
    return resolved


def validate_positive_finite(value: float, name: str, *, max_value: float | None = None) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


def validate_gate_thresholds(
    *,
    max_rel_delta_accel_variance: float,
    max_rel_delta_limb_std: float,
    max_rel_delta_jitter: float,
    max_abs_delta_violation_fraction: float,
    max_mean_npe: float,
    max_p95_npe: float,
    max_under_motion_violations: float,
) -> None:
    """Reject non-physical or dummy threshold configs."""
    validate_positive_finite(max_rel_delta_accel_variance, "max_rel_delta_accel_variance", max_value=10.0)
    validate_positive_finite(max_rel_delta_limb_std, "max_rel_delta_limb_std", max_value=10.0)
    validate_positive_finite(max_rel_delta_jitter, "max_rel_delta_jitter", max_value=10.0)
    validate_positive_finite(
        max_abs_delta_violation_fraction,
        "max_abs_delta_violation_fraction",
        max_value=1.0,
    )
    validate_positive_finite(max_mean_npe, "max_mean_npe", max_value=1.0)
    validate_positive_finite(max_p95_npe, "max_p95_npe", max_value=2.0)
    if max_p95_npe < max_mean_npe:
        raise ValueError("max_p95_npe must be >= max_mean_npe")
    if not math.isfinite(max_under_motion_violations):
        raise ValueError(f"max_under_motion_violations must be finite, got {max_under_motion_violations!r}")
    if max_under_motion_violations < 0:
        raise ValueError(f"max_under_motion_violations must be >= 0, got {max_under_motion_violations}")


def validate_keypoints_payload(data: object) -> int:
    """
    Validate kalman smoothed keypoints JSON structure.
    Returns frame count.
    """
    if not isinstance(data, dict):
        raise ValueError("keypoints JSON root must be an object")
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError("keypoints JSON must contain a 'frames' array")
    if not frames:
        raise ValueError("keypoints JSON has zero frames")
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{i}] must be an object")
        kps = frame.get("keypoints")
        if not isinstance(kps, dict):
            raise ValueError(f"frames[{i}] must contain a 'keypoints' object")
    return len(frames)


def read_json_with_limit(path: Path, *, max_bytes: int = MAX_KEYPOINTS_JSON_BYTES) -> object:
    """Read JSON file with a byte-size cap to mitigate resource exhaustion."""
    resolved = Path(path).resolve()
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"JSON file too large ({size} bytes > {max_bytes}): {resolved}"
        )
    import json

    text = resolved.read_text(encoding="utf-8")
    return json.loads(text)


def assert_no_banned_markers(source: str, *, label: str, banned: Iterable[str]) -> None:
    lowered = source.lower()
    for marker in banned:
        if marker.lower() in lowered:
            raise RuntimeError(f"{label} contains banned marker {marker!r}")
