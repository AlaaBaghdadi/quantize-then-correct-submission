"""Bridge to phase1 kalman pose extraction (OpenPose + MediaPipe + Kalman)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .security import (
    MAX_SUBPROCESS_TIMEOUT_SEC,
    read_json_with_limit,
    safe_child_path,
    sanitize_clip_id,
    validate_keypoints_payload,
    validate_role,
    validate_video_path,
)

DEFAULT_OPENPOSE_REL = Path(
    "openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d"
) / "openpose"
DEFAULT_PHASE1_VENV_REL = Path(".venv_kinematic_gate")
PHASE1_MEDIAPIPE_PIN = "mediapipe==0.10.21"

_PREFLIGHTED_PYTHONS: set[str] = set()


def _venv_python_candidates(repo_root: Path) -> list[Path]:
    venv = repo_root / DEFAULT_PHASE1_VENV_REL
    if sys.platform == "win32":
        return [venv / "Scripts" / "python.exe"]
    return [venv / "bin" / "python"]


def resolve_phase1_python(
    python: Optional[Path],
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Python interpreter for kalman.run_video_pipeline.

    Prefers --phase1-python, then repo .venv_kinematic_gate, then sys.executable.
    """
    if python is not None:
        exe = Path(python).resolve()
        if not exe.is_file():
            raise FileNotFoundError(f"phase1 Python executable not found: {exe}")
        return exe

    if repo_root is not None:
        for candidate in _venv_python_candidates(repo_root.resolve()):
            if candidate.is_file():
                print(f"[kinematic_gate] using phase1 venv python: {candidate}")
                return candidate.resolve()

    return Path(sys.executable).resolve()


def preflight_phase1_python(python: Path, *, phase1_root: Path) -> None:
    """Ensure legacy mediapipe.solutions + opencv work in the kalman subprocess."""
    key = str(python.resolve())
    if key in _PREFLIGHTED_PYTHONS:
        return

    check = (
        "import mediapipe as mp, cv2; "
        "assert hasattr(mp, 'solutions'), "
        "'mediapipe>=0.10.30 removed mp.solutions; use mediapipe==0.10.21'"
    )
    cmd = [str(python), "-c", check]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(phase1_root),
            capture_output=True,
            text=True,
            shell=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("phase1 Python dependency check timed out") from exc

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        venv_hint = (
            f"  py -3.12 -m venv {DEFAULT_PHASE1_VENV_REL}\n"
            f"  {DEFAULT_PHASE1_VENV_REL / ('Scripts' if sys.platform == 'win32' else 'bin') / 'python'} "
            f"-m pip install -r scripts/quant/kinematic_gate/requirements.txt"
        )
        raise RuntimeError(
            f"phase1 kalman pipeline requires mediapipe==0.10.21 (legacy mp.solutions API).\n"
            f"Current interpreter: {python}\n"
            f"Recommended setup from repo root:\n{venv_hint}\n"
            f"Or pass --phase1-python to a Python with {PHASE1_MEDIAPIPE_PIN} installed.\n"
            f"Check failed: {err}"
        )

    _PREFLIGHTED_PYTHONS.add(key)


def resolve_phase1_root(phase1_root: Optional[Path], *, repo_root: Path) -> Path:
    """Resolve phase1 root from explicit path or sibling ../phase1."""
    if phase1_root is not None:
        root = Path(phase1_root).resolve()
    else:
        root = (repo_root / ".." / "phase1").resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"phase1 root not found: {root}\n"
            "Set kinematic_gate.phase1_root in vidit_ptq.json or pass --phase1-root."
        )
    preflight_phase1_module(root)
    return root


def preflight_phase1_module(phase1_root: Path) -> None:
    """Ensure phase1 kalman pipeline module exists (not a dummy path)."""
    pipeline = phase1_root / "kalman" / "run_video_pipeline.py"
    if not pipeline.is_file():
        raise FileNotFoundError(
            f"phase1 kalman pipeline not found: {pipeline}\n"
            "Expected kalman/run_video_pipeline.py under phase1 root."
        )
    metrics_py = phase1_root / "video_metrics.py"
    evaluator_py = phase1_root / "pose_evaluator.py"
    if not metrics_py.is_file() or not evaluator_py.is_file():
        raise FileNotFoundError(
            "phase1 video_metrics.py and pose_evaluator.py are required for kinematic gate."
        )


def resolve_openpose_root(
    openpose_root: Optional[Path],
    *,
    phase1_root: Path,
) -> Path:
    if openpose_root is not None:
        root = Path(openpose_root).resolve()
    else:
        root = (phase1_root / DEFAULT_OPENPOSE_REL).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OpenPose root is not a directory: {root}")
    return root


def preflight_openpose(openpose_root: Path) -> Path:
    """Verify OpenPoseDemo.exe exists; return path to executable."""
    import os  # [REV-R1] Linux build via OPENPOSE_BIN
    exe = Path(os.environ["OPENPOSE_BIN"]) if os.environ.get("OPENPOSE_BIN") else openpose_root / "bin" / "OpenPoseDemo.exe"
    if not exe.is_file():
        raise FileNotFoundError(
            f"OpenPose executable not found: {exe}\n"
            "Install OpenPose 1.7.0 Windows binaries under phase1/openpose-.../openpose/bin/."
        )
    return exe


def _video_sha256(video: Path) -> str:
    h = hashlib.sha256()
    with video.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _video_sha8(video: Path) -> str:
    return _video_sha256(video)[:8]


def staging_video_name(clip_id: str, role: str, video: Path) -> str:
    """Unique staged MP4 name to avoid OpenPose output collisions."""
    safe_clip = sanitize_clip_id(clip_id)
    safe_role = validate_role(role)
    sha = _video_sha8(video)
    return f"{safe_clip}__{safe_role}__{sha}.mp4"


def expected_keypoints_path(out_dir: Path, staged_stem: str) -> Path:
    if ".." in staged_stem or "/" in staged_stem or "\\" in staged_stem:
        raise ValueError(f"Invalid staged stem for keypoints path: {staged_stem!r}")
    return out_dir / f"{staged_stem}_smoothed_keypoints.json"


def cache_meta_path(keypoints_json: Path) -> Path:
    return keypoints_json.with_suffix(keypoints_json.suffix + ".meta.json")


def write_cache_meta(
    keypoints_json: Path,
    *,
    source_video: Path,
    source_sha256: str,
    staged_stem: str,
) -> None:
    meta = {
        "source_video": str(source_video.resolve()),
        "source_sha256": source_sha256,
        "staged_stem": staged_stem,
        "keypoints_json": str(keypoints_json.resolve()),
    }
    cache_meta_path(keypoints_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_cache_meta(keypoints_json: Path) -> Optional[Dict[str, Any]]:
    meta_path = cache_meta_path(keypoints_json)
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def cache_is_valid(cached_json: Path, source_video: Path) -> bool:
    """
    True when cached JSON exists, validates structurally, and matches source video hash.
    """
    if not cached_json.is_file():
        return False
    if cached_json.stat().st_mtime < source_video.stat().st_mtime:
        return False
    try:
        payload = read_json_with_limit(cached_json)
        validate_keypoints_payload(payload)
    except (ValueError, OSError, json.JSONDecodeError):
        return False

    meta = read_cache_meta(cached_json)
    if meta is None:
        return True  # legacy cache without meta; structural validation passed

    expected_sha = meta.get("source_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        return False
    return _video_sha256(source_video) == expected_sha


def extract_keypoints(
    video: Path,
    *,
    phase1_root: Path,
    openpose_root: Path,
    out_dir: Path,
    clip_id: str,
    role: str,
    skip_if_cached: bool = True,
    keep_staging: bool = False,
    phase1_python: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    subprocess_timeout_sec: int = MAX_SUBPROCESS_TIMEOUT_SEC,
) -> Path:
    """
    Run phase1 kalman pipeline on ``video``; return path to smoothed keypoints JSON.

    Uses a uniquely named staged copy of the video so OpenPose JSON output does not
    collide with other runs sharing the same openpose_root/output folder.
    """
    video = validate_video_path(video)
    validate_role(role)
    sanitize_clip_id(clip_id)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_name = staging_video_name(clip_id, role, video)
    staged_stem = Path(staged_name).stem
    cached = expected_keypoints_path(out_dir, staged_stem)

    if skip_if_cached and cache_is_valid(cached, video):
        print(f"[kinematic_gate] cache hit: {cached}")
        return cached

    preflight_openpose(openpose_root)
    preflight_phase1_module(phase1_root)
    python = resolve_phase1_python(phase1_python, repo_root=repo_root)
    preflight_phase1_python(python, phase1_root=phase1_root)

    staging_dir = safe_child_path(out_dir, "_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_video = staging_dir / staged_name
    shutil.copy2(video, staged_video)
    print(f"[kinematic_gate] staged video: {staged_video}")

    cmd = [
        str(python),
        "-m",
        "kalman.run_video_pipeline",
        "--video",
        str(staged_video),
        "--out_dir",
        str(out_dir),
        "--openpose_root",
        str(openpose_root.resolve()),
    ]
    print(f"[kinematic_gate] running kalman pipeline (timeout={subprocess_timeout_sec}s)")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(phase1_root),
            capture_output=True,
            text=True,
            shell=False,
            timeout=subprocess_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"kalman.run_video_pipeline timed out after {subprocess_timeout_sec}s"
        ) from exc

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"kalman.run_video_pipeline failed (exit {result.returncode}): {err}"
        )

    if not cached.is_file():
        raise FileNotFoundError(
            f"Expected keypoints JSON after pipeline: {cached}\n"
            f"stdout: {(result.stdout or '')[-2000:]}"
        )

    payload = read_json_with_limit(cached)
    frame_count = validate_keypoints_payload(payload)
    if frame_count == 0:
        raise RuntimeError(f"Pipeline produced empty keypoints JSON: {cached}")

    write_cache_meta(
        cached,
        source_video=video,
        source_sha256=_video_sha256(video),
        staged_stem=staged_stem,
    )

    if not keep_staging:
        try:
            staged_video.unlink(missing_ok=True)
        except OSError:
            pass

    print(f"[kinematic_gate] keypoints: {cached} ({frame_count} frames)")
    return cached
