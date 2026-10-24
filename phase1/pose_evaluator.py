"""
Pose correction evaluation:

- Temporal Stability (pose acceleration variance)
- Skeleton Metrics (limb length consistency, bio-mechanical plausibility,
  motion jitter)

Compares:
  - Baseline: original (noisy) pose.
  - Experiment: corrected pose.

Dependencies: numpy

Usage:
  # From keypoints JSON: use smoothed file for baseline, corrected file for experiment.
  # Corrected file has keypoints_smoothed (original) and keypoints (corrected).
  from pose_handler.common_io import read_json
  from pose_evaluator import PoseEvaluator, frames_to_landmarks

  data_orig = read_json("kalman_out/video1_smoothed_keypoints.json")
  data_corr = read_json("kalman_out/video1_smoothed_keypoints_corrected.json")
  landmarks_orig = frames_to_landmarks(data_orig["frames"], key_field="keypoints")
  landmarks_corr = frames_to_landmarks(data_corr["frames"], key_field="keypoints")

  ev = PoseEvaluator()
  report, success = ev.generate_report(
      landmarks_original=landmarks_orig,
      landmarks_corrected=landmarks_corr,
      frames_original=data_orig["frames"],
      frames_corrected=data_corr["frames"],
      image_size=data_orig.get("image_size"),
  )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional: use project openpose18 order for converting keypoints -> landmarks array
try:
    from pose_handler.openpose18_mapping import OPENPOSE18_KEYPOINT_NAMES
except ImportError:
    OPENPOSE18_KEYPOINT_NAMES = [
        "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
        "Left Shoulder", "Left Elbow", "Left Wrist",
        "Right Hip", "Right Knee", "Right Ankle",
        "Left Hip", "Left Knee", "Left Ankle",
        "Right Eye", "Left Eye", "Right Ear", "Left Ear",
    ]

# COCO-18 indices for wrists and elbows (temporal stability joints of interest)
WRIST_ELBOW_JOINT_INDICES = (3, 4, 6, 7)  # R Elbow, R Wrist, L Elbow, L Wrist


def _normalize_keypoints_for_corrector(kps: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure keypoint dict uses keys the corrector expects (e.g. left_shoulder, right_elbow).
    pose_handler.calc_angles.keypoints_to_arrays expects lowercase with underscores.
    """
    if not kps:
        return {}
    out: Dict[str, Any] = {}
    for name, val in kps.items():
        if val is None:
            continue
        key = name.lower().replace(" ", "_") if isinstance(name, str) else name
        out[key] = val
    return out


def _frames_needing_correction(
    frames: List[Dict[str, Any]],
    key_field: str = "keypoints",
    alpha: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    Use the same logic as run_video_correction_pipeline: run PoseCorrector on each
    frame and count how many frames need correction (corrections_applied left or right).
    Returns dict compatible with biomechanical_plausibility, or None on failure.
    """
    try:
        import importlib.util
        _root = Path(__file__).resolve().parent
        _path = _root / "pose-corrector" / "pose_corrector.py"
        if not _path.exists():
            return None
        _spec = importlib.util.spec_from_file_location("_pose_corrector_mod", _path)
        _mod = importlib.util.module_from_spec(_spec)
        if _spec.loader is None:
            return None
        _spec.loader.exec_module(_mod)
        PoseCorrector = getattr(_mod, "PoseCorrector", None)
        if PoseCorrector is None:
            return None
    except Exception:
        return None

    frames_needing = 0
    frames_evaluated = 0
    try:
        corrector = PoseCorrector(alpha=alpha)
        for frame in frames:
            raw = frame.get(key_field) or {}
            kps = _normalize_keypoints_for_corrector(raw) if isinstance(raw, dict) else {}
            if not kps:
                continue
            try:
                result = corrector.correct_frame(kps)
            except Exception:
                continue
            if result is None:
                continue
            corrections = getattr(result, "corrections_applied", None)
            if not isinstance(corrections, dict):
                continue
            frames_evaluated += 1
            if corrections.get("left", False) or corrections.get("right", False):
                frames_needing += 1
    except Exception:
        return None

    if frames_evaluated == 0:
        return None
    return {
        "total_violations": float(frames_needing),
        "frames_with_any_violation": float(frames_needing),
        "frames_evaluated": float(frames_evaluated),
        "violation_fraction": (frames_needing / frames_evaluated) if frames_evaluated else float("nan"),
    }


def keypoint_name_to_coco18(name: str) -> str:
    """Normalize keypoint name to COCO-18 style (lowercase, underscores)."""
    return name.lower().replace(" ", "_")


def frames_to_landmarks(
    frames: List[Dict[str, Any]],
    key_field: str = "keypoints",
    num_joints: int = 18,
) -> np.ndarray:
    """
    Convert list of per-frame keypoint dicts to array [Frames, Num_Joints, 3].

    Each frame is a dict with key_field mapping joint name -> {x, y, z?, confidence?}.
    Uses OPENPOSE18_KEYPOINT_NAMES order; missing keypoints are NaN.
    """
    names_ordered = [keypoint_name_to_coco18(n) for n in OPENPOSE18_KEYPOINT_NAMES[:num_joints]]
    F = len(frames)
    out = np.full((F, len(names_ordered), 3), np.nan, dtype=np.float64)
    for t, frame in enumerate(frames):
        kps = frame.get(key_field) or {}
        for j, name in enumerate(names_ordered):
            kp = kps.get(name) if isinstance(kps, dict) else None
            if not kp or not isinstance(kp, dict):
                continue
            x = kp.get("x")
            y = kp.get("y")
            z = kp.get("z", 0.0)
            if x is None or y is None:
                continue
            out[t, j, 0] = float(x)
            out[t, j, 1] = float(y)
            out[t, j, 2] = float(z)
    return out


def compute_velocity(landmarks: np.ndarray) -> np.ndarray:
    """
    Velocity per frame and per joint: Euclidean distance from t-1 to t.
    Shape: [Frames, Num_Joints]. First frame velocity is NaN.
    """
    F, J, _ = landmarks.shape
    vel = np.full((F, J), np.nan, dtype=np.float64)
    for t in range(1, F):
        for j in range(J):
            a = landmarks[t - 1, j]
            b = landmarks[t, j]
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                continue
            vel[t, j] = float(np.linalg.norm(b - a))
    return vel


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """
    Acceleration per frame and per joint: V_t - V_{t-1}.
    Shape: [Frames, Num_Joints]. First two frames are NaN.
    """
    F, J = velocity.shape
    acc = np.full((F, J), np.nan, dtype=np.float64)
    for t in range(2, F):
        for j in range(J):
            v_prev = velocity[t - 1, j]
            v_curr = velocity[t, j]
            if np.isnan(v_prev) or np.isnan(v_curr):
                continue
            acc[t, j] = v_curr - v_prev
    return acc


def acceleration_stats(
    landmarks: np.ndarray,
    joint_indices: Tuple[int, ...] = WRIST_ELBOW_JOINT_INDICES,
) -> Dict[str, float]:
    """
    Compute mean and variance of acceleration over time for given joints.
    Ignores NaN (missing keypoints). Returns mean_acceleration and acceleration_variance.
    """
    vel = compute_velocity(landmarks)
    acc = compute_acceleration(vel)
    # Restrict to joints of interest
    acc_roi = acc[:, joint_indices]
    valid = np.isfinite(acc_roi)
    if not np.any(valid):
        return {"mean_acceleration": float("nan"), "acceleration_variance": float("nan")}
    vals = acc_roi[valid]
    return {
        "mean_acceleration": float(np.mean(vals)),
        "acceleration_variance": float(np.var(vals)),
    }


# ---------------------------------------------------------------------------
# PoseEvaluator class
# ---------------------------------------------------------------------------


class PoseEvaluator:
    """
    Evaluates pose correction by comparing:
      - Baseline: original (noisy) pose.
      - Experiment: corrected pose.

    Metrics:
      1) Temporal stability: acceleration variance on wrists/elbows (lower = smoother).
      2) Skeleton metrics: limb length consistency, biomechanical plausibility, motion jitter.
    """

    def __init__(self):
        pass

    def temporal_stability(
        self,
        landmarks_original: np.ndarray,
        landmarks_corrected: np.ndarray,
        joint_indices: Tuple[int, ...] = WRIST_ELBOW_JOINT_INDICES,
    ) -> Dict[str, Any]:
        """
        Pillar 2: Pose acceleration variance (jitter vs smoothness).

        Inputs:
          landmarks_original: [Frames, Num_Joints, 3] for original skeleton.
          landmarks_corrected: [Frames, Num_Joints, 3] for corrected skeleton.

        Returns dict with:
          original: { mean_acceleration, acceleration_variance }
          corrected: { mean_acceleration, acceleration_variance }
        """
        orig_stats = acceleration_stats(landmarks_original, joint_indices=joint_indices)
        corr_stats = acceleration_stats(landmarks_corrected, joint_indices=joint_indices)
        return {
            "original": orig_stats,
            "corrected": corr_stats,
        }

    def generate_report(
        self,
        *,
        landmarks_original: Optional[np.ndarray] = None,
        landmarks_corrected: Optional[np.ndarray] = None,
        frames_original: Optional[List[Dict[str, Any]]] = None,
        frames_corrected: Optional[List[Dict[str, Any]]] = None,
        image_size: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Run metrics, print a comparison table, and return results + success flag.

        Args:
            landmarks_original: [Frames, Num_Joints, 3] for original skeleton (for temporal stability).
            landmarks_corrected: [Frames, Num_Joints, 3] for corrected skeleton (for temporal stability).
            frames_original: List of frame dicts with keypoints for original (for biomechanical plausibility).
            frames_corrected: List of frame dicts with keypoints for corrected (for biomechanical plausibility).
            image_size: Optional dict with "width" and "height" for angle calculations.

        success: True if the video is considered to have ``succeeded'' under the evaluation
          criterion; False otherwise.

        When is a video considered success vs not?
        - If temporal stability was not computed (no landmarks given), success is True.
        - Otherwise, success = (corrected acceleration variance < original acceleration variance).
        - Specifically: we compute acceleration variance on wrists and elbows for both original
          and corrected poses. Success = True iff the corrected variance is strictly lower than
          the original. If either variance is non-finite (e.g. missing data), success is False.
        - No other metrics (limb length, bio, jitter) affect the success flag; only temporal
          stability does.
        """
        report: Dict[str, Any] = {}
        temporal_ok = True

        # ---- Temporal stability ----
        if landmarks_original is not None and landmarks_corrected is not None:
            ts = self.temporal_stability(landmarks_original, landmarks_corrected)
            report["temporal_stability"] = ts
            var_orig = ts["original"]["acceleration_variance"]
            var_corr = ts["corrected"]["acceleration_variance"]
            if np.isfinite(var_orig) and np.isfinite(var_corr):
                temporal_ok = var_corr < var_orig
            else:
                temporal_ok = False

            # ---- Skeleton metrics (limb length, bio-mechanical plausibility, motion jitter) ----
            # Compute each metric in isolation so one failure does not block the others.
            from video_metrics import (
                compute_limb_length_consistency,
                compute_motion_jitter,
                compute_biomechanical_plausibility,
            )

            # Limb length consistency (per-limb std, plus mean across limbs)
            try:
                llc_orig = compute_limb_length_consistency(landmarks_original)
                llc_corr = compute_limb_length_consistency(landmarks_corrected)
                if llc_orig is not None and llc_corr is not None:
                    report["limb_length_consistency"] = {
                        "original": llc_orig,
                        "corrected": llc_corr,
                    }
            except Exception as e:
                report["limb_length_consistency_error"] = str(e)

            # Bio-mechanical: pipeline logic (frames needing correction) with fallback to angle violations
            if frames_original is not None and frames_corrected is not None:
                try:
                    key_orig = "keypoints_smoothed" if (frames_original and isinstance(frames_original[0], dict) and "keypoints_smoothed" in frames_original[0]) else "keypoints"
                    bio_orig = _frames_needing_correction(
                        frames_original,
                        key_field=key_orig,
                        alpha=0.7,
                    )
                    bio_corr = _frames_needing_correction(
                        frames_corrected,
                        key_field="keypoints",
                        alpha=0.7,
                    )
                    if bio_orig is None or bio_corr is None:
                        # Fallback: angle-violation counts (same as video_metrics)
                        bio_orig = compute_biomechanical_plausibility(
                            frames_original,
                            key_field=key_orig,
                            image_size=image_size,
                        )
                        bio_corr = compute_biomechanical_plausibility(
                            frames_corrected,
                            key_field="keypoints",
                            image_size=image_size,
                        )
                    report["biomechanical_plausibility"] = {
                        "original": bio_orig,
                        "corrected": bio_corr,
                    }
                except Exception as e:
                    report["biomechanical_plausibility_error"] = str(e)
                    try:
                        bio_orig = compute_biomechanical_plausibility(
                            frames_original,
                            key_field="keypoints_smoothed" if (frames_original and isinstance(frames_original[0], dict) and "keypoints_smoothed" in frames_original[0]) else "keypoints",
                            image_size=image_size,
                        )
                        bio_corr = compute_biomechanical_plausibility(
                            frames_corrected,
                            key_field="keypoints",
                            image_size=image_size,
                        )
                        report["biomechanical_plausibility"] = {
                            "original": bio_orig,
                            "corrected": bio_corr,
                        }
                    except Exception:
                        pass

            # Motion jitter (second derivative of positions; lower = smoother)
            try:
                jitter_orig = compute_motion_jitter(
                    landmarks_original,
                    joint_indices=WRIST_ELBOW_JOINT_INDICES,
                )
                jitter_corr = compute_motion_jitter(
                    landmarks_corrected,
                    joint_indices=WRIST_ELBOW_JOINT_INDICES,
                )
                if jitter_orig is not None and jitter_corr is not None:
                    report["motion_jitter"] = {
                        "original": jitter_orig,
                        "corrected": jitter_corr,
                    }
            except Exception as e:
                report["motion_jitter_error"] = str(e)
        else:
            report["temporal_stability"] = None

        # ---- Print table ----
        self._print_report(report, temporal_ok)

        success = report["temporal_stability"] is None or temporal_ok
        report["success"] = success
        return report, success

    def _print_report(
        self,
        report: Dict[str, Any],
        temporal_ok: bool,
    ) -> None:
        ts = report.get("temporal_stability")
        llc = report.get("limb_length_consistency")
        bio = report.get("biomechanical_plausibility")
        mj = report.get("motion_jitter")
        lines = ["============ Pose Correction Evaluation ============"]
        if ts:
            o = ts["original"]
            c = ts["corrected"]
            lines.append("  Temporal stability (wrists & elbows)")
            lines.append(f"    Baseline  (original):  mean_accel = {o['mean_acceleration']:.6f},  var_accel = {o['acceleration_variance']:.6f}")
            lines.append(f"    Experiment (corrected): mean_accel = {c['mean_acceleration']:.6f},  var_accel = {c['acceleration_variance']:.6f}")
            lines.append(f"    -> Corrected is {'smoother (OK)' if temporal_ok else 'not smoother'}")
        if llc:
            o = llc.get("original", {})
            c = llc.get("corrected", {})
            lines.append("  Limb length consistency (std of limb lengths; lower = more rigid limbs)")
            lines.append(
                f"    Baseline  (original):  mean_limb_std = {o.get('mean_limb_length_std', float('nan')):.6f}"
            )
            lines.append(
                f"    Experiment (corrected): mean_limb_std = {c.get('mean_limb_length_std', float('nan')):.6f}"
            )
        if bio:
            o = bio.get("original", {})
            c = bio.get("corrected", {})
            lines.append("  Bio-mechanical plausibility (frames needing correction, pipeline logic; lower = better)")
            lines.append(
                f"    Baseline  (original):  total_violations = {o.get('total_violations', float('nan')):.0f},  violation_fraction = {o.get('violation_fraction', float('nan')):.6f}"
            )
            lines.append(
                f"    Experiment (corrected): total_violations = {c.get('total_violations', float('nan')):.0f},  violation_fraction = {c.get('violation_fraction', float('nan')):.6f}"
            )
        if mj:
            o = mj.get("original", {})
            c = mj.get("corrected", {})
            lines.append("  Motion jitter (acceleration energy on wrists & elbows; lower = smoother)")
            lines.append(
                f"    Baseline  (original):  mean_accel = {o.get('mean_accel_norm', float('nan')):.6f},  var_accel = {o.get('var_accel_norm', float('nan')):.6f}"
            )
            lines.append(
                f"    Experiment (corrected): mean_accel = {c.get('mean_accel_norm', float('nan')):.6f},  var_accel = {c.get('var_accel_norm', float('nan')):.6f}"
            )
        lines.append("==================================================")
        print("\n".join(lines))


def _format_float(value: Any, precision: int = 6) -> str:
    """
    Helper to format numeric values for Markdown output, handling NaN/None gracefully.
    """
    try:
        if value is None:
            return "nan"
        v = float(value)
        if not np.isfinite(v):
            return "nan"
        return f"{v:.{precision}f}"
    except Exception:
        return "nan"


def _format_mean_std(values: List[float], precision: int = 4) -> str:
    """Format as 'mean ± std' from a list of finite values; returns 'nan' if empty."""
    finite = [float(x) for x in values if _num_or_nan(x) is not None]
    if not finite:
        return "nan"
    arr = np.asarray(finite)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"


# Metric keys for unified summary and per-video tables: (label, orig_key_path, corr_key_path)
# Path is (report_section, "original"/"corrected", key_name)
_METRIC_ROWS = [
    ("mean_acceleration (temporal)", "temporal_stability", "mean_acceleration"),
    ("acceleration_variance (temporal)", "temporal_stability", "acceleration_variance"),
    ("mean_limb_length_std", "limb_length_consistency", "mean_limb_length_std"),
    ("total_violations (bio)", "biomechanical_plausibility", "total_violations"),
    ("violation_fraction (bio)", "biomechanical_plausibility", "violation_fraction"),
    ("mean_accel_norm (jitter)", "motion_jitter", "mean_accel_norm"),
    ("var_accel_norm (jitter)", "motion_jitter", "var_accel_norm"),
]


def _get_metric_values(report: Dict[str, Any]) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """Return list of (label, orig_value, corr_value) for _METRIC_ROWS; None if missing."""
    out: List[Tuple[str, Optional[float], Optional[float]]] = []
    for label, section, key in _METRIC_ROWS:
        sec = report.get(section)
        if not sec or not isinstance(sec, dict):
            out.append((label, None, None))
            continue
        o = sec.get("original") if isinstance(sec.get("original"), dict) else {}
        c = sec.get("corrected") if isinstance(sec.get("corrected"), dict) else {}
        vo = _num_or_nan(o.get(key)) if o else None
        vc = _num_or_nan(c.get(key)) if c else None
        out.append((label, vo, vc))
    return out


def _report_to_markdown(title: str, report: Dict[str, Any]) -> str:
    """Convert a single evaluation report into a compact Markdown section: one table (Metric | Original | Corrected)."""
    lines: List[str] = []
    lines.append(f"### {title}")
    lines.append("")
    success = report.get("success")
    if success is not None:
        lines.append(f"- **Success**: {bool(success)}")
        lines.append("")
    lines.append("| Metric | Original | Corrected |")
    lines.append("| --- | --- | --- |")
    for label, vo, vc in _get_metric_values(report):
        lines.append(f"| {label} | {_format_float(vo)} | {_format_float(vc)} |")
    lines.append("")
    errs: List[str] = []
    if report.get("skeleton_metrics_error"):
        errs.append(report["skeleton_metrics_error"])
    if report.get("limb_length_consistency_error"):
        errs.append("Limb length: " + report["limb_length_consistency_error"])
    if report.get("biomechanical_plausibility_error"):
        errs.append("Bio-mechanical: " + report["biomechanical_plausibility_error"])
    if report.get("motion_jitter_error"):
        errs.append("Motion jitter: " + report["motion_jitter_error"])
    if errs:
        lines.append("- **Errors**: " + "; ".join(f"`{e}`" for e in errs))
    return "\n".join(lines).rstrip()


def _summary_from_reports(
    all_reports: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Compute aggregate summary from a list of (video_name, report).
    Returns dict with counts, per-video outcomes, and metric arrays for mean±std tables.
    """
    n = len(all_reports)
    if n == 0:
        return {"total": 0}

    smoother_count = 0
    limb_better_count = 0
    bio_better_count = 0
    jitter_better_count = 0
    bio_total_orig = 0.0
    bio_total_corr = 0.0
    bio_videos_with_data = 0
    rows: List[Dict[str, Any]] = []
    # Collect per-metric values across videos for summary mean±std table
    metric_arrays: Dict[str, Tuple[List[float], List[float]]] = {
        label: ([], []) for label, _, _ in _METRIC_ROWS
    }

    for video_name, report in all_reports:
        ts = report.get("temporal_stability")
        llc = report.get("limb_length_consistency")
        bio = report.get("biomechanical_plausibility")
        mj = report.get("motion_jitter")

        # Smoother = corrected has lower acceleration variance
        smoother = False
        if ts:
            var_o = ts.get("original", {}).get("acceleration_variance")
            var_c = ts.get("corrected", {}).get("acceleration_variance")
            if np.isfinite(var_o) and np.isfinite(var_c):
                smoother = var_c < var_o
                if smoother:
                    smoother_count += 1

        # Limb: lower mean_limb_length_std = better
        limb_better = False
        if llc:
            o_std = llc.get("original", {}).get("mean_limb_length_std")
            c_std = llc.get("corrected", {}).get("mean_limb_length_std")
            if _num_or_nan(o_std) is not None and _num_or_nan(c_std) is not None:
                limb_better = c_std < o_std
                if limb_better:
                    limb_better_count += 1

        # Bio: lower violation_fraction = better; aggregate total violations
        bio_better = False
        if bio:
            o_frac = bio.get("original", {}).get("violation_fraction")
            c_frac = bio.get("corrected", {}).get("violation_fraction")
            o_viol = bio.get("original", {}).get("total_violations")
            c_viol = bio.get("corrected", {}).get("total_violations")
            if _num_or_nan(o_viol) is not None:
                bio_total_orig += float(o_viol)
            if _num_or_nan(c_viol) is not None:
                bio_total_corr += float(c_viol)
            if o_frac is not None or c_frac is not None:
                bio_videos_with_data += 1
            if _num_or_nan(c_frac) is not None and _num_or_nan(o_frac) is not None:
                bio_better = c_frac < o_frac
                if bio_better:
                    bio_better_count += 1

        # Jitter: lower var_accel_norm = smoother
        jitter_better = False
        if mj:
            o_var = mj.get("original", {}).get("var_accel_norm")
            c_var = mj.get("corrected", {}).get("var_accel_norm")
            if _num_or_nan(o_var) is not None and _num_or_nan(c_var) is not None:
                jitter_better = c_var < o_var
                if jitter_better:
                    jitter_better_count += 1

        # Collect metric values for aggregate mean±std
        for label, vo, vc in _get_metric_values(report):
            if vo is not None and np.isfinite(vo):
                metric_arrays[label][0].append(vo)
            if vc is not None and np.isfinite(vc):
                metric_arrays[label][1].append(vc)

        rows.append({
            "video": video_name,
            "smoother": smoother,
            "limb_better": limb_better,
            "bio_better": bio_better,
            "jitter_better": jitter_better,
        })

    return {
        "total": n,
        "smoother_count": smoother_count,
        "limb_better_count": limb_better_count,
        "bio_better_count": bio_better_count,
        "jitter_better_count": jitter_better_count,
        "bio_total_orig": bio_total_orig,
        "bio_total_corr": bio_total_corr,
        "bio_videos_with_data": bio_videos_with_data,
        "rows": rows,
        "metric_arrays": metric_arrays,
    }


def _num_or_nan(value: Any) -> Optional[float]:
    """Return float if finite, else None."""
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _summary_to_markdown(summary: Dict[str, Any]) -> str:
    """Format summary dict as Markdown: summary table (mean±std across videos) + counts + per-video table."""
    n = summary.get("total", 0)
    if n == 0:
        return "No videos evaluated."

    lines: List[str] = []
    lines.append("### Summary across all videos")
    lines.append("")
    lines.append(f"Evaluated **{n}** video(s). Metrics below: **mean ± std** over videos (lower is better unless noted).")
    lines.append("")
    metric_arrays = summary.get("metric_arrays") or {}
    lines.append("| Metric | Original (mean ± std) | Corrected (mean ± std) |")
    lines.append("| --- | --- | --- |")
    for label, _, _ in _METRIC_ROWS:
        orig_list, corr_list = metric_arrays.get(label, ([], []))
        orig_str = _format_mean_std(orig_list) if orig_list else "nan"
        corr_str = _format_mean_std(corr_list) if corr_list else "nan"
        lines.append(f"| {label} | {orig_str} | {corr_str} |")
    lines.append("")
    lines.append("### Videos where corrected is better")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | --- |")
    lines.append(f"| Temporal stability (smoother) | {summary.get('smoother_count', 0)} of {n} |")
    lines.append(f"| Limb length consistency | {summary.get('limb_better_count', 0)} of {n} |")
    lines.append(f"| Bio-mechanical plausibility | {summary.get('bio_better_count', 0)} of {n} |")
    lines.append(f"| Motion jitter (smoother) | {summary.get('jitter_better_count', 0)} of {n} |")
    lines.append("")
    bio_orig = summary.get("bio_total_orig")
    bio_corr = summary.get("bio_total_corr")
    if bio_videos_with_data := summary.get("bio_videos_with_data", 0):
        lines.append(
            f"**Angle / correction violations (across {bio_videos_with_data} video(s)):** "
            f"Original total = {int(bio_orig):.0f}, Corrected total = {int(bio_corr):.0f} (lower is better)."
        )
        lines.append("")

    rows = summary.get("rows", [])
    if rows:
        lines.append("### Per-video summary (corrected better?)")
        lines.append("")
        lines.append("| Video | Smoother? | Limb better? | Bio better? | Jitter better? |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in rows:
            yes_no = lambda b: "Yes" if b else "No"
            lines.append(
                f"| {r['video']} | {yes_no(r['smoother'])} | "
                f"{yes_no(r['limb_better'])} | {yes_no(r['bio_better'])} | {yes_no(r['jitter_better'])} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    from pose_handler.common_io import read_json

    ap = argparse.ArgumentParser(
        description=(
            "Pose correction evaluation: temporal stability + skeleton metrics "
            "(limb length consistency, bio-mechanical plausibility, motion jitter)."
        )
    )
    ap.add_argument("--keypoints_original", help="JSON with frames[].keypoints (original/smoothed).")
    ap.add_argument("--keypoints_corrected", help="JSON with frames[].keypoints (corrected).")
    ap.add_argument(
        "--keypoints_dir",
        help=(
            "Directory (searched recursively) for keypoint JSON pairs per video. Supported names:\n"
            "  VIDEONAME_original_comparison_smoothed_keypoints.json + VIDEONAME_corrected_comparison_smoothed_keypoints.json\n"
            "  VIDEONAME_original_generated_smoothed_keypoints.json + VIDEONAME_corrected_generated_smoothed_keypoints.json\n"
            "  VIDEONAME_smoothed_keypoints.json + VIDEONAME_smoothed_keypoints_corrected.json"
        ),
    )
    ap.add_argument(
        "--key_field",
        default="keypoints",
        help="Key in frames for landmark dict (e.g. keypoints or keypoints_smoothed).",
    )
    ap.add_argument(
        "--key_field_corrected",
        default=None,
        help="Key for corrected keypoints if different (default: same as --key_field).",
    )
    ap.add_argument(
        "--output_md",
        default="evaluation_results.md",
        help="Path to Markdown file where evaluation results will be written.",
    )
    args = ap.parse_args()

    key_corr = args.key_field_corrected if args.key_field_corrected else args.key_field
    output_md_path = Path(args.output_md)

    evaluator = PoseEvaluator()

    # Directory mode: evaluate all keypoint pairs for multiple videos.
    if args.keypoints_dir:
        dir_path = Path(args.keypoints_dir)
        if not dir_path.is_dir():
            print(f"[pose_evaluator] keypoints_dir is not a directory: {dir_path}")
            return 1

        # Support multiple naming patterns (check longer suffixes first so e.g.
        # woman3_corrected_comparison_smoothed_keypoints.json is not treated as generic _smoothed_keypoints.json)
        suffix_orig_comparison = "_original_comparison_smoothed_keypoints.json"
        suffix_corr_comparison = "_corrected_comparison_smoothed_keypoints.json"
        suffix_orig_pattern1 = "_original_generated_smoothed_keypoints.json"
        suffix_corr_pattern1 = "_corrected_generated_smoothed_keypoints.json"
        suffix_corr_pattern2 = "_smoothed_keypoints_corrected.json"
        suffix_orig_pattern2 = "_smoothed_keypoints.json"

        pairs: Dict[str, Dict[str, Path]] = {}
        for json_path in dir_path.rglob("*.json"):
            name = json_path.name
            # Comparison naming: *_original_comparison_smoothed_keypoints.json / *_corrected_comparison_smoothed_keypoints.json → same base (e.g. woman3)
            if name.endswith(suffix_orig_comparison):
                base = name[: -len(suffix_orig_comparison)]
                pairs.setdefault(base, {})["original"] = json_path
            elif name.endswith(suffix_corr_comparison):
                base = name[: -len(suffix_corr_comparison)]
                pairs.setdefault(base, {})["corrected"] = json_path
            # Pattern 1: *_original_generated_smoothed_keypoints.json / *_corrected_generated_smoothed_keypoints.json
            elif name.endswith(suffix_orig_pattern1):
                base = name[: -len(suffix_orig_pattern1)]
                pairs.setdefault(base, {})["original"] = json_path
            elif name.endswith(suffix_corr_pattern1):
                base = name[: -len(suffix_corr_pattern1)]
                pairs.setdefault(base, {})["corrected"] = json_path
            # Pattern 2: *_smoothed_keypoints_corrected.json / *_smoothed_keypoints.json (generic; don't overwrite if already set by comparison)
            elif name.endswith(suffix_corr_pattern2):
                base = name[: -len(suffix_corr_pattern2)]
                entry = pairs.setdefault(base, {})
                if "corrected" not in entry:
                    entry["corrected"] = json_path
            elif name.endswith(suffix_orig_pattern2):
                base = name[: -len(suffix_orig_pattern2)]
                entry = pairs.setdefault(base, {})
                if "original" not in entry:
                    entry["original"] = json_path

        # Only keep bases that have both original and corrected (complete pairs)
        complete_pairs = {b: p for b, p in pairs.items() if p.get("original") and p.get("corrected")}
        incomplete_bases = [b for b, p in pairs.items() if not (p.get("original") and p.get("corrected"))]
        if incomplete_bases:
            for base in sorted(incomplete_bases):
                p = pairs[base]
                missing = []
                if not p.get("original"):
                    missing.append("original")
                if not p.get("corrected"):
                    missing.append("corrected")
                print(f"[pose_evaluator] Skipping {base}: missing {', '.join(missing)} JSON (incomplete pair).")
        pairs = complete_pairs

        if not pairs:
            print(f"[pose_evaluator] No complete keypoint pairs found in directory: {dir_path}")
            print(f"[pose_evaluator] Need both original and corrected JSON per video (see --keypoints_dir help).")
            return 1

        all_reports: List[Tuple[str, Dict[str, Any]]] = []
        overall_success = True
        for video_name in sorted(pairs.keys()):
            pair = pairs[video_name]
            orig_path = pair.get("original")
            corr_path = pair.get("corrected")
            if not orig_path or not corr_path:
                missing = []
                if not orig_path:
                    missing.append("original")
                if not corr_path:
                    missing.append("corrected")
                print(f"[pose_evaluator] Skipping {video_name}: missing {', '.join(missing)} JSON (incomplete pair; only found files for one side).")
                continue

            data_o = read_json(Path(orig_path))
            data_c = read_json(Path(corr_path))
            frames_o = data_o.get("frames", [])
            frames_c = data_c.get("frames", [])
            landmarks_orig = frames_to_landmarks(frames_o, key_field=args.key_field)
            landmarks_corr = frames_to_landmarks(frames_c, key_field=key_corr)

            image_size = data_o.get("image_size")
            report, success = evaluator.generate_report(
                landmarks_original=landmarks_orig,
                landmarks_corrected=landmarks_corr,
                frames_original=frames_o,
                frames_corrected=frames_c,
                image_size=image_size,
            )
            all_reports.append((video_name, report))
            overall_success = overall_success and bool(success)

        summary = _summary_from_reports(all_reports)
        summary_md = _summary_to_markdown(summary)

        header_lines = ["## Pose Correction Evaluation Results", ""]
        with output_md_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
            f.write(summary_md)
            f.write("\n\n---\n\n## Per-video metrics\n\n")
            for idx, (video_name, rep) in enumerate(all_reports):
                if idx > 0:
                    f.write("\n\n")
                f.write(_report_to_markdown(f"Video: {video_name}", rep))
            f.write("\n")

        return 0 if overall_success else 1

    # Single-pair mode: original behavior, plus Markdown output.
    landmarks_orig = None
    landmarks_corr = None
    if args.keypoints_original and args.keypoints_corrected:
        data_o = read_json(Path(args.keypoints_original))
        data_c = read_json(Path(args.keypoints_corrected))
        frames_o = data_o.get("frames", [])
        frames_c = data_c.get("frames", [])
        landmarks_orig = frames_to_landmarks(frames_o, key_field=args.key_field)
        landmarks_corr = frames_to_landmarks(frames_c, key_field=key_corr)

        image_size = data_o.get("image_size")
        report, success = evaluator.generate_report(
            landmarks_original=landmarks_orig,
            landmarks_corrected=landmarks_corr,
            frames_original=frames_o,
            frames_corrected=frames_c,
            image_size=image_size,
        )

    if args.keypoints_original:
        title = f"Video: {Path(args.keypoints_original).stem}"
    else:
        title = "Single run"

    header_lines = ["## Pose Correction Evaluation Results", ""]
    with output_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write(_report_to_markdown(title, report))
        f.write("\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
