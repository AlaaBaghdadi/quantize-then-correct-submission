"""Aggregate per-video SmoothNet metrics into a CSV table."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

CSV_COLUMNS = [
    "video",
    "method",
    "accel_var",
    "viol_frac",
    "limb_std",
    "jitter",
    "accel_var_orig",
    "viol_frac_orig",
    "limb_std_orig",
    "success",
]


def write_benchmark_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def append_or_merge_rows(
    existing_path: Path,
    new_rows: List[Dict[str, Any]],
    method: str = "smoothnet",
) -> List[Dict[str, Any]]:
    """Merge new rows; replace same (video, method) entries."""
    merged: Dict[str, Dict[str, Any]] = {}

    if existing_path.is_file():
        with existing_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("video", "")
                m = row.get("method", method)
                merged[f"{vid}|{m}"] = row

    for row in new_rows:
        vid = row.get("video", "")
        m = row.get("method", method)
        merged[f"{vid}|{m}"] = row

    return list(merged.values())


def format_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No results."

    lines = [
        f"SmoothNet benchmark: {len(rows)} video(s)",
        "",
        f"{'Video':<30} {'AccelVar':>12} {'ViolFrac':>10} {'LimbStd':>10} {'Jitter':>10}",
        "-" * 76,
    ]
    for row in rows:
        lines.append(
            f"{str(row.get('video', '')):<30} "
            f"{_fmt(row.get('accel_var')):>12} "
            f"{_fmt(row.get('viol_frac')):>10} "
            f"{_fmt(row.get('limb_std')):>10} "
            f"{_fmt(row.get('jitter')):>10}"
        )
    return "\n".join(lines)


def _fmt(val: Any) -> str:
    try:
        return f"{float(val):.6f}"
    except (TypeError, ValueError):
        return "nan"
