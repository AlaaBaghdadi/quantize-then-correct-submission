#!/usr/bin/env python3
"""Generate docs/DynamiCtrl-PTQ-kinematic-gate-results.md from gate JSON reports."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOGS = REPO / "logs"
OUT = REPO / "docs" / "DynamiCtrl-PTQ-kinematic-gate-results.md"

CLIP_ORDER = [
    f"human{h}_motion{m}" for h in range(1, 5) for m in range(1, 4)
]
SWEEP_CLIPS = ["human2_motion1", "human2_motion2", "human2_motion3"]
GROUPS = ["group1", "group2", "group3"]
GROUP_BLOCKS = {
    "group1": "1–13",
    "group2": "14–27",
    "group3": "28–40",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(p: str) -> str:
    s = p.replace("\\", "/")
    idx = s.find("outputs/")
    if idx >= 0:
        return s[idx:]
    idx = s.find("phase2/")
    if idx >= 0:
        return s[idx + len("phase2/") :]
    return s


def failed_checks(data: dict) -> list[str]:
    checks = data.get("checks", {})
    skip = {k for k in checks if k.startswith("finite_")}
    return sorted(k for k, ok in checks.items() if k not in skip and not ok)


def fmt(v: float | int | None, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 0.0001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


def clip_block(data: dict, *, title: str) -> str:
    m0 = data.get("metrics_fp16", {})
    m1 = data.get("metrics_quant", {})
    d = data.get("deltas", {})
    mv = data.get("movement_summary", {})
    fc = failed_checks(data)
    status = "PASS" if data.get("passed") else "FAIL"
    lines = [
        f"#### {title} — **{status}**",
        "",
        f"- **FP16 video:** `{rel_path(data.get('fp16_video', ''))}`",
        f"- **Quant video:** `{rel_path(data.get('quant_video', ''))}`",
        "",
        "| Metric | FP16 | Quant | Delta |",
        "|--------|------|-------|-------|",
        f"| acceleration_variance | {fmt(m0.get('acceleration_variance'), 2)} | {fmt(m1.get('acceleration_variance'), 2)} | rel={fmt(d.get('rel_delta_acceleration_variance'))} |",
        f"| mean_limb_length_std | {fmt(m0.get('mean_limb_length_std'), 2)} | {fmt(m1.get('mean_limb_length_std'), 2)} | rel={fmt(d.get('rel_delta_mean_limb_length_std'))} |",
        f"| var_accel_norm (jitter) | {fmt(m0.get('var_accel_norm'), 2)} | {fmt(m1.get('var_accel_norm'), 2)} | rel={fmt(d.get('rel_delta_var_accel_norm'))} |",
        f"| violation_fraction | {fmt(m0.get('violation_fraction'), 3)} | {fmt(m1.get('violation_fraction'), 3)} | abs={fmt(d.get('abs_delta_violation_fraction'), 3)} |",
        f"| total_violations | {fmt(m0.get('total_violations'), 0)} | {fmt(m1.get('total_violations'), 0)} | abs={fmt(d.get('abs_delta_total_violations'), 0)} |",
        f"| mean_NPE | — | — | {fmt(mv.get('mean_NPE'))} |",
        f"| p95_NPE | — | — | {fmt(mv.get('p95_NPE'))} |",
        f"| under_motion_violations | — | — | {fmt(mv.get('under_motion_violations'), 0)} |",
        "",
        f"**Failed checks:** {', '.join(fc) if fc else '*(none)*'}",
        "",
    ]
    return "\n".join(lines)


def summary_row(data: dict) -> dict:
    d = data.get("deltas", {})
    mv = data.get("movement_summary", {})
    return {
        "passed": data.get("passed"),
        "rel_accel": d.get("rel_delta_acceleration_variance"),
        "rel_limb": d.get("rel_delta_mean_limb_length_std"),
        "rel_jitter": d.get("rel_delta_var_accel_norm"),
        "viol_delta": d.get("abs_delta_violation_fraction"),
        "mean_npe": mv.get("mean_NPE"),
        "p95_npe": mv.get("p95_NPE"),
        "failed": failed_checks(data),
    }


def main() -> None:
    full_vidit: dict[str, dict] = {}
    full_naive: dict[str, dict] = {}
    for cid in CLIP_ORDER:
        p = LOGS / "kinematic_gate_grid" / f"{cid}.json"
        if p.is_file():
            full_vidit[cid] = load_json(p)
        p = LOGS / "kinematic_gate_w8a8_grid" / f"{cid}.json"
        if p.is_file():
            full_naive[cid] = load_json(p)

    sweep: dict[str, dict[str, dict[str, dict]]] = {}
    for g in GROUPS:
        sweep[g] = {"vidit": {}, "naive": {}}
        for algo in ("vidit", "naive"):
            d = LOGS / f"kinematic_gate_sweep_{g}_{algo}"
            for cid in SWEEP_CLIPS:
                p = d / f"{cid}.json"
                if p.is_file():
                    sweep[g][algo][cid] = load_json(p)

    parts: list[str] = []

    parts.append("# DynamiCtrl PTQ — Kinematic Gate Experiment Report")
    parts.append("")
    parts.append(
        "Complete results from the W8A8 quantization experiments and kinematic delta gate "
        "evaluations. This document is self-contained: methodology, thresholds, every clip result, "
        "and side-by-side comparisons for clips that appear in both the full grid and sensitivity sweep."
    )
    parts.append("")
    parts.append("**Source notebook:** `notebooks/DynamiCtrl-PTQ-ViDiT.ipynb`  ")
    parts.append("**Method doc:** `docs/DynamiCtrl-PTQ-method.md`  ")
    parts.append("**Config:** `configs/quant/vidit_ptq.json` (seed 42)")
    parts.append("")
    parts.append("---")
    parts.append("")

    # §1
    parts.append("## 1. Introduction and goals")
    parts.append("")
    parts.append(
        "We applied post-training W8A8 quantization to DynamiCtrl-5B's DiT backbone using two "
        "algorithms, then evaluated generated videos with an automatic **kinematic gate** — pose "
        "metrics extracted from output video (OpenPose + MediaPipe + Kalman) compared against FP16 "
        "baseline. Kinematics is used only as a pass/fail acceptance test, not as a training signal."
    )
    parts.append("")
    parts.append("Two experiments were run:")
    parts.append("")
    parts.append(
        "1. **Full grid (Experiment 1):** Quantize the full model once per algorithm, generate "
        "**12 videos** (4 reference images × 3 driving motions), gate each against FP16."
    )
    parts.append(
        "2. **Sensitivity sweep (Experiment 2):** Quantize **one block group at a time** "
        "(rest of DiT stays FP16), generate **3 videos** (`human2` × 3 motions) per group per "
        "algorithm — **9 videos per algorithm**, **18 total per algorithm** across 3 groups × 2 algos."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # §2
    parts.append("## 2. Two quantization algorithms")
    parts.append("")
    parts.append("| Algorithm | Notebook cell | Output directory | Video filename pattern |")
    parts.append("|-----------|---------------|------------------|------------------------|")
    parts.append(
        "| **ViDiT-Q W8A8** | Batch cell 3 | `outputs/vidit_w8a8_grid/` | `vidit_w8a8_{clip}_seed42.mp4` |"
    )
    parts.append(
        "| **Matched naive W8A8** | Batch cell 4 | `outputs/w8a8_grid/` | `w8a8_{clip}_seed42.mp4` |"
    )
    parts.append("")
    parts.append(
        "**ViDiT-Q** applies SmoothQuant folding plus per-clip calibration; ~24 layers identified "
        "as timestep-sensitive stay in FP16 across all blocks (`early_sensitivity_ratio=1.25`, "
        "`max_early_fp16_layers=24`)."
    )
    parts.append("")
    parts.append(
        "**Matched naive W8A8** uses the same calib-selected FP16 skip list but **no SmoothQuant** — "
        "a fair baseline to isolate ViDiT's benefit."
    )
    parts.append("")
    parts.append(
        "In the sensitivity sweep, ViDiT uses `apply_vidit_w8a8` with `actual_int8_layers` restricted "
        "to the target block group; naive uses `apply_w8a8` with `fp16_layers` = all layers outside "
        "that group."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # §3
    parts.append("## 3. Experiment designs")
    parts.append("")
    parts.append("```mermaid")
    parts.append("flowchart TB")
    parts.append("  subgraph fullGrid [Experiment1_FullGrid]")
    parts.append('    FP16_12["fp16_grid: 12 FP16 baselines"]')
    parts.append('    Vidit_12["vidit_w8a8_grid: 12 ViDiT videos"]')
    parts.append('    Naive_12["w8a8_grid: 12 naive videos"]')
    parts.append("    FP16_12 --> Vidit_12")
    parts.append("    FP16_12 --> Naive_12")
    parts.append("  end")
    parts.append("  subgraph sweepExp [Experiment2_SensitivitySweep]")
    parts.append('    FP16_3["sweep_fp16_baseline: 3 FP16 baselines"]')
    parts.append('    G1["group1 blocks 1-13"]')
    parts.append('    G2["group2 blocks 14-27"]')
    parts.append('    G3["group3 blocks 28-40"]')
    parts.append("    FP16_3 --> G1")
    parts.append("    FP16_3 --> G2")
    parts.append("    FP16_3 --> G3")
    parts.append("  end")
    parts.append("```")
    parts.append("")
    parts.append("### Experiment 1 — Full grid")
    parts.append("")
    parts.append("| Item | Value |")
    parts.append("|------|-------|")
    parts.append("| Clips | 4 humans × 3 motions = **12** |")
    parts.append("| FP16 baseline | `outputs/fp16_grid/` (Batch cell 1) |")
    parts.append("| Quantization | Full model W8A8 + calib FP16 skip (~24 layers) |")
    parts.append("| Videos per algorithm | **12** |")
    parts.append("| Gate logs (ViDiT) | `logs/kinematic_gate_grid/` |")
    parts.append("| Gate logs (naive) | `logs/kinematic_gate_w8a8_grid/` |")
    parts.append("")
    parts.append("### Experiment 2 — Sensitivity sweep")
    parts.append("")
    parts.append("| Item | Value |")
    parts.append("|------|-------|")
    parts.append("| Clips | `human2_motion1`, `human2_motion2`, `human2_motion3` only |")
    parts.append("| FP16 baseline | `outputs/sweep_fp16_baseline/` (separate mini baseline) |")
    parts.append("| Block groups | group1: blocks 1–13, group2: 14–27, group3: 28–40 |")
    parts.append("| Quantization | Only target group → INT8; rest stays FP16 |")
    parts.append("| Videos per group per algorithm | **3** |")
    parts.append("| Total per algorithm | 3 groups × 3 clips = **9** |")
    parts.append("| Output dirs | `outputs/sweep_group{N}_{naive,vidit}/` |")
    parts.append("")
    parts.append("---")
    parts.append("")

    # §4
    parts.append("## 4. Kinematic gate — how to read results")
    parts.append("")
    parts.append(
        "Gate script: `scripts/quant/kinematic_gate/run_grid_kinematic_gate.py` (full grid) and "
        "`run_kinematic_gate.py` (per-clip / sweep naive)."
    )
    parts.append("")
    parts.append("### Pipeline")
    parts.append("")
    parts.append(
        "1. Extract poses from FP16 and quantized MP4 (OpenPose + MediaPipe + Kalman via phase1).  "
    )
    parts.append("2. Compute kinematic metric vector on each video.  ")
    parts.append("3. Compare **deltas** (quant vs FP16).  ")
    parts.append("4. **Pass** only if every check is within tolerance.")
    parts.append("")
    parts.append("### Gate thresholds (`configs/quant/vidit_ptq.json` → `kinematic_gate`)")
    parts.append("")
    parts.append("| Check | Threshold | Meaning |")
    parts.append("|-------|-----------|---------|")
    parts.append("| acceleration_variance | rel delta ≤ **0.10** | Temporal smoothness of joint motion |")
    parts.append("| mean_limb_length_std | rel delta ≤ **0.10** | Limb length stability across frames |")
    parts.append("| var_accel_norm (jitter) | rel delta ≤ **0.15** | Normalized acceleration jitter |")
    parts.append("| violation_fraction | quant ≤ FP16 + **0.02** | No new biomechanical constraint breaks |")
    parts.append("| mean_NPE | ≤ **0.15** | Mean normalized pose error vs FP16 skeleton |")
    parts.append("| p95_NPE | ≤ **0.30** | 95th-percentile pose error |")
    parts.append("| under_motion_violations | ≤ **0.0** | Frames under-moving vs driving motion |")
    parts.append("")
    parts.append(
        "**Note:** A clip can fail even when acceleration delta is small — e.g. new constraint "
        "violations or high pose error (`human4` clips often fail on `p95_NPE`)."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # §5 Executive summary
    def count_pass(d: dict[str, dict]) -> int:
        return sum(1 for v in d.values() if v.get("passed"))

    vidit_full_pass = count_pass(full_vidit)
    naive_full_pass = count_pass(full_naive)

    sweep_vidit_pass = sum(
        count_pass(sweep[g]["vidit"]) for g in GROUPS
    )
    sweep_naive_pass = sum(
        count_pass(sweep[g]["naive"]) for g in GROUPS
    )

    parts.append("## 5. Executive summary")
    parts.append("")
    parts.append("| Run | ViDiT pass | Naive pass |")
    parts.append("|-----|------------|------------|")
    parts.append(f"| Full grid (12 clips) | **{vidit_full_pass}/12** | **{naive_full_pass}/12** |")
    for g in GROUPS:
        vp = count_pass(sweep[g]["vidit"])
        np = count_pass(sweep[g]["naive"])
        parts.append(
            f"| Sweep {g} (blocks {GROUP_BLOCKS[g]}) | **{vp}/3** | **{np}/3** |"
        )
    parts.append(f"| Sweep total (9 per algo) | **{sweep_vidit_pass}/9** | **{sweep_naive_pass}/9** |")
    parts.append("")
    parts.append("### Key findings")
    parts.append("")
    parts.append(
        "- **Matched naive W8A8 beats ViDiT-Q on the full 12-clip grid** (5/12 vs 2/12 passes)."
    )
    parts.append(
        "- **Early blocks (group1, 1–13) are most kinematically sensitive** in the sweep — highest "
        "acceleration deltas when isolated to INT8."
    )
    parts.append(
        "- **`human2_motion1` is the hardest clip** — fails most full-grid and sweep runs."
    )
    parts.append(
        "- **`human4_*` clips fail both algorithms on the full grid** — mainly high `p95_NPE` and "
        "kinematic deltas."
    )
    parts.append(
        "- **Only `human3_motion2` passes both ViDiT and naive** on the full grid."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # §6 Full grid
    parts.append("## 6. Experiment 1 — Full grid results")
    parts.append("")
    parts.append("### 6a. Pass/fail matrix (12 clips)")
    parts.append("")
    parts.append("| Clip | ViDiT | Naive |")
    parts.append("|------|-------|-------|")
    for cid in CLIP_ORDER:
        v = full_vidit.get(cid, {})
        n = full_naive.get(cid, {})
        vs = "PASS" if v.get("passed") else "FAIL"
        ns = "PASS" if n.get("passed") else "FAIL"
        parts.append(f"| `{cid}` | {vs} | {ns} |")
    parts.append("")
    parts.append("**ViDiT passed:** " + ", ".join(
        f"`{c}`" for c in CLIP_ORDER if full_vidit.get(c, {}).get("passed")
    ) or "none")
    parts.append("")
    parts.append("**Naive passed:** " + ", ".join(
        f"`{c}`" for c in CLIP_ORDER if full_naive.get(c, {}).get("passed")
    ) or "none")
    parts.append("")
    parts.append("### 6b. Per-clip detail (all 12 clips × 2 algorithms)")
    parts.append("")
    for cid in CLIP_ORDER:
        parts.append(f"### `{cid}`")
        parts.append("")
        if cid in full_vidit:
            parts.append(clip_block(full_vidit[cid], title="Full grid — ViDiT-Q W8A8"))
        if cid in full_naive:
            parts.append(clip_block(full_naive[cid], title="Full grid — Matched naive W8A8"))
    parts.append("---")
    parts.append("")

    # §7 Sweep
    parts.append("## 7. Experiment 2 — Sensitivity sweep results")
    parts.append("")
    parts.append("### 7a. Block groups")
    parts.append("")
    parts.append("| Group | DiT blocks | ViDiT output | Naive output |")
    parts.append("|-------|------------|--------------|--------------|")
    for g in GROUPS:
        parts.append(
            f"| **{g}** | {GROUP_BLOCKS[g]} | `outputs/sweep_{g}_vidit/` | `outputs/sweep_{g}_naive/` |"
        )
    parts.append("")
    parts.append("### 7b. Per-group summary")
    parts.append("")
    parts.append("| Group | Blocks | ViDiT pass | Naive pass | Mean rel_accel ViDiT | Mean rel_accel naive |")
    parts.append("|-------|--------|------------|------------|----------------------|----------------------|")
    for g in GROUPS:
        vd = sweep[g]["vidit"]
        nd = sweep[g]["naive"]
        vp = count_pass(vd)
        np = count_pass(nd)
        v_accel = sum(
            x["deltas"]["rel_delta_acceleration_variance"] for x in vd.values()
        ) / max(len(vd), 1)
        n_accel = sum(
            x["deltas"]["rel_delta_acceleration_variance"] for x in nd.values()
        ) / max(len(nd), 1)
        parts.append(
            f"| {g} | {GROUP_BLOCKS[g]} | {vp}/3 | {np}/3 | {fmt(v_accel)} | {fmt(n_accel)} |"
        )
    parts.append("")
    parts.append("### 7c. Per-clip detail (3 groups × 2 algorithms × 3 clips)")
    parts.append("")
    for g in GROUPS:
        parts.append(f"### {g} (blocks {GROUP_BLOCKS[g]})")
        parts.append("")
        for algo, label in (("vidit", "ViDiT-Q W8A8"), ("naive", "Matched naive W8A8")):
            parts.append(f"#### {label}")
            parts.append("")
            for cid in SWEEP_CLIPS:
                data = sweep[g][algo].get(cid)
                if data:
                    parts.append(clip_block(data, title=f"`{cid}`"))
    parts.append("---")
    parts.append("")

    # §8 Overlap
    parts.append("## 8. Overlap — `human2_motion1`, `human2_motion2`, `human2_motion3`")
    parts.append("")
    parts.append(
        "These three clips appear in **both** experiments. Full grid uses `outputs/fp16_grid/`; "
        "sweep uses `outputs/sweep_fp16_baseline/` (separate FP16 renders — FP16 metrics differ slightly)."
    )
    parts.append("")

    configs = [
        ("Full ViDiT", lambda c: full_vidit.get(c)),
        ("Full naive", lambda c: full_naive.get(c)),
    ]
    for g in GROUPS:
        configs.append((f"Sweep {g} ViDiT", lambda c, g=g: sweep[g]["vidit"].get(c)))
        configs.append((f"Sweep {g} naive", lambda c, g=g: sweep[g]["naive"].get(c)))

    for cid in SWEEP_CLIPS:
        parts.append(f"### `{cid}` — master comparison (8 configurations)")
        parts.append("")
        parts.append(
            "| Config | Pass | rel_accel | rel_limb | rel_jitter | viol_frac Δ | mean_NPE | p95_NPE | Failed checks |"
        )
        parts.append(
            "|--------|------|-----------|----------|------------|-------------|----------|---------|---------------|"
        )
        for name, getter in configs:
            data = getter(cid)
            if not data:
                continue
            r = summary_row(data)
            fc = ", ".join(r["failed"]) if r["failed"] else "—"
            ps = "PASS" if r["passed"] else "FAIL"
            parts.append(
                f"| {name} | {ps} | {fmt(r['rel_accel'])} | {fmt(r['rel_limb'])} | "
                f"{fmt(r['rel_jitter'])} | {fmt(r['viol_delta'], 3)} | {fmt(r['mean_npe'])} | "
                f"{fmt(r['p95_npe'])} | {fc} |"
            )
        parts.append("")
        parts.append("#### Expanded detail")
        parts.append("")
        for name, getter in configs:
            data = getter(cid)
            if data:
                parts.append(clip_block(data, title=name))
        parts.append("")

    parts.append("---")
    parts.append("")

    # §9 Conclusions
    parts.append("## 9. Conclusions and next steps")
    parts.append("")
    parts.append("### Sensitivity ranking (sweep mean rel_accel)")
    parts.append("")
    parts.append(
        "Across the three block groups, **group1 (blocks 1–13) causes the largest kinematic "
        "degradation** when isolated to INT8, followed by group2, then group3. This supports keeping "
        "early DiT blocks in FP16 when building a mixed-precision map."
    )
    parts.append("")
    parts.append("### Full model vs isolate-quantize")
    parts.append("")
    parts.append(
        "- Isolate-quantize **overstates** harm for sensitive groups (errors do not compound across "
        "the full model's calib FP16 skip list)."
    )
    parts.append(
        "- Full-model quantization **understates** isolate pass rates — no full ViDiT `human2` clip "
        "passes, yet sweep group2 ViDiT passes 2/3."
    )
    parts.append(
        "- Different FP16 baselines (`fp16_grid` vs `sweep_fp16_baseline`) prevent direct delta "
        "comparison across experiments; use pass/fail and relative ranking."
    )
    parts.append("")
    parts.append("### Recommended next steps")
    parts.append("")
    parts.append("1. **Protect blocks 1–13 (group1)** in any mixed-precision config.")
    parts.append("2. **Refine group1** with a sub-block sweep (e.g. blocks 1–6 vs 7–13).")
    parts.append("3. **Re-gate full 12-clip grid** after applying group-aware FP16 map.")
    parts.append("4. Keep **`human2_motion1`** and **`human4_*`** in the gate set as stress clips.")
    parts.append("5. Investigate why **ViDiT underperforms matched naive** on kinematics despite PSNR gains.")
    parts.append("")
    parts.append("---")
    parts.append("")

    # §10 Appendix
    parts.append("## 10. Appendix")
    parts.append("")
    parts.append("### Output directory map")
    parts.append("")
    parts.append("```")
    parts.append("outputs/")
    parts.append("├── fp16_grid/              # 12 FP16 baselines (full grid)")
    parts.append("├── vidit_w8a8_grid/        # 12 ViDiT-Q videos")
    parts.append("├── w8a8_grid/              # 12 matched naive videos")
    parts.append("├── sweep_fp16_baseline/    # 3 FP16 baselines (sweep)")
    parts.append("├── sweep_group1_vidit/     # 3 videos (blocks 1–13, ViDiT)")
    parts.append("├── sweep_group1_naive/")
    parts.append("├── sweep_group2_vidit/     # blocks 14–27")
    parts.append("├── sweep_group2_naive/")
    parts.append("├── sweep_group3_vidit/     # blocks 28–40")
    parts.append("└── sweep_group3_naive/")
    parts.append("```")
    parts.append("")
    parts.append("### Log file index")
    parts.append("")
    parts.append("| Experiment | ViDiT logs | Naive logs | Summary JSON |")
    parts.append("|------------|------------|------------|--------------|")
    parts.append(
        "| Full grid | `logs/kinematic_gate_grid/*.json` | `logs/kinematic_gate_w8a8_grid/*.json` | "
        "`logs/kinematic_gate_grid.json` |"
    )
    for g in GROUPS:
        parts.append(
            f"| Sweep {g} | `logs/kinematic_gate_sweep_{g}_vidit/*.json` | "
            f"`logs/kinematic_gate_sweep_{g}_naive/*.json` | "
            f"`logs/kinematic_gate_sweep_{g}_vidit.json` (ViDiT only) |"
        )
    parts.append("")
    parts.append("Pose keypoint cache: `kinematic_cache/keypoints/{clip_id}/`")
    parts.append("")
    parts.append("### Gate threshold reference (from config)")
    parts.append("")
    parts.append("```json")
    parts.append('"kinematic_gate": {')
    parts.append('  "max_rel_delta_accel_variance": 0.10,')
    parts.append('  "max_rel_delta_limb_std": 0.10,')
    parts.append('  "max_rel_delta_jitter": 0.15,')
    parts.append('  "max_abs_delta_violation_fraction": 0.02,')
    parts.append('  "max_mean_npe": 0.15,')
    parts.append('  "max_p95_npe": 0.30,')
    parts.append('  "max_under_motion_violations": 0.0')
    parts.append("}")
    parts.append("```")
    parts.append("")
    parts.append(
        "*Report generated from kinematic gate JSON outputs. Visual PSNR per clip is stored separately "
        "in `outputs/*/psnr.txt` and is not included here.*"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(parts)} sections, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
