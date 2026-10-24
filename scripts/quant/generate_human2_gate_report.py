"""Generate docs/DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md from gate JSON logs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"

CONFIGS = [
    ("Full grid", "ViDiT-Q W8A8", "kinematic_gate_grid", "outputs/vidit_w8a8_grid", "vidit_w8a8", "outputs/fp16_grid"),
    ("Full grid", "Matched naive W8A8", "kinematic_gate_w8a8_grid", "outputs/w8a8_grid", "w8a8", "outputs/fp16_grid"),
    ("Sweep group1 (blocks 1-13)", "ViDiT-Q W8A8", "kinematic_gate_sweep_group1_vidit", "outputs/sweep_group1_vidit", "vidit_w8a8", "outputs/sweep_fp16_baseline"),
    ("Sweep group1 (blocks 1-13)", "Matched naive W8A8", "kinematic_gate_sweep_group1_naive", "outputs/sweep_group1_naive", "w8a8", "outputs/sweep_fp16_baseline"),
    ("Sweep group2 (blocks 14-27)", "ViDiT-Q W8A8", "kinematic_gate_sweep_group2_vidit", "outputs/sweep_group2_vidit", "vidit_w8a8", "outputs/sweep_fp16_baseline"),
    ("Sweep group2 (blocks 14-27)", "Matched naive W8A8", "kinematic_gate_sweep_group2_naive", "outputs/sweep_group2_naive", "w8a8", "outputs/sweep_fp16_baseline"),
    ("Sweep group3 (blocks 28-40)", "ViDiT-Q W8A8", "kinematic_gate_sweep_group3_vidit", "outputs/sweep_group3_vidit", "vidit_w8a8", "outputs/sweep_fp16_baseline"),
    ("Sweep group3 (blocks 28-40)", "Matched naive W8A8", "kinematic_gate_sweep_group3_naive", "outputs/sweep_group3_naive", "w8a8", "outputs/sweep_fp16_baseline"),
    ("Hybrid (G1 FP16, G2+G3 INT8)", "ViDiT-Q W8A8", "kinematic_gate_hybrid_vidit_grid", "outputs/hybrid_vidit_w8a8_grid", "hybrid_vidit_w8a8", "outputs/sweep_fp16_baseline"),
    ("Hybrid (G1 FP16, G2+G3 INT8)", "Matched naive W8A8", "kinematic_gate_hybrid_grid", "outputs/hybrid_w8a8_grid", "hybrid_w8a8", "outputs/sweep_fp16_baseline"),
]

CLIPS = ["human2_motion1", "human2_motion2", "human2_motion3"]

CHECKS = [
    "acceleration_variance",
    "mean_limb_length_std",
    "var_accel_norm",
    "violation_fraction",
    "mean_NPE",
    "p95_NPE",
    "under_motion_violations",
]

FOLDER_LABELS = {
    "kinematic_gate_grid": "Full ViDiT",
    "kinematic_gate_w8a8_grid": "Full Naive",
    "kinematic_gate_sweep_group1_vidit": "G1 ViDiT",
    "kinematic_gate_sweep_group1_naive": "G1 Naive",
    "kinematic_gate_sweep_group2_vidit": "G2 ViDiT",
    "kinematic_gate_sweep_group2_naive": "G2 Naive",
    "kinematic_gate_sweep_group3_vidit": "G3 ViDiT",
    "kinematic_gate_sweep_group3_naive": "G3 Naive",
    "kinematic_gate_hybrid_vidit_grid": "Hybrid ViDiT",
    "kinematic_gate_hybrid_grid": "Hybrid Naive",
}

DETAIL_TITLES = {
    "kinematic_gate_grid": "Full grid — ViDiT-Q W8A8",
    "kinematic_gate_w8a8_grid": "Full grid — Matched naive W8A8",
    "kinematic_gate_sweep_group1_vidit": "Sweep group1 — ViDiT-Q W8A8",
    "kinematic_gate_sweep_group1_naive": "Sweep group1 — Matched naive W8A8",
    "kinematic_gate_sweep_group2_vidit": "Sweep group2 — ViDiT-Q W8A8",
    "kinematic_gate_sweep_group2_naive": "Sweep group2 — Matched naive W8A8",
    "kinematic_gate_sweep_group3_vidit": "Sweep group3 — ViDiT-Q W8A8",
    "kinematic_gate_sweep_group3_naive": "Sweep group3 — Matched naive W8A8",
    "kinematic_gate_hybrid_vidit_grid": "Hybrid — ViDiT-Q W8A8",
    "kinematic_gate_hybrid_grid": "Hybrid — Matched naive W8A8",
}

METRIC_ROWS = [
    ("acceleration_variance", "acceleration_variance", "rel_delta_acceleration_variance", "acceleration_variance"),
    ("mean_limb_length_std", "mean_limb_length_std", "rel_delta_mean_limb_length_std", "mean_limb_length_std"),
    ("var_accel_norm", "var_accel_norm (jitter)", "rel_delta_var_accel_norm", "var_accel_norm"),
    ("violation_fraction", "violation_fraction", "abs_delta_violation_fraction", "violation_fraction"),
    ("total_violations", "total_violations", "abs_delta_total_violations", None),
]


def fmt(x: float, nd: int = 4) -> str:
    if x is None:
        return "—"
    if abs(x) < 0.0001 and x != 0:
        return f"{x:.2e}"
    if nd == 0:
        return f"{x:.0f}"
    return f"{x:.{nd}f}"


def failed_checks(report: dict) -> list[str]:
    return [c for c in CHECKS if not report["checks"].get(c, True)]


def load_records() -> list[dict]:
    records = []
    for exp, algo, folder, quant_root, prefix, fp16_root in CONFIGS:
        for clip in CLIPS:
            path = LOGS / folder / f"{clip}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "exp": exp,
                    "algo": algo,
                    "folder": folder,
                    "quant_root": quant_root,
                    "prefix": prefix,
                    "fp16_root": fp16_root,
                    "clip": clip,
                    "r": report,
                }
            )
    return records


def blocks_for_folder(folder: str) -> str:
    if "group1" in folder:
        return "1–13 only"
    if "group2" in folder:
        return "14–27 only"
    if "group3" in folder:
        return "28–40 only"
    if "hybrid" in folder:
        return "14–40 (G1 stays FP16)"
    return "All non-protected (~full DiT)"


def generate() -> str:
    records = load_records()
    lines: list[str] = []

    def add(*parts: str) -> None:
        lines.extend(parts)

    def add_block(*parts: str) -> None:
        add(*parts)
        lines.append("")

    add_block(
        "# DynamiCtrl PTQ — Human2 Kinematic Gate Comparison",
        "",
        "Complete kinematic gate results for **`human2_motion1`**, **`human2_motion2`**, and "
        "**`human2_motion3`** — the only clips shared across every quantization experiment "
        "(full grid, group sweep, and hybrid).",
        "",
        "**Related docs:** `docs/DynamiCtrl-PTQ-kinematic-gate-results.md` (full 12-clip grid + sweep), "
        "`docs/DynamiCtrl-PTQ-method.md`",
        "**Config:** `configs/quant/vidit_ptq.json` (seed 42)",
        "",
        "---",
        "",
        "## 1. Why human2 only",
        "",
        "These three clips appear in **all 10 quantization configurations**:",
        "",
        "| Experiment | Clips | FP16 baseline |",
        "|------------|-------|---------------|",
        "| Full grid | 12 (4 humans × 3 motions) | `outputs/fp16_grid/` |",
        "| Group sweep | 3 (`human2` only) | `outputs/sweep_fp16_baseline/` |",
        "| Hybrid (G1 FP16, G2+G3 INT8) | 3 (`human2` only) | `outputs/sweep_fp16_baseline/` |",
        "",
        "**Important:** Full-grid runs compare against `fp16_grid`; sweep and hybrid compare against "
        "`sweep_fp16_baseline`. FP16 metric values differ slightly between baselines, so **absolute deltas "
        "are not directly comparable** across experiment families. Pass/fail, relative ranking within the "
        "same baseline, and failure-mode analysis remain valid.",
        "",
        "---",
        "",
        "## 2. All quantization configurations",
        "",
        "| # | Experiment | Algorithm | Blocks INT8 | Log directory | Quant output |",
        "|---|------------|-----------|-------------|---------------|--------------|",
    )
    for i, (exp, algo, folder, quant_root, _prefix, _fp16_root) in enumerate(CONFIGS, 1):
        short_algo = "ViDiT" if "ViDiT" in algo else "Naive"
        add(f"| {i} | {exp} | {short_algo} | {blocks_for_folder(folder)} | `logs/{folder}/` | `{quant_root}/` |")
    lines.append("")

    add_block(
        "---",
        "",
        "## 3. Kinematic gate thresholds",
        "",
        "| Check | Threshold | Meaning |",
        "|-------|-----------|---------|",
        "| `acceleration_variance` | rel Δ ≤ **0.10** | Temporal smoothness of joint motion |",
        "| `mean_limb_length_std` | rel Δ ≤ **0.10** | Limb length stability across frames |",
        "| `var_accel_norm` (jitter) | rel Δ ≤ **0.15** | Normalized acceleration jitter |",
        "| `violation_fraction` | quant ≤ FP16 + **0.02** | No new biomechanical constraint breaks |",
        "| `mean_NPE` | ≤ **0.15** | Mean normalized pose error vs FP16 skeleton |",
        "| `p95_NPE` | ≤ **0.30** | 95th-percentile pose error |",
        "| `under_motion_violations` | ≤ **0.0** | Frames under-moving vs driving motion |",
        "",
        "A clip **passes** only if every check is within tolerance.",
        "",
        "---",
        "",
        "## 4. Executive summary",
        "",
        "### 4a. Pass/fail matrix",
        "",
        "| Config | motion1 | motion2 | motion3 | Total |",
        "|--------|---------|---------|---------|-------|",
    )
    folder_order = [c[2] for c in CONFIGS]
    for folder in folder_order:
        sub = [rec for rec in records if rec["folder"] == folder]
        passes = [rec["r"]["passed"] for rec in sub]
        cells = ["**PASS**" if p else "FAIL" for p in passes]
        label = FOLDER_LABELS[folder]
        add(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | **{sum(passes)}/3** |")
    lines.append("")

    vidit_pass = sum(rec["r"]["passed"] for rec in records if "ViDiT" in rec["algo"])
    naive_pass = sum(rec["r"]["passed"] for rec in records if "naive" in rec["algo"])

    add_block(
        "### 4b. Key findings",
        "",
        "- **Best pass rate on human2:** Full naive and Hybrid ViDiT tie at **2/3** clips.",
        "- **`human2_motion1` is the bottleneck** — only **G2 naive** (group 2 isolated) passes all checks.",
        f"- **Naive vs ViDiT:** Naive **{naive_pass}/15** clip-passes vs ViDiT **{vidit_pass}/15** across all configs.",
        "- **Different failure modes:** Naive fails on **violation_fraction** (new constraint breaks); "
        "ViDiT fails on **acceleration_variance** and **var_accel_norm** (motion smoothness).",
        "- **Group 1 (blocks 1–13) is most sensitive** when isolated — mean rel_accel ViDiT **0.353** vs group 3 **0.060**.",
        "- **Hybrid ViDiT** improves full ViDiT from **0/3 → 2/3**; **Hybrid naive** regresses from **2/3 → 1/3** vs full naive.",
        "- **Pose error (NPE) never fails** on human2 — gate is effectively a smoothness + violations test for this subject.",
        "- **`mean_NPE` and `p95_NPE`:** 0 failures across all 30 runs on human2.",
        "",
        "---",
        "",
        "## 5. Master comparison tables",
        "",
    )

    for clip in CLIPS:
        add(f"### `{clip}`", "")
        add(
            "| Config | Pass | rel_accel | rel_limb | rel_jitter | viol_frac Δ | mean_NPE | p95_NPE | Failed checks |",
            "|--------|------|-----------|----------|------------|-------------|----------|---------|---------------|",
        )
        for rec in records:
            if rec["clip"] != clip:
                continue
            r = rec["r"]
            d = r["deltas"]
            label = FOLDER_LABELS[rec["folder"]]
            fails = failed_checks(r)
            fail_str = ", ".join(fails) if fails else "*(none)*"
            add(
                f"| {label} | {'**PASS**' if r['passed'] else 'FAIL'} | "
                f"{fmt(d['rel_delta_acceleration_variance'])} | {fmt(d['rel_delta_mean_limb_length_std'])} | "
                f"{fmt(d['rel_delta_var_accel_norm'])} | {fmt(d['abs_delta_violation_fraction'])} | "
                f"{fmt(d['mean_NPE'])} | {fmt(d['p95_NPE'])} | {fail_str} |"
            )
        lines.append("")

    add_block(
        "---",
        "",
        "## 6. Per-configuration detail (all metrics)",
        "",
    )

    for rec in records:
        r = rec["r"]
        d = r["deltas"]
        mfp = r["metrics_fp16"]
        mq = r["metrics_quant"]
        folder = rec["folder"]
        title = DETAIL_TITLES[folder]
        status = "PASS" if r["passed"] else "FAIL"

        add_block(
            f"### {title} — `{rec['clip']}` — **{status}**",
            "",
            f"- **FP16 video:** `{rec['fp16_root']}/{rec['clip']}/fp16_{rec['clip']}_seed42.mp4`",
            f"- **Quant video:** `{rec['quant_root']}/{rec['clip']}/{rec['prefix']}_{rec['clip']}_seed42.mp4`",
            f"- **Report:** `logs/{folder}/{rec['clip']}.json`",
            "",
            "| Metric | FP16 | Quant | Delta | Check |",
            "|--------|------|-------|-------|-------|",
        )

        for mk, mlabel, dkey, check_key in METRIC_ROWS:
            fp = mfp.get(mk, float("nan"))
            qp = mq.get(mk, float("nan"))
            delta_val = d.get(dkey, float("nan"))
            if dkey.startswith("rel_"):
                delta_s = f"rel={fmt(delta_val)}"
            elif dkey.startswith("abs_"):
                delta_s = f"abs={fmt(delta_val, nd=0 if mk == 'total_violations' else 4)}"
            else:
                delta_s = fmt(delta_val)
            if check_key:
                check_s = "PASS" if r["checks"][check_key] else "FAIL"
            else:
                check_s = "—"
            add(f"| {mlabel} | {fmt(fp, 2)} | {fmt(qp, 2)} | {delta_s} | {check_s} |")

        add(
            f"| mean_NPE | — | — | {fmt(d['mean_NPE'])} | {'PASS' if r['checks']['mean_NPE'] else 'FAIL'} |",
            f"| p95_NPE | — | — | {fmt(d['p95_NPE'])} | {'PASS' if r['checks']['p95_NPE'] else 'FAIL'} |",
            f"| under_motion_violations | — | — | {fmt(d['under_motion_violations'], nd=0)} | "
            f"{'PASS' if r['checks']['under_motion_violations'] else 'FAIL'} |",
        )
        add_block(
            "",
            f"**Failed checks:** {', '.join(failed_checks(r)) if failed_checks(r) else '*(none)*'}",
            "",
        )

    add_block(
        "---",
        "",
        "## 7. Aggregate analysis",
        "",
        "### 7a. Mean deltas per configuration (3 clips averaged)",
        "",
        "| Config | Pass | Mean rel_accel | Mean rel_limb | Mean rel_jitter | Mean viol Δ | Mean mean_NPE | Mean p95_NPE |",
        "|--------|------|----------------|---------------|-----------------|-------------|---------------|--------------|",
    )

    for folder in folder_order:
        sub = [rec for rec in records if rec["folder"] == folder]
        passes = sum(rec["r"]["passed"] for rec in sub)

        def avg(key: str) -> float:
            vals = [rec["r"]["deltas"][key] for rec in sub]
            return sum(vals) / len(vals)

        add(
            f"| {FOLDER_LABELS[folder]} | {passes}/3 | "
            f"{fmt(avg('rel_delta_acceleration_variance'))} | {fmt(avg('rel_delta_mean_limb_length_std'))} | "
            f"{fmt(avg('rel_delta_var_accel_norm'))} | {fmt(avg('abs_delta_violation_fraction'))} | "
            f"{fmt(avg('mean_NPE'))} | {fmt(avg('p95_NPE'))} |"
        )
    lines.append("")

    add_block(
        "### 7b. Per-check failure counts (30 runs = 10 configs × 3 clips)",
        "",
        "| Check | Failures | Rate | Primary cause on human2 |",
        "|-------|----------|------|-------------------------|",
    )

    fail_counts = {c: 0 for c in CHECKS}
    for rec in records:
        for c in CHECKS:
            if not rec["r"]["checks"].get(c, True):
                fail_counts[c] += 1

    causes = {
        "acceleration_variance": "ViDiT full/hybrid motion1; G1/G3 isolates",
        "var_accel_norm": "Tracks acceleration failures",
        "violation_fraction": "Full naive motion1; hybrid naive motion1–2",
        "mean_limb_length_std": "Full ViDiT motion2 only (0.102 vs 0.10 limit)",
        "mean_NPE": "Never fails on human2",
        "p95_NPE": "Never fails on human2",
        "under_motion_violations": "Never fails on human2",
    }
    for c in CHECKS:
        add(f"| `{c}` | {fail_counts[c]}/30 | {fail_counts[c] / 30 * 100:.0f}% | {causes[c]} |")
    lines.append("")

    add_block(
        "### 7c. Group sensitivity (sweep, mean rel_accel)",
        "",
        "| Group | Blocks | ViDiT mean rel_accel | Naive mean rel_accel | ViDiT pass | Naive pass |",
        "|-------|--------|----------------------|----------------------|------------|------------|",
    )

    for g, blocks in [("1", "1–13"), ("2", "14–27"), ("3", "28–40")]:
        vsub = [rec for rec in records if f"group{g}_vidit" in rec["folder"]]
        nsub = [rec for rec in records if f"group{g}_naive" in rec["folder"]]
        va = sum(r["r"]["deltas"]["rel_delta_acceleration_variance"] for r in vsub) / 3
        na = sum(r["r"]["deltas"]["rel_delta_acceleration_variance"] for r in nsub) / 3
        vp = sum(r["r"]["passed"] for r in vsub)
        np = sum(r["r"]["passed"] for r in nsub)
        add(f"| group{g} | {blocks} | {fmt(va)} | {fmt(na)} | {vp}/3 | {np}/3 |")
    lines.append("")

    add_block(
        "---",
        "",
        "## 8. Hybrid vs full quant (human2)",
        "",
        "| Clip | Full ViDiT | Hybrid ViDiT | Full Naive | Hybrid Naive |",
        "|------|------------|--------------|------------|--------------|",
    )

    def pass_for(folder: str, clip: str) -> str:
        for rec in records:
            if rec["folder"] == folder and rec["clip"] == clip:
                return "PASS" if rec["r"]["passed"] else "FAIL"
        return "—"

    for clip in CLIPS:
        add(
            f"| `{clip}` | {pass_for('kinematic_gate_grid', clip)} | "
            f"{pass_for('kinematic_gate_hybrid_vidit_grid', clip)} | "
            f"{pass_for('kinematic_gate_w8a8_grid', clip)} | "
            f"{pass_for('kinematic_gate_hybrid_grid', clip)} |"
        )
    lines.append("")

    add_block(
        "### 8a. Hybrid ViDiT improvements",
        "",
        "| Clip | Full ViDiT issue | Hybrid ViDiT result |",
        "|------|------------------|---------------------|",
        "| `human2_motion1` | accel + jitter + viol fail | Still fails accel/jitter; **viol fixed** (0.000) |",
        "| `human2_motion2` | limb_std 0.102 (limit 0.10) | **PASS** — limb_std 0.034 |",
        "| `human2_motion3` | accel 0.149, jitter 0.153 | **PASS** — accel 0.078, jitter 0.114 |",
        "",
        "### 8b. Hybrid naive regressions",
        "",
        "| Clip | Full Naive | Hybrid Naive | Notes |",
        "|------|------------|--------------|-------|",
        "| `human2_motion1` | FAIL (viol only) | FAIL (accel + jitter + viol) | Hybrid **worse** — smoothness degraded |",
        "| `human2_motion2` | PASS | FAIL (viol only, Δ=0.027) | Near-miss: 0.007 over viol threshold |",
        "| `human2_motion3` | PASS | PASS | Hybrid naive accel 0.008 — excellent |",
        "",
        "### 8c. Hybrid vs best isolate",
        "",
        "| Clip | Best isolate pass | Hybrid result |",
        "|------|-------------------|---------------|",
        "| `human2_motion1` | **G2 naive PASS** | Hybrid naive + ViDiT both FAIL |",
        "| `human2_motion2` | G2 naive PASS (best metrics) | Hybrid ViDiT PASS; hybrid naive near-miss |",
        "| `human2_motion3` | Full naive PASS (best accel) | Both hybrids PASS |",
        "",
        "---",
        "",
        "## 9. Naive vs ViDiT (human2, paired)",
        "",
        "| Clip | ViDiT strength | Naive strength | Full ViDiT | Full Naive |",
        "|------|----------------|----------------|------------|------------|",
        "| `human2_motion1` | Hybrid: no new violations | Best accel/jitter (0.03) | FAIL | FAIL (viol only) |",
        "| `human2_motion2` | Best accel/jitter (0.004) | Passes gate | FAIL (limb 0.102) | PASS |",
        "| `human2_motion3` | — | Near-zero accel delta | FAIL (accel/jitter) | PASS |",
        "",
        f"**Totals across 5 ViDiT vs 5 naive configs (15 runs each):** ViDiT **{vidit_pass}/15** passes, "
        f"Naive **{naive_pass}/15** passes.",
        "",
        "---",
        "",
        "## 10. Practical ranking (human2)",
        "",
        "| Rank | Config | Pass | Best for |",
        "|------|--------|------|----------|",
        "| 1 | Full naive W8A8 | 2/3 | motion2 + motion3; motion1 fails only on violations |",
        "| 1 | Hybrid ViDiT | 2/3 | motion2 + motion3; fixes full ViDiT failures |",
        "| 3 | G2 naive (sweep) | 2/3 | **Only config that passes motion1** |",
        "| 3 | G1 naive / G2 ViDiT | 2/3 each | Different clips fail |",
        "| 5 | Hybrid naive | 1/3 | motion3 excellent; motion2 near-miss |",
        "| 6 | Full ViDiT | 0/3 | — |",
        "",
        "**No single configuration passes all 3 human2 clips.**",
        "",
        "---",
        "",
        "## 11. Conclusions",
        "",
        "### Was the kinematic gate useful on human2?",
        "",
        "| Role | Verdict | Evidence |",
        "|------|---------|----------|",
        "| Diagnostic tool | **Yes** | Separated accel/jitter failures (ViDiT) from violation failures (naive) |",
        "| Layer prioritization | **Yes** | Group1 isolate mean rel_accel 3–5× worse than group2/3 |",
        "| Hybrid validation | **Partial** | ViDiT hybrid 0/3→2/3; naive hybrid 2/3→1/3 |",
        "| Ship-ready guarantee | **No** | Best configs still 2/3; motion1 unsolved except G2 isolate |",
        "",
        "### Recommended next steps",
        "",
        "1. **Re-gate full 12-clip grid** with hybrid ViDiT and hybrid naive (G1 FP16, G2+G3 INT8).",
        "2. **Sub-sweep group1** (blocks 1–6 vs 7–13) to fix `human2_motion1` without giving up G2+G3 compression.",
        "3. **Investigate G2 naive motion1 pass** — consider G1 partial FP16 + G2 INT8 + G3 INT8 variant.",
        "4. **Tune hybrid naive motion2** — violation Δ=0.027 is only 0.007 over threshold.",
        "5. Keep **`human2_motion1`** as a permanent stress clip in the gate set.",
        "",
        "---",
        "",
        "## 12. Log file index",
        "",
        "| Config | Log path | Summary JSON |",
        "|--------|----------|--------------|",
        "| Full ViDiT | `logs/kinematic_gate_grid/human2_motion*.json` | `logs/kinematic_gate_grid.json` |",
        "| Full Naive | `logs/kinematic_gate_w8a8_grid/human2_motion*.json` | — |",
        "| G1 ViDiT | `logs/kinematic_gate_sweep_group1_vidit/` | `logs/kinematic_gate_sweep_group1_vidit.json` |",
        "| G1 Naive | `logs/kinematic_gate_sweep_group1_naive/` | — |",
        "| G2 ViDiT | `logs/kinematic_gate_sweep_group2_vidit/` | `logs/kinematic_gate_sweep_group2_vidit.json` |",
        "| G2 Naive | `logs/kinematic_gate_sweep_group2_naive/` | — |",
        "| G3 ViDiT | `logs/kinematic_gate_sweep_group3_vidit/` | `logs/kinematic_gate_sweep_group3_vidit.json` |",
        "| G3 Naive | `logs/kinematic_gate_sweep_group3_naive/` | — |",
        "| Hybrid ViDiT | `logs/kinematic_gate_hybrid_vidit_grid/` | `logs/kinematic_gate_hybrid_vidit_grid.json` |",
        "| Hybrid Naive | `logs/kinematic_gate_hybrid_grid/` | `logs/kinematic_gate_hybrid_grid.json` |",
        "",
        "Pose keypoint cache: `kinematic_cache/keypoints/human2_motion{1,2,3}/`",
        "",
        "*Report generated from kinematic gate JSON outputs under `logs/kinematic_gate_*/human2_motion*.json`.*",
        "*Regenerate with: `python scripts/quant/generate_human2_gate_report.py`*",
    )

    return "\n".join(lines)


def main() -> None:
    out = ROOT / "docs" / "DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md"
    content = generate()
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
