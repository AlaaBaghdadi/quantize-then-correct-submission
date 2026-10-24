"""Export the leave-one-clip-out (LOCO) gate thresholds, one row per left-out clip, post mode and check.

The threshold for clip c and check k is the maximum of the BF16 seed-pair deltas of the other clips
(qtc.revision_stats.loco_pass). Input: paper/data/gate_long.csv. Output: revision/ijies-r1/stats/loco_thresholds.csv.
Deterministic; no random draws.
"""
from pathlib import Path
import pandas as pd
from qtc.revision_stats import load, gate_arrays, loco_pass, CHECKS, POSTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    g, *_ = load(ROOT / "paper" / "data")
    clips = sorted(g["clip"].unique())
    rows = []
    for post in POSTS:
        S, _ = gate_arrays(g, post, clips)
        _, tau = loco_pass(S, S, list(range(len(clips))))
        for i, c in enumerate(clips):
            for j, k in enumerate(CHECKS):
                rows.append({"post": post, "left_out_clip": c, "check": k,
                             "seed_pair_delta": S[i, j], "loco_threshold": tau[i, j]})
    out = ROOT / "revision" / "ijies-r1" / "stats" / "loco_thresholds.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
