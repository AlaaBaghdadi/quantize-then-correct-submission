#!/usr/bin/env python3
"""IJIES 20266618, R1 revision: CPU re-analysis of the submitted data (no new GPU runs).

    python -m qtc.revision_stats [--data paper/data] [--out revision/ijies-r1/stats]

Inputs (committed, produced by qtc.paper_stats from branch results/wo0ny5hy0vo2vt):
    paper/data/gate_long.csv      gate deltas per pair (quantized vs same-seed BF16 reference; BF16 s7 vs s42)
    paper/data/absolute_long.csv  absolute kinematic statistics per video and post mode
    paper/data/pixel_long.csv     pixel metrics per pair
    paper/data/assoc_long.csv     gate deltas joined with LPIPS (the 96 rows of the Spearman analysis)
    paper/data/numbers.json       numbers quoted in the submitted manuscript (reproduction check)

Reviewer comments addressed:
    R1.3 / R2.1  clustering: clip-level aggregation, hierarchical (clip, seed) bootstrap, clip-stratified
                 permutation, mixed-effects models with a clip random intercept, for every Mann-Whitney,
                 Wilcoxon and Spearman test of the manuscript.
    R1.4 / R2.1  equivalence: margins = 0.5 x median BF16 seed-to-seed delta per statistic, TOST, 90% and
                 95% intervals on the between-condition differences.
    R1.5 / R2.4  LOCO gate stability: threshold structure, clip bootstrap, calibration-size subsampling,
                 all 924 6/6 calibration/evaluation splits, seed-reference swap, verdict flip rates.
    R1.6         every analysis repeated without the pilot clip human2_motion1 (and, for the
                 pre-correction contrasts, without all motion-1 clips, whose correction field was the one
                 tuned on the pilot).
    R1.9         pass/rejection count of the calibrated gate on the 12 seed pairs.

The arm labelled "fp16" in the data files is the bfloat16 reference (R1.1); it is written BF16 here.
All random draws use numpy Generator(PCG64) seeded with SEED.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261012
B_BOOT = 10_000      # hierarchical bootstrap replicates
N_PERM = 20_000      # Monte-Carlo permutations (stratified label permutation)
B_GATE = 5_000       # clip bootstrap of the LOCO calibration
B_SUB = 2_000        # calibration-size subsampling draws
ALPHA = 0.05
MARGIN_FRAC = 0.5    # equivalence margin = MARGIN_FRAC x median seed-to-seed delta
PILOT = "human2_motion1"

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ["accel", "limb", "jitter", "viol", "npe", "npe95", "under"]
CHECK_LABEL = {"accel": "Accel. var. delta", "limb": "Limb-length s.d. delta", "jitter": "Jitter delta",
               "viol": "Violation delta", "npe": "Mean NPE", "npe95": "NPE95", "under": "Under-motion"}
KIN = ["acceleration_variance", "mean_limb_length_std", "var_accel_norm", "violation_fraction"]
KIN_LABEL = {"acceleration_variance": "Accel. var.", "mean_limb_length_std": "Limb-length s.d.",
             "var_accel_norm": "Jitter", "violation_fraction": "Violation fraction"}
SEEDS = [42, 7]
POSTS = ["none", "kapt_smoothnet"]
ARMS = [("vidit", "raw"), ("w4a8", "raw"), ("vidit", "pre"), ("w4a8", "pre")]
ARM_LABEL = {"vidit_raw": "W8A8 raw", "w4a8_raw": "W4A8 raw", "vidit_pre": "W8A8 pre", "w4a8_pre": "W4A8 pre"}

warnings.filterwarnings("ignore")


# ============================================================================ helpers
def rng_for(*key) -> np.random.Generator:
    """Independent, reproducible stream per analysis cell (order-independent)."""
    h = int(hashlib.sha256(("|".join(map(str, key))).encode()).hexdigest()[:8], 16)
    return np.random.Generator(np.random.PCG64([SEED, h]))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * p[i]))
        out[i] = run
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def q(x, lo, hi):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, lo)), float(np.quantile(x, hi))] if len(x) else [float("nan")] * 2


def pfmt(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def safe_wilcoxon(d):
    d = np.asarray(d, float)
    if len(d) < 5 or np.all(d == 0):
        return float("nan")
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def safe_mwu(x, y):
    z = np.concatenate([x, y])
    if np.all(z == z[0]):
        return float("nan")
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def mixedlm(df: pd.DataFrame, formula: str, term: str, vc_seed: bool = False):
    """Linear mixed model with a clip random intercept (optionally a seed-within-clip variance component).
    Returns estimate, 95% CI, p and the convergence flag for `term`."""
    import statsmodels.formula.api as smf
    try:
        kw = {"vc_formula": {"seed": "0 + C(seed)"}} if vc_seed else {}
        m = smf.mixedlm(formula, df, groups=df["clip"], **kw)
        r, last, used = None, None, None
        for meth in ("lbfgs", "powell", "nm", "bfgs"):       # first optimiser that returns a fit
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = m.fit(reml=True, method=meth)
                if np.isfinite(r.pvalues[term]):
                    used = meth
                    break
            except Exception as e:  # noqa: BLE001
                last = e
                r = None
        if r is None:
            raise last if last else RuntimeError("no fit")
        ci = r.conf_int().loc[term]
        return {"est": float(r.params[term]), "ci95": [float(ci[0]), float(ci[1])],
                "p": float(r.pvalues[term]), "converged": bool(r.converged), "optimizer": used,
                "clip_var": float(r.cov_re.iloc[0, 0]) if r.cov_re.size else float("nan")}
    except Exception as e:  # noqa: BLE001
        return {"est": float("nan"), "ci95": [float("nan")] * 2, "p": float("nan"), "converged": False,
                "error": str(e)[:120]}


def positive(*arrs):
    return all(np.all(np.asarray(a, float) > 0) for a in arrs)


# ============================================================================ data
def clip_of(uid):
    m = re.search(r"(human\d_motion\d)-s(\d+)", uid)
    return m.group(1), int(m.group(2))


def load(data: Path):
    g = pd.read_csv(data / "gate_long.csv")
    a = pd.read_csv(data / "absolute_long.csv")
    px = pd.read_csv(data / "pixel_long.csv")
    px[["clip", "seed"]] = px["cand"].map(clip_of).apply(pd.Series)
    asc = pd.read_csv(data / "assoc_long.csv")
    nums = json.loads((data / "numbers.json").read_text())
    return g, a, px, asc, nums


def seed_floor(g, post, clips):
    s = g[(g.tag == "fp16_seed") & (g.post == post)].set_index("clip").loc[clips]
    return s


def quant_matrix(df, clips, col):
    """clips x seeds [42, 7] matrix of `col`."""
    p = df.pivot_table(index="clip", columns="seed", values=col, aggfunc="first")
    return p.loc[clips, SEEDS].to_numpy(float)


# ============================================================================ R1.3 / R1.4: quantized delta vs seed delta
def unpaired_vs_seed(Q, S, key, logscale):
    """Q: clips x 2 (quantized values), S: clips (seed-pair values). Original test and clustered variants."""
    n = len(S)
    x, y = Q.ravel(), S
    out = {"n_clips": n, "median_quant": float(np.median(x)), "median_seed": float(np.median(y)),
           "diff_median": float(np.median(x) - np.median(y)),
           "p_mwu_naive": safe_mwu(x, y)}
    # clip-level aggregation: mean over the two seeds, then paired (clip-matched) and unpaired tests
    qc = Q.mean(axis=1)
    out["p_clip_wilcoxon_paired"] = safe_wilcoxon(qc - S)
    out["p_clip_mwu"] = safe_mwu(qc, S)
    out["clip_diff_median"] = float(np.median(qc - S))
    # clip-stratified permutation: within each clip the three values (2 quantized + 1 seed) are exchangeable
    rng = rng_for("perm", key)
    V = np.column_stack([Q, S])                          # n x 3, column 2 = seed
    obs = np.median(Q) - np.median(S)
    j = rng.integers(0, 3, size=(N_PERM, n))             # which column is labelled "seed"
    seedv = V[np.arange(n)[None, :], j]                  # N x n
    allv = np.broadcast_to(V, (N_PERM, n, 3))
    mask = np.ones((N_PERM, n, 3), bool)
    mask[np.arange(N_PERM)[:, None], np.arange(n)[None, :], j] = False
    quantv = allv[mask].reshape(N_PERM, 2 * n)
    null = np.median(quantv, axis=1) - np.median(seedv, axis=1)
    out["p_perm_stratified"] = float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1))
    # hierarchical bootstrap (clips, then seeds within clip) of median(quant) - median(seed)
    rng = rng_for("boot", key)
    ci = rng.integers(0, n, size=(B_BOOT, n))
    si = rng.integers(0, 2, size=(B_BOOT, n, 2))
    qb = Q[ci[:, :, None], si].reshape(B_BOOT, 2 * n)
    sb = S[ci]
    db = np.median(qb, axis=1) - np.median(sb, axis=1)
    out["ci95_diff_median"] = q(db, 0.025, 0.975)
    out["ci90_diff_median"] = q(db, 0.05, 0.95)
    # mixed model: value ~ quantized + (1 | clip); log scale when all values are positive
    rows = [{"clip": c, "seed": sd, "quant": 1, "y": Q[i, k]} for i, c in enumerate(range(n)) for k, sd in enumerate(SEEDS)]
    rows += [{"clip": i, "seed": 0, "quant": 0, "y": S[i]} for i in range(n)]
    md = pd.DataFrame(rows)
    use_log = logscale and positive(md.y)
    md["yt"] = np.log(md.y) if use_log else md.y
    mm = mixedlm(md, "yt ~ quant", "quant")
    mm["scale"] = "log" if use_log else "linear"
    if use_log:
        mm["ratio"] = float(np.exp(mm["est"]))
    out["mixed"] = mm
    return out


def tost_clip(D, margin):
    """Paired t-TOST on clip-level contrasts D (one per clip) against +/- margin."""
    D = np.asarray(D, float)
    n = len(D)
    m, se = D.mean(), D.std(ddof=1) / np.sqrt(n)
    if not np.isfinite(margin) or margin <= 0 or se == 0:
        return {"p_tost": float("nan"), "p_noninf": float("nan"), "mean": float(m), "ci90_mean": [float("nan")] * 2,
                "ci95_mean": [float("nan")] * 2}
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    t90, t95 = stats.t.ppf(0.95, n - 1), stats.t.ppf(0.975, n - 1)
    return {"p_tost": float(max(p_lo, p_hi)), "p_noninf": float(p_hi), "mean": float(m),
            "ci90_mean": [float(m - t90 * se), float(m + t90 * se)],
            "ci95_mean": [float(m - t95 * se), float(m + t95 * se)]}


def equivalence_vs_seed(Q, S, res, margin):
    """Equivalence of quantized and seed-pair deltas: margin on the difference of medians (bootstrap) and on
    the clip-level mean difference (t-TOST)."""
    lo, hi = res["ci90_diff_median"]
    eq_boot = bool(np.isfinite(margin) and margin > 0 and lo > -margin and hi < margin)
    t = tost_clip(Q.mean(axis=1) - S, margin)
    need = max(abs(lo), abs(hi))
    return {"margin": float(margin), "ci90_diff_median": [lo, hi], "ci95_diff_median": res["ci95_diff_median"],
            "equivalent_boot90": eq_boot, "tost_t": t,
            "noninferior_boot90": bool(np.isfinite(margin) and margin > 0 and hi < margin),
            "noninferior_t": bool(np.isfinite(t["p_noninf"]) and t["p_noninf"] < ALPHA),
            "equivalent_tost": bool(np.isfinite(t["p_tost"]) and t["p_tost"] < ALPHA),
            "margin_needed_boot90": float(need),
            "margin_needed_over_median_seed": float(need / np.median(S)) if np.median(S) > 0 else float("nan")}


def analysis_vs_seed(g, px, clips, tagset="main"):
    """Every Mann-Whitney comparison of the manuscript: quantized arm vs BF16 seed pairs (post none) for the
    seven gate quantities and for PSNR / LPIPS."""
    out = {}
    for v, drv in ARMS:
        arm = f"{v}_{drv}"
        for post in ["none", "kapt_smoothnet"]:
            S_df = seed_floor(g, post, clips)
            s = g[(g.variant == v) & (g.drive == drv) & (g.post == post) & g["clip"].isin(clips)]
            for k in CHECKS:
                Q = quant_matrix(s, clips, k)
                S = S_df[k].to_numpy(float)
                key = (tagset, arm, post, k)
                r = unpaired_vs_seed(Q, S, key, logscale=k not in ("viol", "under"))
                med_seed = float(np.median(S))
                r["equivalence"] = equivalence_vs_seed(Q, S, r, MARGIN_FRAC * med_seed)
                out[f"{arm}|{post}|{k}"] = r
        # pixel metrics (no post mode)
        fs = px[(px.tag == "fp16_seed") & px["clip"].isin(clips)].set_index("clip").loc[clips]
        s = px[(px.tag == arm) & px["clip"].isin(clips)]
        for m in ["psnr_db", "lpips"]:
            Q = quant_matrix(s, clips, m)
            S = fs[m].to_numpy(float)
            r = unpaired_vs_seed(Q, S, (tagset, arm, "pix", m), logscale=True)
            out[f"{arm}|pixel|{m}"] = r
    return out


# ============================================================================ R1.3: paired Wilcoxon families H1-H4
def paired_family_defs(a):
    def sel(**kw):
        d = a
        for k_, v_ in kw.items():
            d = d[d[k_] == v_]
        return d
    base = sel(variant="fp16", drive="raw", post="none")
    fams = {}
    for v in ("vidit", "w4a8"):
        fams[f"H1_damage_{v}"] = (sel(variant=v, drive="raw", post="none"), base)
        fams[f"H2_pre_{v}"] = (sel(variant=v, drive="pre", post="none"), sel(variant=v, drive="raw", post="none"))
        fams[f"H3_post_{v}"] = (sel(variant=v, drive="raw", post="kapt_smoothnet"), sel(variant=v, drive="raw", post="none"))
        fams[f"H4_combined_{v}"] = (sel(variant=v, drive="pre", post="kapt_smoothnet"), base)
        fams[f"H1post_{v}"] = (sel(variant=v, drive="raw", post="kapt_smoothnet"),
                               sel(variant="fp16", drive="raw", post="kapt_smoothnet"))
    return fams


def analysis_paired(a, clips, tagset="main"):
    out = {}
    for fam, (A_, B_) in paired_family_defs(a).items():
        rows = []
        for k in KIN:
            QA = quant_matrix(A_[A_["clip"].isin(clips)], clips, k)
            QB = quant_matrix(B_[B_["clip"].isin(clips)], clips, k)
            D = QA - QB                                   # clips x seeds
            n = len(clips)
            r = {"family": fam, "metric": k, "n_pairs": int(D.size), "n_clips": n,
                 "median_a": float(np.median(QA)), "median_b": float(np.median(QB)),
                 "median_diff": float(np.median(D)), "p_naive": safe_wilcoxon(D.ravel())}
            Dc = D.mean(axis=1)
            r["p_clip"] = safe_wilcoxon(Dc)
            # exact clip-level sign-flip permutation; statistic = median of all pairwise differences
            signs = np.array(list(itertools.product([1, -1], repeat=n)))      # 2^n x n
            null = np.median((signs[:, :, None] * D[None, :, :]).reshape(len(signs), -1), axis=1)
            obs = np.median(D)
            r["p_perm_clipflip"] = float(np.mean(np.abs(null) >= abs(obs) - 1e-12)) if np.any(D != 0) else float("nan")
            rng = rng_for("pboot", tagset, fam, k)
            ci = rng.integers(0, n, size=(B_BOOT, n))
            si = rng.integers(0, 2, size=(B_BOOT, n, 2))
            db = np.median(D[ci[:, :, None], si].reshape(B_BOOT, -1), axis=1)
            r["ci95_median_diff"] = q(db, 0.025, 0.975)
            # mixed model on the paired contrast: log ratio for positive statistics, difference otherwise
            use_log = k != "violation_fraction" and positive(QA, QB)
            Dt = np.log(QA / QB) if use_log else D
            md = pd.DataFrame([{"clip": i, "seed": sd, "d": Dt[i, j]} for i in range(n) for j, sd in enumerate(SEEDS)])
            if np.allclose(md.d, 0):
                mm = {"est": 0.0, "ci95": [0.0, 0.0], "p": float("nan"), "converged": False, "error": "all zero"}
            else:
                mm = mixedlm(md, "d ~ 1", "Intercept")
            mm["scale"] = "log ratio" if use_log else "difference"
            r["mixed"] = mm
            rows.append(r)
        for col in ["p_naive", "p_clip", "p_perm_clipflip"]:
            adj = holm([r[col] for r in rows])
            for r, p_ in zip(rows, adj):
                r[col + "_holm"] = float(p_)
        adj = holm([r["mixed"]["p"] for r in rows])
        for r, p_ in zip(rows, adj):
            r["mixed"]["p_holm"] = float(p_)
        for r in rows:
            out[f"{fam}|{r['metric']}"] = r
    return out


def equivalence_absolute(a, clips, tagset="main"):
    """R1.4: H1 (quantized vs BF16, raw drive) and the matched post-corrected H1 as equivalence questions.
    Contrast per clip = mean over seeds of log(Q/BF16) (difference for the violation fraction).
    Margin = MARGIN_FRAC x median over clips of the seed-induced change |log(BF16_s7 / BF16_s42)|."""
    out = {}
    fams = paired_family_defs(a)
    for post, fam_t in [("none", "H1_damage_{}"), ("kapt_smoothnet", "H1post_{}")]:
        F = a[(a.variant == "fp16") & (a.post == post) & a["clip"].isin(clips)]
        for k in KIN:
            Fm = quant_matrix(F, clips, k)                # clips x [42, 7]
            use_log = k != "violation_fraction" and positive(Fm)
            sd = np.abs(np.log(Fm[:, 1] / Fm[:, 0])) if use_log else np.abs(Fm[:, 1] - Fm[:, 0])
            margin = MARGIN_FRAC * float(np.median(sd))
            for v in ("vidit", "w4a8"):
                A_, B_ = fams[fam_t.format(v)]
                QA = quant_matrix(A_[A_["clip"].isin(clips)], clips, k)
                QB = quant_matrix(B_[B_["clip"].isin(clips)], clips, k)
                if use_log and not positive(QA, QB):
                    use_log = False
                Dt = np.log(QA / QB) if use_log else QA - QB
                t = tost_clip(Dt.mean(axis=1), margin)
                n = len(clips)
                rng = rng_for("eqabs", tagset, post, k, v)
                ci = rng.integers(0, n, size=(B_BOOT, n))
                si = rng.integers(0, 2, size=(B_BOOT, n, 2))
                mb = Dt[ci[:, :, None], si].reshape(B_BOOT, -1).mean(axis=1)
                c90, c95 = q(mb, 0.05, 0.95), q(mb, 0.025, 0.975)
                eqb = bool(margin > 0 and c90[0] > -margin and c90[1] < margin)
                need = max(abs(c90[0]), abs(c90[1]))
                out[f"{v}|{post}|{k}"] = {
                    "scale": "log ratio" if use_log else "difference", "margin": margin,
                    "median_seed_change": float(np.median(sd)), "mean_contrast": float(Dt.mean()),
                    "ci90_boot": c90, "ci95_boot": c95, "equivalent_boot90": eqb, "tost_t": t,
                    "noninferior_boot90": bool(margin > 0 and c90[1] < margin),
                    "noninferior_t": bool(np.isfinite(t["p_noninf"]) and t["p_noninf"] < ALPHA),
                    "equivalent_tost": bool(np.isfinite(t["p_tost"]) and t["p_tost"] < ALPHA),
                    "margin_needed_over_median_seed": float(need / np.median(sd)) if np.median(sd) > 0 else float("nan")}
    return out


# ============================================================================ R1.3: Spearman association (96 rows)
def analysis_assoc(asc, clips, tagset="main"):
    d = asc[(asc.post == "none") & asc["clip"].isin(clips)].copy()
    cid = d["clip"].map({c: i for i, c in enumerate(clips)}).to_numpy()
    n = len(clips)
    groups = [np.where(cid == i)[0] for i in range(n)]
    out = {}
    for m in ["accel", "jitter", "npe", "npe95"]:
        x, y = d["lpips"].to_numpy(float), d[m].to_numpy(float)
        rho, p = stats.spearmanr(x, y)
        r = {"n_rows": int(len(d)), "n_clips": n, "rho": float(rho), "p_naive": float(p)}
        # cluster bootstrap of rho (resample clips with all their rows)
        rng = rng_for("sp_boot", tagset, m)
        rb = np.empty(B_BOOT // 2)
        for b in range(len(rb)):
            idx = np.concatenate([groups[i] for i in rng.integers(0, n, n)])
            rb[b] = stats.spearmanr(x[idx], y[idx]).statistic
        r["ci95_cluster_boot"] = q(rb, 0.025, 0.975)
        # within-clip association: correlation of clip-centred ranks, tested by permuting y within clips
        rng = rng_for("sp_perm", tagset, m)
        rx, ry = stats.rankdata(x), stats.rankdata(y)

        def centred(v):
            v = v.astype(float).copy()
            for gi in groups:
                v[gi] -= v[gi].mean()
            return v
        cx = centred(rx)
        w_obs = float(np.corrcoef(cx, centred(ry))[0, 1])
        r["rho_within_clip"] = w_obs
        npm = N_PERM // 4
        cnt = 0
        for _ in range(npm):
            yy = ry.copy()
            for gi in groups:
                yy[gi] = ry[rng.permutation(gi)]
            cnt += abs(np.corrcoef(cx, centred(yy))[0, 1]) >= abs(w_obs) - 1e-12
        r["p_perm_within_clip"] = float((cnt + 1) / (npm + 1))
        # between-clip association: clip means (n = number of clips)
        cm = d.groupby("clip")[["lpips", m]].mean()
        rc, pc = stats.spearmanr(cm["lpips"], cm[m])
        r["rho_clip_means"], r["p_clip_means"] = float(rc), float(pc)
        # mixed model on standardised ranks with a clip random intercept
        md = pd.DataFrame({"clip": cid, "ry": stats.zscore(ry), "rx": stats.zscore(rx)})
        r["mixed"] = mixedlm(md, "ry ~ rx", "rx")
        out[m] = r
    return out


# ============================================================================ R1.5 / R2.4: LOCO gate stability
def gate_arrays(g, post, clips, alt_reference=None):
    S = seed_floor(g, post, clips)[CHECKS].to_numpy(float)     # clips x checks
    if alt_reference is not None:
        S = S.copy()
        for j, k in enumerate(CHECKS):
            if k in alt_reference:
                S[:, j] = alt_reference[k]
    evals = {}
    for v, drv in ARMS:
        s = g[(g.variant == v) & (g.drive == drv) & (g.post == post) & g["clip"].isin(clips)]
        s = s.set_index(["clip", "seed"]).loc[[(c, sd) for c in clips for sd in SEEDS]]
        evals[f"{v}_{drv}"] = (s[CHECKS].to_numpy(float), np.repeat(np.arange(len(clips)), 2))
    evals["bf16_seed"] = (S, np.arange(len(clips)))
    return S, evals


def loco_pass(S, X, cidx):
    n = S.shape[0]
    tau = np.stack([np.max(np.delete(S, c, axis=0), axis=0) for c in range(n)])   # n x checks
    return np.all(X <= tau[cidx] + 1e-12, axis=1), tau


def alt_reference_deltas(a, post, clips):
    """Seed-pair deltas recomputed with seed 7 as the reference (symmetric for the violation delta)."""
    F = a[(a.variant == "fp16") & (a.post == post)]
    P = {k: quant_matrix(F, clips, k) for k in KIN}
    return {"accel": np.abs(P["acceleration_variance"][:, 0] - P["acceleration_variance"][:, 1]) / P["acceleration_variance"][:, 1],
            "limb": np.abs(P["mean_limb_length_std"][:, 0] - P["mean_limb_length_std"][:, 1]) / P["mean_limb_length_std"][:, 1],
            "jitter": np.abs(P["var_accel_norm"][:, 0] - P["var_accel_norm"][:, 1]) / P["var_accel_norm"][:, 1],
            "viol": np.abs(P["violation_fraction"][:, 0] - P["violation_fraction"][:, 1])}


def analysis_gate(g, a, clips, tagset="main"):
    out = {}
    n = len(clips)
    for post in POSTS:
        S, evals = gate_arrays(g, post, clips)
        res = {"threshold_structure": {}, "arms": {}}
        # check the reference reconstruction used for the seed-reference swap
        alt42 = alt_reference_deltas(a, post, clips)
        # threshold structure: LOCO threshold = max of the other clips -> only the two largest values matter
        for j, k in enumerate(CHECKS):
            order = np.argsort(-S[:, j])
            vals = S[order, j]
            res["threshold_structure"][k] = {
                "largest": float(vals[0]), "largest_clip": clips[order[0]],
                "second": float(vals[1]), "second_clip": clips[order[1]],
                "ratio_largest_to_median": float(vals[0] / np.median(S[:, j])) if np.median(S[:, j]) > 0 else float("nan"),
                "distinct_loco_thresholds": int(len(np.unique([np.max(np.delete(S[:, j], c)) for c in range(n)])))}
        # clip bootstrap of the calibration set (LOCO: the evaluated clip is never in its own calibration)
        rng = rng_for("gboot", tagset, post)
        idx = rng.integers(0, n, size=(B_GATE, n))
        Sb = S[idx]                                                            # B x n x checks
        msk = idx[:, :, None] != np.arange(n)[None, None, :]                   # B x n(cal) x n(eval clip)
        tau_b = np.where(msk[..., None], Sb[:, :, None, :], -np.inf).max(axis=1)   # B x n(eval) x checks
        tau_b[~np.isfinite(tau_b)] = np.nan
        res["threshold_boot"] = {k: {"median": float(np.nanmedian(tau_b[:, :, j])),
                                     "ci95": q(tau_b[:, :, j].ravel(), 0.025, 0.975),
                                     "p_below_largest": float(np.nanmean(tau_b[:, :, j] < S[:, j].max() - 1e-12))}
                                 for j, k in enumerate(CHECKS)}
        # calibration-size subsampling (m of the 11 other clips, without replacement)
        rng = rng_for("gsub", tagset, post)
        keys = rng.random((B_SUB, n, n))
        keys[:, np.arange(n), np.arange(n)] = np.inf                            # exclude the evaluated clip
        perm = np.argsort(keys, axis=2)                                        # B x n(eval) x n(cal)
        sub_tau = {}
        for m in (4, 6, 8, 10, 11):
            sel_ = perm[:, :, :m]                                              # B x n x m
            sub_tau[m] = S[sel_].max(axis=2)                                   # B x n x checks
        # all 6/6 calibration / evaluation splits
        splits = list(itertools.combinations(range(n), n // 2))
        for arm, (X, cidx) in evals.items():
            ok0, tau0 = loco_pass(S, X, cidx)
            e = {"n": int(len(ok0)), "loco_pass": int(ok0.sum()), "loco_rate": float(ok0.mean()),
                 "wilson95": wilson(int(ok0.sum()), len(ok0))}
            # cluster bootstrap CI of the pass rate with thresholds fixed (clips, then seeds within clip)
            rng = rng_for("gpr", tagset, post, arm)
            ci_ = rng.integers(0, n, size=(B_BOOT, n))
            if arm == "bf16_seed":
                pr = ok0[ci_].mean(axis=1)
            else:
                okm = ok0.reshape(n, 2)
                si = rng.integers(0, 2, size=(B_BOOT, n, 2))
                pr = okm[ci_[:, :, None], si].reshape(B_BOOT, -1).mean(axis=1)
            e["cluster_boot_ci95"] = q(pr, 0.025, 0.975)
            # clip bootstrap of the calibration: pass-rate distribution and per-video verdict flips
            passb = np.all(X[None, :, :] <= tau_b[:, cidx, :] + 1e-12, axis=2)  # B x videos (nan -> False)
            valid = ~np.any(np.isnan(tau_b[:, cidx, :]), axis=2)
            prb = np.where(valid, passb, np.nan)
            e["calboot_rate_median"] = float(np.nanmedian(np.nanmean(prb, axis=1)))
            e["calboot_rate_ci95"] = q(np.nanmean(prb, axis=1), 0.025, 0.975)
            flip = np.nanmean(np.where(valid, passb != ok0[None, :], np.nan), axis=0)   # per video
            e["flip_rate_mean"] = float(np.mean(flip))
            e["videos_flip_gt_10pct"] = int(np.sum(flip > 0.10))
            e["videos_flip_gt_25pct"] = int(np.sum(flip > 0.25))
            e["flip_rate_by_video"] = {f"{clips[c]}" + ("" if arm == "bf16_seed" else f"-s{SEEDS[i % 2]}"): float(fv)
                                       for i, (c, fv) in enumerate(zip(cidx, flip))}
            # calibration size
            e["calsize"] = {}
            for m, tb in sub_tau.items():
                pm = np.all(X[None, :, :] <= tb[:, cidx, :] + 1e-12, axis=2).mean(axis=1)
                e["calsize"][m] = {"mean": float(pm.mean()), "ci95": q(pm, 0.025, 0.975)}
            # held-out 6/6 splits: calibrate on 6 seed pairs, evaluate on the other 6 clips
            rates = []
            for cal in splits:
                ev = np.setdiff1d(np.arange(n), cal)
                tau = S[list(cal)].max(axis=0)
                sel_ = np.isin(cidx, ev)
                rates.append(np.all(X[sel_] <= tau + 1e-12, axis=1).mean())
            rates = np.array(rates)
            e["split66"] = {"n_splits": len(splits), "mean": float(rates.mean()), "median": float(np.median(rates)),
                            "ci95": q(rates, 0.025, 0.975), "_rates": rates}
            res["arms"][arm] = e
        # quantized minus seed pass rate across the same 6/6 splits
        for arm in list(evals):
            if arm == "bf16_seed":
                continue
            d = res["arms"][arm]["split66"]["_rates"] - res["arms"]["bf16_seed"]["split66"]["_rates"]
            res["arms"][arm]["split66"]["diff_vs_seed_mean"] = float(d.mean())
            res["arms"][arm]["split66"]["diff_vs_seed_ci95"] = q(d, 0.025, 0.975)
            res["arms"][arm]["split66"]["frac_splits_quant_ge_seed"] = float(np.mean(d >= 0))
        for arm in evals:
            res["arms"][arm]["split66"].pop("_rates", None)
        # seed-reference swap (seed 7 as reference for the relative checks)
        S2, evals2 = gate_arrays(g, post, clips, alt_reference=alt42)
        res["ref_swap"] = {arm: {"loco_pass": int(loco_pass(S2, X, c)[0].sum()), "n": int(len(c))}
                           for arm, (X, c) in evals2.items()}
        out[post] = res
    return out


def analysis_r19(g, clips):
    S, evals = gate_arrays(g, "none", clips)
    X, c = evals["bf16_seed"]
    ok, tau = loco_pass(S, X, c)
    failed = {}
    for i in np.where(~ok)[0]:
        failed[clips[i]] = [k for j, k in enumerate(CHECKS) if X[i, j] > tau[i, j] + 1e-12]
    return {"n": int(len(ok)), "n_pass": int(ok.sum()), "n_reject": int((~ok).sum()),
            "pass_rate": float(ok.mean()), "reject_rate": float((~ok).mean()), "rejected": failed}


# ============================================================================ conclusions
def verdict(paper_sig: bool, clustered: list[float]):
    ok = [p for p in clustered if p is not None and np.isfinite(p)]
    if not ok:
        return "not testable"
    agree = [(p < ALPHA) == paper_sig for p in ok]
    return "holds" if all(agree) else ("does not hold" if not any(agree) else "partly holds")


def build_conclusions(R):
    """Map each inferential statement of the submitted manuscript to its clustered re-analysis."""
    C = []
    vs = R["main"]["vs_seed"]
    for arm in ("vidit_raw", "w4a8_raw"):
        for k in ("accel", "jitter", "limb", "npe", "npe95"):
            r = vs[f"{arm}|none|{k}"]
            sig = r["p_mwu_naive"] < ALPHA
            C.append({"claim": f"{ARM_LABEL[arm]}: {CHECK_LABEL[k]} vs seed pairs ({'differs' if sig else 'no difference'}; MWU p={pfmt(r['p_mwu_naive'])})",
                      "verdict": verdict(sig, [r["p_clip_wilcoxon_paired"], r["p_perm_stratified"], r["mixed"]["p"]]),
                      "p_clip": r["p_clip_wilcoxon_paired"], "p_perm": r["p_perm_stratified"], "p_mixed": r["mixed"]["p"]})
        for m in ("psnr_db", "lpips"):
            r = vs[f"{arm}|pixel|{m}"]
            sig = r["p_mwu_naive"] < ALPHA
            C.append({"claim": f"{ARM_LABEL[arm]}: {m} vs seed pairs (MWU p={pfmt(r['p_mwu_naive'])})",
                      "verdict": verdict(sig, [r["p_clip_wilcoxon_paired"], r["p_perm_stratified"], r["mixed"]["p"]]),
                      "p_clip": r["p_clip_wilcoxon_paired"], "p_perm": r["p_perm_stratified"], "p_mixed": r["mixed"]["p"]})
    for arm in ("vidit_pre", "w4a8_pre"):
        for m in ("psnr_db", "lpips"):
            r = vs[f"{arm}|pixel|{m}"]
            sig = r["p_mwu_naive"] < ALPHA
            C.append({"claim": f"{ARM_LABEL[arm]}: {m} vs seed pairs (MWU p={pfmt(r['p_mwu_naive'])})",
                      "verdict": verdict(sig, [r["p_clip_wilcoxon_paired"], r["p_perm_stratified"], r["mixed"]["p"]]),
                      "p_clip": r["p_clip_wilcoxon_paired"], "p_perm": r["p_perm_stratified"], "p_mixed": r["mixed"]["p"]})
        r = vs[f"{arm}|none|npe"]
        sig = r["p_mwu_naive"] < ALPHA
        C.append({"claim": f"{ARM_LABEL[arm]}: Mean NPE vs seed pairs (MWU p={pfmt(r['p_mwu_naive'])})",
                  "verdict": verdict(sig, [r["p_clip_wilcoxon_paired"], r["p_perm_stratified"], r["mixed"]["p"]]),
                  "p_clip": r["p_clip_wilcoxon_paired"], "p_perm": r["p_perm_stratified"], "p_mixed": r["mixed"]["p"]})
    for key, r in R["main"]["paired"].items():
        sig = r["p_naive_holm"] < ALPHA
        C.append({"claim": f"{r['family']}: {KIN_LABEL[r['metric']]} ({'differs' if sig else 'no difference'}; Wilcoxon Holm p={pfmt(r['p_naive_holm'])})",
                  "verdict": verdict(sig, [r["p_clip_holm"], r["p_perm_clipflip_holm"], r["mixed"]["p_holm"]]),
                  "p_clip": r["p_clip_holm"], "p_perm": r["p_perm_clipflip_holm"], "p_mixed": r["mixed"]["p_holm"]})
    for m, r in R["main"]["assoc"].items():
        sig = r["p_naive"] < ALPHA
        C.append({"claim": f"Spearman LPIPS vs {CHECK_LABEL[m]} (rho={r['rho']:.2f}, p={pfmt(r['p_naive'])})",
                  "verdict": verdict(sig, [r["p_perm_within_clip"], r["mixed"]["p"]] +
                                     ([0.0] if (r['ci95_cluster_boot'][0] > 0 or r['ci95_cluster_boot'][1] < 0) else [1.0])),
                  "p_clip": r["p_clip_means"], "p_perm": r["p_perm_within_clip"], "p_mixed": r["mixed"]["p"]})
    return C


# ============================================================================ report
def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def reproduce(R, nums, g, clips):
    """Check that the committed CSVs reproduce the submitted numbers.json."""
    chk = []
    for arm in ("vidit_raw", "w4a8_raw"):
        for k in ("accel", "jitter", "limb", "npe", "npe95"):
            new = R["main"]["vs_seed"][f"{arm}|none|{k}"]["p_mwu_naive"]
            old = nums["arms"][f"{arm}_none"]["vs_seed_noise"][k]["p"]
            chk.append((f"MWU p {arm} {k}", old, new))
    for k_, v_ in nums["arms"].items():
        arm, post = k_.rsplit("_", 1) if not k_.endswith("kapt_smoothnet") else (k_[:-15], "kapt_smoothnet")
        new = R["main"]["gate"][post]["arms"][arm]["loco_pass"]
        chk.append((f"LOCO pass {k_}", v_["pass_loco"], new))
    for post in POSTS:
        chk.append((f"LOCO self {post}", nums["noise_floor_loco_self"][post]["n_pass"],
                    R["main"]["gate"][post]["arms"]["bf16_seed"]["loco_pass"]))
    for m in ("accel", "jitter", "npe", "npe95"):
        chk.append((f"Spearman rho {m}", nums["assoc"][f"lpips_{m}"]["rho"], R["main"]["assoc"][m]["rho"]))
    for key, r in R["main"]["paired"].items():
        fam, met = key.split("|")
        if fam.startswith("H1post"):
            v = fam.split("_")[1]
            old = nums["paired_post_matched"][f"{v}|{met}"]["p_holm"]
        else:
            old = nums["paired"][f"{fam}|{met}"]["p_holm"]
        chk.append((f"Wilcoxon Holm p {key}", old, r["p_naive_holm"]))
    rows = []
    for name, o, n_ in chk:
        o, n_ = float(o), float(n_)
        same = abs(o - n_) < 1e-9 or (not np.isfinite(o) and not np.isfinite(n_))
        dec = ((o < ALPHA) == (n_ < ALPHA)) if name.startswith(("MWU", "Wilcoxon")) else same
        rows.append((name, o, n_, same, dec))
    return all(r[3] for r in rows), rows


def table(rows, header):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def ci_s(c, nd=3):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]" if np.all(np.isfinite(c)) else "n/a"


def write_report(R, out: Path, md_path: Path, prov: dict):
    M = R["main"]
    L = []
    L.append("# IJIES 20266618 - R1 revision: re-analysis of the submitted data (CPU)\n")
    L.append("All numbers below are generated by `qtc/revision_stats.py` and must not be edited by hand. "
             "Re-run: `python -m qtc.revision_stats`.\n")
    L.append("## Provenance\n")
    L.append(table([[k, f"`{v}`"] for k, v in prov.items()], ["Item", "Value"]))
    L.append("\nThe data files were produced by `qtc/paper_stats.py` from branch `results/wo0ny5hy0vo2vt` "
             "(submitted version). The arm labelled `fp16` in the data is the bfloat16 reference and is written "
             "BF16 here (R1.1). W8A8 = `vidit`, W4A8 = `w4a8`; raw / pre = driving pose without / with pre-correction; "
             "post none / kapt_smoothnet = skeleton without / with post-correction.\n")
    L.append(f"Settings: seed {SEED}; hierarchical bootstrap {B_BOOT} replicates (clips with replacement, then "
             f"seeds within clip with replacement); stratified permutation {N_PERM} draws; cluster bootstrap of "
             f"Spearman {B_BOOT // 2}; within-clip Spearman permutation {N_PERM // 4}; LOCO calibration bootstrap "
             f"{B_GATE}; calibration-size subsampling {B_SUB}; alpha {ALPHA}.\n")
    nm = [r for r in R["repro"] if not r[3]]
    L.append(f"\nReproduction check of the submitted `numbers.json` from the committed CSVs: "
             f"{len(R['repro']) - len(nm)} of {len(R['repro'])} quantities reproduced exactly "
             f"(`stats/reproduction_check.csv`).")
    if nm:
        L.append(" The remaining quantities are Wilcoxon p-values of families that contain the violation "
                 "fraction, whose paired differences include zeros. With zero_method='wilcox' (zeros dropped, "
                 f"normal approximation) the installed SciPy {__import__('scipy').__version__} returns different "
                 "p-values for those differences from the SciPy version used for the submission; through the Holm "
                 "step-down this also changes other p-values of the same family. "
                 f"{'No' if all(r[4] for r in nm) else 'At least one'} decision at alpha = 0.05 changes. "
                 "The revised manuscript uses the recomputed values.\n\n")
        L.append(table([[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", "same" if r[4] else "CHANGED"] for r in nm],
                       ["Quantity", "Submitted", "Recomputed", "Decision at 0.05"]))
    L.append("\n")

    # ---- R1.3
    L.append("\n## R1.3 / R2.1 - clustered re-analysis\n")
    L.append("Methods. (a) Clip-level aggregation: the two seeds of each clip are averaged, giving 12 clip-level "
             "observations; the quantized-vs-seed contrasts are then clip-matched and tested with the Wilcoxon "
             "signed-rank test (n = 12). (b) Permutation: for quantized-vs-seed contrasts, the arm label is "
             "permuted within each clip among its three values (two quantized deltas, one seed delta), statistic = "
             "difference of medians; for paired contrasts, the sign of the difference is flipped per clip (both "
             "seeds together), exact enumeration of 2^12 assignments, statistic = median difference. "
             "(c) Linear mixed model with a clip random intercept (statsmodels MixedLM, REML), on the log scale for "
             "positive quantities (log ratio for paired contrasts) and on the raw scale otherwise. For Spearman: "
             "cluster bootstrap of rho over clips, within-clip permutation of the kinematic delta, Spearman on clip "
             "means (n = 12), and a mixed model on standardised ranks.\n")
    L.append("\n### Quantized delta vs BF16 seed-pair delta (Mann-Whitney family, post none)\n")
    rows = []
    for arm in ("vidit_raw", "w4a8_raw", "vidit_pre", "w4a8_pre"):
        for k in CHECKS + ["psnr_db", "lpips"]:
            key = f"{arm}|pixel|{k}" if k in ("psnr_db", "lpips") else f"{arm}|none|{k}"
            r = M["vs_seed"][key]
            mm = r["mixed"]
            est = (f"x{mm['ratio']:.2f}" if mm.get("scale") == "log" and np.isfinite(mm["est"]) else f3(mm["est"]))
            rows.append([ARM_LABEL[arm], CHECK_LABEL.get(k, k), f3(r["median_quant"]), f3(r["median_seed"]),
                         pfmt(r["p_mwu_naive"]), pfmt(r["p_clip_wilcoxon_paired"]), pfmt(r["p_perm_stratified"]),
                         ci_s(r["ci95_diff_median"]), est, pfmt(mm["p"])])
    L.append(table(rows, ["Arm", "Quantity", "Median quant.", "Median seed", "p MWU (submitted)",
                          "p clip Wilcoxon", "p strat. perm.", "95% CI diff. medians (hier. boot.)",
                          "Mixed effect", "p mixed"]))
    L.append("\nMixed effect: multiplicative ratio quantized/seed on the log scale (xN) or additive difference.\n")

    L.append("\n### Paired Wilcoxon families (H1-H4, matched post-corrected H1)\n")
    rows = []
    for key, r in M["paired"].items():
        mm = r["mixed"]
        est = (f"x{np.exp(mm['est']):.2f}" if mm["scale"] == "log ratio" and np.isfinite(mm["est"]) else f3(mm["est"]))
        rows.append([r["family"], KIN_LABEL[r["metric"]], f3(r["median_diff"]), pfmt(r["p_naive_holm"]),
                     pfmt(r["p_clip_holm"]), pfmt(r["p_perm_clipflip_holm"]), ci_s(r["ci95_median_diff"]),
                     est, pfmt(mm.get("p_holm"))])
    L.append(table(rows, ["Family", "Metric", "Median diff.", "p Holm (submitted, n=24)", "p Holm clip (n=12)",
                          "p Holm clip sign-flip", "95% CI median diff. (hier. boot.)", "Mixed effect", "p Holm mixed"]))

    L.append("\n### Spearman association, LPIPS vs kinematic delta (96 quantization pairs, post none)\n")
    rows = []
    for m, r in M["assoc"].items():
        rows.append([CHECK_LABEL[m], f"{r['rho']:.3f}", pfmt(r["p_naive"]), ci_s(r["ci95_cluster_boot"]),
                     f"{r['rho_within_clip']:.3f}", pfmt(r["p_perm_within_clip"]), f"{r['rho_clip_means']:.3f}", pfmt(r["p_clip_means"]),
                     f3(r["mixed"]["est"]), pfmt(r["mixed"]["p"])])
    L.append(table(rows, ["Delta", "rho", "p (submitted)", "95% CI rho (cluster boot.)", "rho within clip (centred ranks)", "p within-clip perm.",
                          "rho clip means (n=12)", "p clip means", "Mixed slope (std. ranks)", "p mixed"]))

    L.append("\n### Status of each inferential statement\n")
    L.append("`holds`: every clustered method reaches the same decision at alpha = 0.05 as the submitted test; "
             "`partly holds`: the methods disagree; `does not hold`: no clustered method reproduces the decision.\n\n")
    L.append(table([[c["claim"], f"**{c['verdict']}**", pfmt(c["p_clip"]), pfmt(c["p_perm"]), pfmt(c["p_mixed"])]
                    for c in R["conclusions"]],
                   ["Submitted statement", "Status", "p clip-level", "p permutation", "p mixed"]))

    # ---- R1.4
    L.append("\n## R1.4 / R2.1 - equivalence\n")
    L.append(f"Margin rule (proposed): the equivalence margin of each statistic is {MARGIN_FRAC} x the median "
             "BF16 seed-to-seed delta of that statistic over the 12 clips. Justification: the seed-to-seed delta is "
             "the change produced by the unquantized model when only the sampling noise changes; a quantization "
             "effect smaller than half of the typical sampling effect is not of practical consequence for a "
             "per-video acceptance decision, whereas a difference of the order of the sampling effect itself "
             "would be. The margin is fixed from the BF16 data before any quantized data enter the contrast. "
             "Decision: equivalence is declared when the 90% interval of the contrast lies inside +/- margin "
             "(equivalent to two one-sided tests at alpha = 0.05). Two contrasts are reported: the difference "
             "of medians with a hierarchical bootstrap interval, and the clip-level mean difference with a "
             "paired t-TOST (n = 12). `Margin needed` is the half-width the margin would need for the 90% "
             "bootstrap interval to fit, as a multiple of the median seed delta. Non-inferiority (R1.4 alternative): "
             "the quantized quantity is not larger than the reference by more than the margin, i.e. the upper "
             "limit of the 90% interval is below +margin (one-sided test at alpha = 0.05); larger deltas, larger "
             "absolute acceleration variance, jitter, limb-length variability and violation fraction are the "
             "unfavourable direction. The Under-motion check (integer count, median seed delta 0) has no "
             "defined margin and is not tested.\n")
    L.append("\n### Gate deltas: quantized vs seed pairs (post none)\n")
    rows = []
    for arm in ("vidit_raw", "w4a8_raw", "vidit_pre", "w4a8_pre"):
        for k in CHECKS:
            r = M["vs_seed"][f"{arm}|none|{k}"]
            e = r["equivalence"]
            if k == "under":
                continue
            rows.append([ARM_LABEL[arm], CHECK_LABEL[k], f3(e["margin"]), f3(r["diff_median"]),
                         ci_s(e["ci90_diff_median"]), ci_s(e["ci95_diff_median"]),
                         "yes" if e["equivalent_boot90"] else "no", "yes" if e["noninferior_boot90"] else "no",
                         ci_s(e["tost_t"]["ci90_mean"]), pfmt(e["tost_t"]["p_tost"]),
                         "yes" if e["equivalent_tost"] else "no", "yes" if e["noninferior_t"] else "no",
                         f"{e['margin_needed_over_median_seed']:.2f}"])
    L.append(table(rows, ["Arm", "Quantity", "Margin", "Diff. medians", "90% CI", "95% CI", "Equiv. (boot)",
                          "Non-inf. (boot)", "90% CI clip mean diff.", "p TOST", "Equiv. (TOST)", "Non-inf. (t)",
                          "Margin needed / median seed"]))
    L.append("\nNPE, NPE95 and the pixel measures of the raw-drive arms are lower than the seed level "
             "(previous table); for these quantities the relevant statement is a difference, not equivalence.\n")
    L.append("\n### Absolute kinematic statistics: quantized vs BF16 (H1, raw drive)\n")
    L.append("Contrast per clip: mean over seeds of log(quantized / BF16) (difference for the violation fraction). "
             "Margin: 0.5 x median over clips of |log(BF16 s7 / BF16 s42)| (|difference| for the violation fraction).\n\n")
    rows = []
    for key, e in M["eq_abs"].items():
        v, post, k = key.split("|")
        rows.append(["W8A8" if v == "vidit" else "W4A8", post, KIN_LABEL[k], e["scale"], f3(e["margin"]),
                     f3(e["mean_contrast"]), ci_s(e["ci90_boot"]), ci_s(e["ci95_boot"]),
                     "yes" if e["equivalent_boot90"] else "no", "yes" if e["noninferior_boot90"] else "no",
                     pfmt(e["tost_t"]["p_tost"]), "yes" if e["equivalent_tost"] else "no",
                     "yes" if e["noninferior_t"] else "no", f"{e['margin_needed_over_median_seed']:.2f}"])
    L.append(table(rows, ["Arm", "Post", "Statistic", "Scale", "Margin", "Mean contrast", "90% CI (hier. boot.)",
                          "95% CI", "Equiv. (boot)", "Non-inf. (boot)", "p TOST", "Equiv. (TOST)", "Non-inf. (t)",
                          "Margin needed / median seed"]))

    # ---- R1.5
    L.append("\n## R1.5 / R2.4 - stability of the leave-one-clip-out gate\n")
    for post in POSTS:
        G = M["gate"][post]
        L.append(f"\n### Post mode: {post}\n")
        L.append("Threshold structure. The LOCO threshold of a check is the maximum of the seed deltas of the other "
                 "11 clips; it therefore takes only two values: the largest seed delta (for 11 clips) and the "
                 "second largest (for the clip that holds the largest).\n\n")
        L.append(table([[CHECK_LABEL[k], f3(t["largest"]), t["largest_clip"], f3(t["second"]), t["second_clip"],
                         f"{t['ratio_largest_to_median']:.1f}" if np.isfinite(t["ratio_largest_to_median"]) else "n/a",
                         ci_s(G["threshold_boot"][k]["ci95"]), f"{G['threshold_boot'][k]['p_below_largest']:.2f}"]
                        for k, t in G["threshold_structure"].items()],
                       ["Check", "Largest seed delta", "Clip", "Second", "Clip", "Largest / median",
                        "95% range of threshold (clip bootstrap)", "P(threshold < largest)"]))
        L.append("\nPass rates.\n\n")
        rows = []
        for arm, e in G["arms"].items():
            lab = "BF16 seed pairs" if arm == "bf16_seed" else ARM_LABEL[arm]
            rows.append([lab, f"{e['loco_pass']}/{e['n']}", f"{e['loco_rate']:.3f}", ci_s(e["wilson95"]),
                         ci_s(e["cluster_boot_ci95"]), ci_s(e["calboot_rate_ci95"]),
                         f"{e['split66']['mean']:.3f}", ci_s(e["split66"]["ci95"]),
                         ci_s(e["split66"].get("diff_vs_seed_ci95", [np.nan, np.nan])),
                         f"{e['flip_rate_mean']:.3f}", f"{e['videos_flip_gt_10pct']}/{e['n']}",
                         f"{e['videos_flip_gt_25pct']}/{e['n']}",
                         f"{G['ref_swap'][arm]['loco_pass']}/{G['ref_swap'][arm]['n']}"])
        L.append(table(rows, ["Arm", "LOCO pass", "Rate", "Wilson 95%", "Cluster boot. 95% (clip, seed)",
                              "95% range, calibration bootstrap", "Held-out 6/6 mean", "6/6 95% range",
                              "6/6 quant - seed 95% range", "Mean verdict flip rate", "Videos flip >10%",
                              "Videos flip >25%", "LOCO pass, s7 reference"]))
        L.append("\nThe calibration bootstrap can only remove or duplicate seed pairs, so its thresholds never exceed "
                 "the LOCO thresholds and its pass-rate range is bounded above by the LOCO rate. The 6/6 columns "
                 "use all 924 splits of the 12 clips into 6 calibration and 6 evaluation clips (threshold = "
                 "maximum of the 6 calibration seed pairs; quantized videos and seed pairs of the 6 evaluation "
                 "clips are scored). The flip rate is the fraction of calibration-bootstrap replicates in which a "
                 "video's verdict differs from its LOCO verdict.\n")
        L.append("\nCalibration size (m clips drawn without replacement from the 11 non-evaluated clips; mean pass "
                 "rate and 95% range).\n\n")
        ms = list(next(iter(G["arms"].values()))["calsize"].keys())
        rows = [["BF16 seed pairs" if arm == "bf16_seed" else ARM_LABEL[arm]] +
                [f"{e['calsize'][m]['mean']:.3f} {ci_s(e['calsize'][m]['ci95'], 2)}" for m in ms]
                for arm, e in G["arms"].items()]
        L.append(table(rows, ["Arm"] + [f"m = {m}" for m in ms]))

    # ---- R1.6
    L.append("\n## R1.6 - sensitivity to the pilot clip\n")
    for tag, lab in [("no_pilot", f"without {PILOT} (11 clips)"),
                     ("no_motion1", "without all motion-1 clips (8 clips; motion-1 correction field was tuned on the pilot)")]:
        X = R[tag]
        L.append(f"\n### Analyses {lab}\n")
        rows = []
        for arm in ("vidit_raw", "w4a8_raw", "vidit_pre", "w4a8_pre"):
            for k in ("accel", "jitter", "limb", "viol", "npe", "npe95"):
                a0, a1 = M["vs_seed"][f"{arm}|none|{k}"], X["vs_seed"][f"{arm}|none|{k}"]
                rows.append([ARM_LABEL[arm], CHECK_LABEL[k], pfmt(a0["p_mwu_naive"]), pfmt(a1["p_mwu_naive"]),
                             pfmt(a0["p_clip_wilcoxon_paired"]), pfmt(a1["p_clip_wilcoxon_paired"]),
                             pfmt(a1["p_perm_stratified"]), "yes" if a1["equivalence"]["equivalent_boot90"] else "no",
                             "yes" if a0["equivalence"]["noninferior_boot90"] else "no",
                             "yes" if a1["equivalence"]["noninferior_boot90"] else "no"])
        L.append(table(rows, ["Arm", "Quantity", "p MWU all", "p MWU excl.", "p clip all", "p clip excl.",
                              "p perm. excl.", "Equiv. excl. (boot)", "Non-inf. all", "Non-inf. excl."]))
        rows = []
        for key, r1 in X["paired"].items():
            r0 = M["paired"][key]
            rows.append([r1["family"], KIN_LABEL[r1["metric"]], f3(r0["median_diff"]), f3(r1["median_diff"]),
                         pfmt(r0["p_naive_holm"]), pfmt(r1["p_naive_holm"]), pfmt(r1["p_clip_holm"]),
                         pfmt(r1["p_perm_clipflip_holm"])])
        L.append("\n" + table(rows, ["Family", "Metric", "Median diff. all", "Median diff. excl.",
                                     "p Holm all", "p Holm excl.", "p Holm clip excl.", "p Holm sign-flip excl."]))
        rows = [[CHECK_LABEL[m], f"{M['assoc'][m]['rho']:.3f}", f"{r['rho']:.3f}", pfmt(r["p_naive"]),
                 ci_s(r["ci95_cluster_boot"]), pfmt(r["p_perm_within_clip"])] for m, r in X["assoc"].items()]
        L.append("\n" + table(rows, ["Delta", "rho all", "rho excl.", "p excl.", "95% CI excl. (cluster boot.)",
                                     "p within-clip perm. excl."]))
        rows = []
        for post in POSTS:
            for arm, e in X["gate"][post]["arms"].items():
                e0 = M["gate"][post]["arms"][arm]
                rows.append([post, "BF16 seed pairs" if arm == "bf16_seed" else ARM_LABEL[arm],
                             f"{e0['loco_pass']}/{e0['n']} ({e0['loco_rate']:.3f})",
                             f"{e['loco_pass']}/{e['n']} ({e['loco_rate']:.3f})", ci_s(e["wilson95"]),
                             f"{e['split66']['mean']:.3f}" if e["split66"]["n_splits"] else "n/a"])
        L.append("\n" + table(rows, ["Post", "Arm", "LOCO pass all", "LOCO pass excl.", "Wilson 95% excl.",
                                     "Held-out split mean excl."]))
        if tag == "no_motion1":
            L.append("\nWith 8 clips the smallest attainable two-sided p of the exact clip sign-flip test is "
                     f"{2 / 2 ** 8:.4f} before Holm adjustment, so loss of significance in this subset partly "
                     "reflects reduced resolution rather than a change of effect.\n")
        ch = [c for c in R["changes"][tag]]
        L.append(f"\nDecisions at alpha = 0.05 that change when these clips are excluded: "
                 f"{'none' if not ch else '; '.join(ch)}.\n")

    # ---- R1.9
    r9 = M["r19"]
    L.append("\n## R1.9 - pass and rejection counts of the calibrated gate on the seed pairs\n")
    L.append(f"With leave-one-clip-out thresholds (post none), {r9['n_pass']} of {r9['n']} BF16 seed pairs pass "
             f"(pass rate {r9['pass_rate']:.3f}) and {r9['n_reject']} of {r9['n']} are rejected (rejection rate "
             f"{r9['reject_rate']:.3f}); the rejection count is the complement of the pass count. Rejected clips "
             f"and the checks that reject them: " +
             "; ".join(f"{c} ({', '.join(v)})" for c, v in r9["rejected"].items()) + ".\n")
    old = md_path.read_text() if md_path.is_file() else ""
    m = re.search(r"<!-- GPU:BEGIN -->.*<!-- GPU:END -->\n?", old, flags=re.S)
    md_path.write_text("\n".join(L) + ("\n\n" + m.group(0) if m else ""))      # keep the GPU-run section


# ============================================================================ main
def run_all(g, a, px, asc, clips, tag):
    return {"vs_seed": analysis_vs_seed(g, px, clips, tag), "paired": analysis_paired(a, clips, tag),
            "eq_abs": equivalence_absolute(a, clips, tag), "assoc": analysis_assoc(asc, clips, tag),
            "gate": analysis_gate(g, a, clips, tag), "r19": analysis_r19(g, clips)}


def decision_changes(M, X):
    ch = []
    for key, r in X["vs_seed"].items():
        if "|none|" not in key and "|pixel|" not in key:
            continue
        for col in ("p_mwu_naive", "p_clip_wilcoxon_paired"):
            p0, p1 = M["vs_seed"][key][col], r[col]
            if np.isfinite(p0) and np.isfinite(p1) and (p0 < ALPHA) != (p1 < ALPHA):
                ch.append(f"{key} {col}: {pfmt(p0)} -> {pfmt(p1)}")
        e0, e1 = M["vs_seed"][key].get("equivalence"), r.get("equivalence")
        for col in ("equivalent_boot90", "noninferior_boot90"):
            if e0 and e1 and e0[col] != e1[col]:
                ch.append(f"{key} {col}: {e0[col]} -> {e1[col]}")
    for key, r in X["paired"].items():
        for col in ("p_naive_holm", "p_clip_holm"):
            p0, p1 = M["paired"][key][col], r[col]
            if np.isfinite(p0) and np.isfinite(p1) and (p0 < ALPHA) != (p1 < ALPHA):
                ch.append(f"{key} {col}: {pfmt(p0)} -> {pfmt(p1)}")
    for m, r in X["assoc"].items():
        p0, p1 = M["assoc"][m]["p_naive"], r["p_naive"]
        if (p0 < ALPHA) != (p1 < ALPHA):
            ch.append(f"Spearman {m}: {pfmt(p0)} -> {pfmt(p1)}")
    for key, e in X["eq_abs"].items():
        for col in ("equivalent_boot90", "noninferior_boot90"):
            if e[col] != M["eq_abs"][key][col]:
                ch.append(f"absolute {key} {col}: {M['eq_abs'][key][col]} -> {e[col]}")
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "paper" / "data"))
    ap.add_argument("--out", default=str(ROOT / "revision" / "ijies-r1" / "stats"))
    ap.add_argument("--md", default=str(ROOT / "revision" / "ijies-r1" / "RESULTS.md"))
    ap.add_argument("--quick", action="store_true", help="fewer replicates (smoke test only)")
    args = ap.parse_args()
    global B_BOOT, N_PERM, B_GATE, B_SUB
    if args.quick:
        B_BOOT, N_PERM, B_GATE, B_SUB = 400, 800, 200, 100
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g, a, px, asc, nums = load(data)
    clips = sorted(g["clip"].unique())
    R = {"main": run_all(g, a, px, asc, clips, "main")}
    R["no_pilot"] = run_all(g, a, px, asc, [c for c in clips if c != PILOT], "no_pilot")
    R["no_motion1"] = run_all(g, a, px, asc, [c for c in clips if not c.endswith("motion1")], "no_motion1")
    R["changes"] = {t: decision_changes(R["main"], R[t]) for t in ("no_pilot", "no_motion1")}
    R["conclusions"] = build_conclusions(R)
    R["repro_ok"], R["repro"] = reproduce(R, nums, g, clips)
    prov = {"script": "qtc/revision_stats.py", "script_sha256_16": sha256(Path(__file__)),
            "git_commit_at_run": git("rev-parse", "HEAD"),
            "working_tree_clean_for_script": "yes" if not git("status", "--porcelain", "qtc/revision_stats.py", "paper/data") else "no",
            **{f"input {f}": f"paper/data/{f} sha256:{sha256(data / f)}" for f in
               ["gate_long.csv", "absolute_long.csv", "pixel_long.csv", "assoc_long.csv", "numbers.json"]},
            "seed": SEED, "replicates": f"boot {B_BOOT}, perm {N_PERM}, gate {B_GATE}, sub {B_SUB}",
            "numpy/scipy/pandas": f"{np.__version__}/{__import__('scipy').__version__}/{pd.__version__}"}
    R["provenance"] = prov
    (out / "revision_stats.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    # flat CSVs
    for tag in ("main", "no_pilot", "no_motion1"):
        pd.json_normalize([{"key": k, **v} for k, v in R[tag]["vs_seed"].items()]).to_csv(out / f"vs_seed_{tag}.csv", index=False)
        pd.json_normalize([{"key": k, **v} for k, v in R[tag]["paired"].items()]).to_csv(out / f"paired_{tag}.csv", index=False)
        pd.json_normalize([{"key": k, **v} for k, v in R[tag]["eq_abs"].items()]).to_csv(out / f"equivalence_abs_{tag}.csv", index=False)
        pd.json_normalize([{"key": k, **v} for k, v in R[tag]["assoc"].items()]).to_csv(out / f"assoc_{tag}.csv", index=False)
        pd.json_normalize([{"post": p, "arm": arm, **{kk: vv for kk, vv in e.items() if kk != "flip_rate_by_video"}}
                           for p in POSTS for arm, e in R[tag]["gate"][p]["arms"].items()]).to_csv(out / f"gate_{tag}.csv", index=False)
    pd.DataFrame(R["conclusions"]).to_csv(out / "conclusions.csv", index=False)
    pd.DataFrame(R["repro"], columns=["quantity", "submitted", "recomputed", "identical", "same_decision"]).to_csv(out / "reproduction_check.csv", index=False)
    if not args.quick:
        write_report(R, out, Path(args.md), prov)
    print(json.dumps({"repro_ok": R["repro_ok"], "r19": R["main"]["r19"], "changes": R["changes"],
                      "verdicts": pd.Series([c["verdict"] for c in R["conclusions"]]).value_counts().to_dict()}, indent=1))


if __name__ == "__main__":
    main()
