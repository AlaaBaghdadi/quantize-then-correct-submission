"""CPU tests for the Quantize-then-Correct code (no GPU, no DynamiCtrl weights, no OpenPose).

Run from the repository root:  python -m pytest -q tests
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "quant"))
sys.path.insert(0, str(ROOT / "phase1"))

from qtc import units as QU  # noqa: E402
from qtc.w4a8 import (W4A8Linear, apply_w4a8, fake_quant_act_int8_per_token,  # noqa: E402
                      quantize_weight_int4_groupwise)
from qtc import precorrect as PC  # noqa: E402

SAMPLE = ROOT / "tests" / "data" / "sample_smoothed_keypoints.json"


# ----------------------------------------------------------------------------- units
def test_core_matrix_is_120_videos():
    us = QU.units_for("core")
    assert len(us) == 120
    c = Counter((u.variant, u.drive) for u in us)
    assert c == {("fp16", "raw"): 24, ("vidit", "raw"): 24, ("vidit", "pre"): 24, ("w4a8", "raw"): 24,
                 ("w4a8", "pre"): 24}
    assert us[0].variant == "fp16"                                     # references rendered first


def test_uids_match_article2_revision():
    sys.path.insert(0, str(ROOT / "revision"))
    import units as RU                                                  # Article 2 registry
    a2 = RU.fp16("x", "human2_motion1", seed=7).uid
    assert QU.Unit("fp16", "human2_motion1", 7).uid == a2
    assert QU.Unit("vidit", "human2_motion1", 42).uid == RU.q("x", "vidit", "full", "own", "human2_motion1").uid
    assert QU.Unit("w4a8", "human2_motion1", 42, "pre").uid.endswith("-dpre3")


def test_pairs_reference_same_seed_fp16():
    ps = QU.pairs("core")
    tags = Counter(t for *_, t in ps)
    assert tags["vidit_raw"] == tags["vidit_pre"] == tags["w4a8_raw"] == tags["w4a8_pre"] == 24
    assert tags["fp16_seed"] == 12
    for c, r, t in ps:
        if t != "fp16_seed":
            assert r.startswith("fp16-none-na-") and c.split("-")[3:5] == r.split("-")[3:5]


# ----------------------------------------------------------------------------- W4A8
def test_int4_groupwise_grid_and_error():
    torch.manual_seed(0)
    w = torch.randn(64, 300)
    deq, sc = quantize_weight_int4_groupwise(w, 128)
    assert deq.shape == w.shape and sc.shape == (64, 3)
    # every de-quantised value is an integer multiple (in [-8, 7]) of its group scale
    for g in range(3):
        blk = deq[:, g * 128:(g + 1) * 128] / sc[:, g:g + 1]
        assert torch.allclose(blk, blk.round(), atol=1e-4)
        assert blk.min() >= -8 - 1e-4 and blk.max() <= 7 + 1e-4
    rel = (deq - w).norm() / w.norm()
    assert 0.02 < rel < 0.2                                              # typical RTN-g128 INT4 error


def test_int8_per_token_activation():
    x = torch.randn(2, 5, 64) * torch.linspace(0.1, 10, 64)
    xq = fake_quant_act_int8_per_token(x)
    s = x.abs().amax(-1, keepdim=True) / 127
    assert torch.allclose(xq / s, (xq / s).round(), atol=1e-3)
    assert ((xq - x).abs() <= s / 2 + 1e-6).all()


def test_w4a8_linear_close_to_fp_and_smoothquant_invariant():
    torch.manual_seed(1)
    lin = nn.Linear(256, 128)
    x = torch.randn(4, 16, 256)
    x[..., 7] *= 40                                                      # an outlier channel
    ref = lin(x)
    s = (x.abs().amax(dim=(0, 1)) ** 0.5 / lin.weight.abs().amax(0) ** 0.5).clamp(min=1e-5)
    q_s = W4A8Linear(copy.deepcopy(lin), s, 128)(x)
    q_n = W4A8Linear(copy.deepcopy(lin), None, 128)(x)
    e_s = ((q_s - ref).norm() / ref.norm()).item()
    e_n = ((q_n - ref).norm() / ref.norm()).item()
    assert e_s < 0.25 and e_n < 0.35
    assert e_s < e_n                                                     # SmoothQuant helps with outliers


def test_apply_w4a8_replaces_named_layers(tmp_path):
    torch.manual_seed(2)
    net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 16))
    x = torch.randn(3, 32)
    h = net[0](x)
    stats = {"0": {"early": {"channel_absmax": x.abs().amax(0), "tensor_absmax": float(x.abs().max())}},
             "2": {"early": {"channel_absmax": torch.nn.functional.gelu(h).abs().amax(0),
                             "tensor_absmax": 1.0}}}
    cp = tmp_path / "calib.pt"
    torch.save({"meta": {"format": "vidit_v1"}, "stats": stats}, cp)
    ref = net(x)
    m = apply_w4a8(net, {"0", "2"}, cp, group_size=16)
    assert isinstance(net[0], W4A8Linear) and isinstance(net[2], W4A8Linear)
    assert m["w4a8_layers"] == 2 and m["w4a8_packed_weight_bytes"] < m["w4a8_dense_weight_bytes"] / 3
    out = net(x)
    assert ((out - ref).norm() / ref.norm()) < 0.3


# ----------------------------------------------------------------------------- pre-correction
def _toy_pose(n_people=1):
    body = np.random.rand(18 * n_people, 2)
    return {"bodies": {"candidate": body.copy(), "subset": np.arange(18 * n_people).reshape(n_people, 18),
                       "score": np.ones((n_people, 18))},
            "hands": np.random.rand(2 * n_people, 21, 2), "hands_score": np.ones((2 * n_people, 21)),
            "faces": np.random.rand(n_people, 68, 2), "faces_score": np.ones((n_people, 68))}


def test_apply_delta_moves_body_hands_face():
    p = _toy_pose(2)
    before = copy.deepcopy(p)
    d = np.zeros((18, 2))
    d[PC.OPENPOSE18.index("left_wrist")] = [0.01, -0.02]
    d[PC.OPENPOSE18.index("right_wrist")] = [-0.03, 0.0]
    d[PC.OPENPOSE18.index("nose")] = [0.005, 0.005]
    PC.apply_delta_to_pose(p, d)
    assert np.allclose(p["bodies"]["candidate"][:18] - before["bodies"]["candidate"][:18], d)
    assert np.allclose(p["bodies"]["candidate"][18:], before["bodies"]["candidate"][18:])     # person 1 untouched
    assert np.allclose(p["hands"][0] - before["hands"][0], [0.01, -0.02])                     # left hand, person 0
    assert np.allclose(p["hands"][2] - before["hands"][2], [-0.03, 0.0])                      # right hand, person 0
    assert np.allclose(p["hands"][1], before["hands"][1]) and np.allclose(p["hands"][3], before["hands"][3])
    assert np.allclose(p["faces"][0] - before["faces"][0], 0.005)


def test_correction_field_from_kapt_on_sample(tmp_path):
    from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
    src = json.loads(SAMPLE.read_text())
    for f in src["frames"]:                                             # emulate the pipeline's raw field
        f["keypoints_averaged"] = copy.deepcopy(f["keypoints"])
        f["keypoints_averaged"]["left_elbow"]["x"] += 3.0
    sm = tmp_path / "sm.json"
    sm.write_text(json.dumps(src))
    kp = tmp_path / "kapt.json"
    run_kapt_correction(sm, kp, force=True)
    kapt = json.loads(kp.read_text())
    delta, st = PC.correction_field(src, kapt, 36)
    assert delta.shape == (36, 18, 2) and np.isfinite(delta).all()
    assert st["joints_set"] > 0 and np.hypot(delta[..., 0], delta[..., 1]).max() <= PC.CAP + 1e-9
    other = [j for j, n in enumerate(PC.OPENPOSE18) if n not in PC.CORRECTED_JOINTS]
    assert np.all(delta[:, other] == 0)                                  # v2: arms only
    le = PC.OPENPOSE18.index("left_elbow")
    w = src["image_size"]["width"]
    # raw elbow was shifted +3 px, so the field must pull it back by ~3/W on average (plus KAPT's own change)
    assert np.median(delta[:, le, 0]) < -1.0 / w


# ----------------------------------------------------------------------------- scoring plumbing
SMOOTHNET_SRC = Path(__import__("os").environ.get("SMOOTHNET_ROOT", "/root/w/SmoothNet"))


@pytest.mark.skipif(not (SMOOTHNET_SRC / "lib" / "models" / "smoothnet.py").is_file(), reason="SmoothNet source absent")
def test_refine_and_gate_plumbing(tmp_path):
    """KAPT -> SmoothNet (random weights: plumbing only) -> metric vector -> gate verdicts."""
    sys.path.insert(0, str(SMOOTHNET_SRC))
    from lib.models.smoothnet import SmoothNet
    from experiments.smoothnet_baseline import config as C
    from kinematic_gate.compare import GateThresholds, compute_deltas, evaluate_gate
    from kinematic_gate.metrics import (compute_joint_movement_summary_between, compute_metric_vector,
                                        load_keypoints_json, strip_internal_fields)
    from qtc.refine import kapt_smoothnet
    torch.manual_seed(0)
    m = SmoothNet(window_size=32, output_size=32, hidden_size=C.HIDDEN_SIZE, res_hidden_size=C.RES_HIDDEN_SIZE,
                  num_blocks=C.NUM_BLOCKS, dropout=C.DROPOUT).eval()
    post = kapt_smoothnet(SAMPLE, tmp_path / "post.json", model=m)
    data = json.loads(post.read_text())
    assert data["refinement"]["method"].startswith("Kalman -> KAPT -> SmoothNet")
    assert len(data["frames"]) == len(json.loads(SAMPLE.read_text())["frames"])
    ref = strip_internal_fields(compute_metric_vector(SAMPLE, phase1_root=ROOT / "phase1"))
    cand = strip_internal_fields(compute_metric_vector(post, phase1_root=ROOT / "phase1"))
    mov = compute_joint_movement_summary_between(load_keypoints_json(SAMPLE), load_keypoints_json(post))
    d = compute_deltas(ref, cand, movement_summary=mov)
    ok, checks = evaluate_gate(ref, cand, d, GateThresholds())
    assert set(checks) >= {"acceleration_variance", "violation_fraction", "mean_NPE"}
    assert all(np.isfinite(v) for v in d.values())


# ----------------------------------------------------------------------------- statistics
def test_stats_helpers():
    from qtc.analyze import boot_ci_median, cliffs_delta, holm, wilson
    assert cliffs_delta(np.array([2, 3, 4]), np.array([0, 1, 1])) == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    lo, hi = wilson(5, 12)
    assert 0.15 < lo < 0.42 < hi < 0.7
    lo, hi = boot_ci_median(np.r_[np.ones(20), 3 * np.ones(4)])
    assert lo <= 1.0 <= hi


def test_analyze_end_to_end_on_synthetic_results(tmp_path):
    """Synthetic results tree -> analyze.main(): every table and figure is produced."""
    from qtc import analyze as A
    rng = np.random.default_rng(0)
    runs = tmp_path / "runs"
    for d in ("meta", "qtc/metrics", "qtc/gate", "qtc/recon"):
        (runs / d).mkdir(parents=True)
    damage = {"fp16": 1.0, "vidit": 1.2, "w4a8": 1.8}
    lv = {}
    for u in QU.units_for("core"):
        base = damage[u.variant] * (0.8 if u.drive == "pre" else 1.0)
        for post in QU.POSTS:
            f = base * (0.6 if post != "none" else 1.0) * rng.lognormal(0, 0.1)
            rec = {"uid": u.uid, "post": post, "acceleration_variance": 10 * f, "mean_limb_length_std": 5 * f,
                   "var_accel_norm": 2 * f, "violation_fraction": min(1, 0.05 * f)}
            lv[(u.uid, post)] = rec
            (runs / "qtc/metrics" / f"{u.uid}__{post}.json").write_text(json.dumps(rec))
        (runs / "meta" / f"{u.uid}.json").write_text(json.dumps({"uid": u.uid, "gen_seconds": 200.0, "peak_vram_gb": 30,
            "dit_params": 5.77e9, "int8_param_fraction": 0.9 if u.variant != "fp16" else 0.0,
            "w4a8_packed_weight_bytes": 2.9e9 if u.variant == "w4a8" else None,
            "w4a8_dense_weight_bytes": 10.4e9 if u.variant == "w4a8" else None}))
    for c, r, t in QU.pairs("core"):
        for post in QU.POSTS:
            a, b = lv[(c, post)], lv[(r, post)]
            d = {k: abs(a[m] - b[m]) / b[m] for k, m in (("rel_delta_acceleration_variance", "acceleration_variance"),
                 ("rel_delta_mean_limb_length_std", "mean_limb_length_std"), ("rel_delta_var_accel_norm", "var_accel_norm"))}
            d.update(signed_delta_violation_fraction=a["violation_fraction"] - b["violation_fraction"],
                     mean_NPE=float(rng.uniform(0.02, 0.2)), p95_NPE=float(rng.uniform(0.05, 0.4)))
            passed = d["rel_delta_acceleration_variance"] < 0.1
            (runs / "qtc/gate" / f"{c}__vs__{r}__{post}.json").write_text(json.dumps(
                {"cand": c, "ref": r, "tag": t, "post": post, "passed": passed, "deltas": d,
                 "checks": {"acceleration_variance": passed, "finite_x": True}}))
        (runs / "qtc/recon" / f"{c}__vs__{r}.json").write_text(json.dumps(
            {"cand": c, "ref": r, "psnr_db": 25.0, "ssim": 0.8, "lpips": float(rng.uniform(0.05, 0.3)),
             "tlpips": 0.01, "tmse": 0.001}))
    out = tmp_path / "out"
    sys.argv = ["analyze", "--runs", str(runs), "--out", str(out)]
    A.main()
    for f in ("noise_floor.csv", "absolute_levels.tex", "paired_tests.csv", "paired_tests.tex", "pass_rates.tex",
              "pixel_vs_kinematic_arms.csv", "frontier.csv"):
        assert (out / "tables" / f).is_file(), f
    for f in ("deltas_by_arm", "cost_plausibility_frontier", "kinematic_vs_pixel"):
        assert (out / "figures" / f"{f}.png").is_file() and (out / "figures" / f"{f}.pdf").is_file()
    t = __import__("pandas").read_csv(out / "tables" / "paired_tests.csv")
    h1 = t[(t.family == "H1_damage_w4a8") & (t.metric == "acceleration_variance")].iloc[0]
    assert h1.median_diff > 0 and h1.p_holm < 0.01                      # synthetic W4A8 damage is detected
    h3 = t[(t.family == "H3_post_w4a8") & (t.metric == "acceleration_variance")].iloc[0]
    assert h3.median_diff < 0 and h3.p_holm < 0.01                      # synthetic post-correction gain detected
