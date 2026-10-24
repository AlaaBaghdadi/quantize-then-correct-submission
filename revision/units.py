"""[REV-R1] Registry of every generation unit and every gating pair used in the revision.

A *unit* is one generated video. A *pair* is (candidate video, FP16 reference video) that is
scored by the kinematic gate (gate_worker.py) and by reconstruction metrics (recon_worker.py).

Naming: uid = f"{variant}-{map}-{calib}-{clip}-s{seed}-t{steps}-r{rep}"
  variant : fp16 | vidit | naive
  map     : none (fp16) | full | lhyb (legacy hybrid as published: blocks 14-40 INT8, K layers NOT kept)
            | nhyb (nested hybrid: P0 U G1, i.e. full-model protected set incl. K, plus blocks 1-13)
            | iso1 | iso2 | iso3 (only that block group INT8, as the published sweep)
            | c<spec> (custom maps for WI-7, e.g. c1-6 = FP16 blocks 1-6 + K, INT8 elsewhere)
  calib   : own (per-clip file from that clip's seed-42 FP16 run, the published protocol)
            | loso (merged statistics of the other subjects' clips; WI-5) | na (fp16 / naive-without-calib)
Legacy videos already on Drive are registered with uid prefix "L-" and a fixed path.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

W = Path("/workspace")
RUNS = W / "runs"
DRIVE = W / "drive"

SUBJECTS = ["human1", "human2", "human3", "human4"]
HELDOUT = "human5"
MOTIONS = ["motion1", "motion2", "motion3"]
GROUPS = {"iso1": (1, 13), "iso2": (14, 27), "iso3": (28, 40)}
NEW_SEEDS = [7, 123]


def cid(subject: str, motion: str) -> str:
    return f"{subject}_{motion}"


def clips(subjects=SUBJECTS):
    return [cid(s, m) for s in subjects for m in MOTIONS]


@dataclass
class Unit:
    wi: str
    variant: str
    map: str
    calib: str
    clip: str
    seed: int = 42
    steps: int = 25
    rep: int = 0
    collect_calib: bool = False
    fp16_blocks: list = field(default_factory=list)  # custom maps only
    keep_k: bool = True                               # custom maps only

    @property
    def uid(self) -> str:
        return f"{self.variant}-{self.map}-{self.calib}-{self.clip}-s{self.seed}-t{self.steps}-r{self.rep}"

    @property
    def video(self) -> Path:
        return RUNS / "videos" / f"{self.uid}.mp4"

    @property
    def meta(self) -> Path:
        return RUNS / "meta" / f"{self.uid}.json"

    def quant_key(self) -> str:
        """Units sharing this key share one quantized transformer (only the seed differs)."""
        return f"{self.variant}-{self.map}-{self.calib}-{self.clip}-{self.fp16_blocks}-{self.keep_k}"

    def fp16_ref_uid(self, rep: int = 0) -> str:
        return f"fp16-none-na-{self.clip}-s{self.seed}-t{self.steps}-r{rep}"

    def to_json(self):
        d = asdict(self)
        d["uid"] = self.uid
        return d


def fp16(wi, clip, seed=42, steps=25, rep=0, collect_calib=False):
    return Unit(wi, "fp16", "none", "na", clip, seed, steps, rep, collect_calib)


def q(wi, variant, map_, calib, clip, seed=42, steps=25, **kw):
    if variant == "naive" and map_ in ("lhyb", "iso1", "iso2", "iso3"):
        calib = "na"  # the published naive hybrid / isolates use no calibration statistics
    return Unit(wi, variant, map_, calib, clip, seed, steps, **kw)


# ----------------------------------------------------------------------------- legacy (Drive)
LEGACY_DIRS = {
    # family -> (drive subdir, filename prefix, steps)
    "fp16_grid": ("fp16_grid", "fp16", 25),
    "naive_full": ("w8a8_grid", "w8a8", 25),
    "vidit_lhyb": ("hybrid_vidit_w8a8_grid", "hybrid_vidit_w8a8", 25),
    "naive_lhyb": ("hybrid_w8a8_grid", "hybrid_w8a8", 25),
    "sweep_fp16": ("sweep_fp16_baseline", "fp16", 10),
    "vidit_iso1": ("sweep_group1_vidit", "vidit_w8a8", 10),
    "vidit_iso2": ("sweep_group2_vidit", "vidit_w8a8", 10),
    "vidit_iso3": ("sweep_group3_vidit", "vidit_w8a8", 10),
    "naive_iso1": ("sweep_group1_naive", "w8a8", 10),
    "naive_iso2": ("sweep_group2_naive", "w8a8", 10),
    "naive_iso3": ("sweep_group3_naive", "w8a8", 10),
}


def legacy_path(family: str, clip: str) -> Path:
    sub, prefix, _ = LEGACY_DIRS[family]
    return DRIVE / "outputs" / sub / clip / f"{prefix}_{clip}_seed42.mp4"


def legacy_uid(family: str, clip: str) -> str:
    return f"L-{family}-{clip}"


def legacy_items():
    out = {}
    for fam in LEGACY_DIRS:
        cl = clips(["human2"]) if fam.startswith("sweep") or "iso" in fam else clips()
        for c in cl:
            out[legacy_uid(fam, c)] = legacy_path(fam, c)
    return out


# ----------------------------------------------------------------------------- work items
def wi2():
    """Regenerate the missing full-grid ViDiT-Q family (published protocol)."""
    return [q("WI2", "vidit", "full", "own", c) for c in clips()]


def wi3():
    """FP16 noise floor: rep0 (reference for all RunPod seed-42 pairs) + two re-renders."""
    u = [fp16("WI3", c, rep=r) for c in clips() for r in (0, 1, 2)]
    return u


def wi4():
    """Multi-seed replication (seed 42 re-rendered on RunPod + seeds 7, 123)."""
    u = []
    for c in clips():
        for s in [42] + NEW_SEEDS:
            if s != 42:
                u.append(fp16("WI4", c, seed=s))
            for v in ("vidit", "naive"):
                for m in ("full", "lhyb", "nhyb"):
                    if s == 42 and (m, v) in (("full", "vidit"),):
                        continue  # produced by WI2
                    u.append(q("WI4", v, m, "own", c, seed=s))
    return u


def wi5():
    """Held-out (leave-one-subject-out) calibration."""
    u = []
    for c in clips():
        for v in ("vidit", "naive"):
            for m in ("full", "nhyb"):
                u.append(q("WI5", v, m, "loso", c))
        u.append(q("WI5", "vidit", "lhyb", "loso", c))
    return u


def wi6():
    """25-step sweep on S1-S4 + held-out human5; human5 full/hybrid; LOSO group selection offline."""
    u = []
    for c in clips(SUBJECTS + [HELDOUT]):
        for g in GROUPS:
            for v in ("vidit", "naive"):
                u.append(q("WI6", v, g, "own", c))
    for c in clips([HELDOUT]):
        u.append(fp16("WI6", c, collect_calib=True))
        for v in ("vidit", "naive"):
            for m in ("full", "lhyb", "nhyb"):
                u.append(q("WI6", v, m, "own", c))
    return u


def wi7(spec_file: Path | None = None):
    """Second loop iteration / variant-specific maps. Maps are written after WI-6 is analysed
    (revision/wi7_maps.json: list of {variant, name, fp16_blocks, keep_k})."""
    spec_file = spec_file or Path(__file__).with_name("wi7_maps.json")
    if not spec_file.is_file():
        return []
    u = []
    for m in json.loads(spec_file.read_text()):
        for c in clips():
            u.append(Unit("WI7", m["variant"], m["name"], "own", c, 42, 25,
                          fp16_blocks=m["fp16_blocks"], keep_k=m.get("keep_k", True)))
    return u


WI = {"WI2": wi2, "WI3": wi3, "WI4": wi4, "WI5": wi5, "WI6": wi6, "WI7": wi7}


def all_units(wis):
    seen, out = set(), []
    for w in wis:
        for u in WI[w]():
            if u.uid not in seen:
                seen.add(u.uid)
                out.append(u)
    # FP16 references first (every quantized unit needs its rep0 FP16 of the same seed)
    out.sort(key=lambda u: (u.variant != "fp16", u.quant_key(), u.seed))
    return out


def required_fp16(units):
    """FP16 rep0 references implied by quantized units but not listed explicitly."""
    have = {u.uid for u in units}
    extra = []
    for u in units:
        if u.variant != "fp16":
            r = fp16(u.wi, u.clip, u.seed, u.steps)
            if r.uid not in have:
                have.add(r.uid)
                extra.append(r)
    return extra


# ----------------------------------------------------------------------------- gating pairs
def pairs():
    """Every (candidate, reference, tag) scored by the gate and recon workers.
    Paths are resolved by gate_worker (RunPod uid -> runs/videos, L- uid -> Drive)."""
    P = []
    L = legacy_uid
    # (a) WI-0 equivalence: published pairs, exactly as in the manuscript
    for c in clips():
        P.append((L("naive_full", c), L("fp16_grid", c), "pub_full"))
        P.append((L("vidit_lhyb", c), L("fp16_grid", c), "pub_hyb"))
        P.append((L("naive_lhyb", c), L("fp16_grid", c), "pub_hyb"))
    for c in clips(["human2"]):
        for fam in ("vidit_iso1", "vidit_iso2", "vidit_iso3", "naive_iso1", "naive_iso2", "naive_iso3",
                    "vidit_lhyb", "naive_lhyb"):
            P.append((L(fam, c), L("sweep_fp16", c), "pub_sweep"))
    # (b) FP16 noise floor
    for c in clips():
        r0 = fp16("x", c).uid
        P.append((fp16("x", c, rep=1).uid, r0, "fp16_rerun"))
        P.append((fp16("x", c, rep=2).uid, r0, "fp16_rerun"))
        P.append((fp16("x", c, rep=2).uid, fp16("x", c, rep=1).uid, "fp16_rerun"))
        P.append((r0, L("fp16_grid", c), "fp16_platform"))
        for s in NEW_SEEDS:
            P.append((fp16("x", c, seed=s).uid, r0, "fp16_seed"))
    for c in clips(["human2"]):
        P.append((L("sweep_fp16", c), L("fp16_grid", c), "fp16_legacy_10v25"))
    # (c) every RunPod quantized unit vs its RunPod FP16 rep0 of the same seed/steps
    for w in WI:
        for u in WI[w]():
            if u.variant != "fp16":
                P.append((u.uid, u.fp16_ref_uid(), "main"))
                if u.seed == 42 and u.steps == 25 and u.clip.split("_")[0] in SUBJECTS:
                    P.append((u.uid, L("fp16_grid", u.clip), "vs_legacy_fp16"))
    # (d) legacy videos also re-scored against the RunPod FP16 (platform sensitivity)
    for c in clips():
        for fam in ("naive_full", "vidit_lhyb", "naive_lhyb"):
            P.append((L(fam, c), fp16("x", c).uid, "legacy_vs_runpod_fp16"))
    out, seen = [], set()
    for p in P:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            out.append(p)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wi", default="WI2,WI3,WI4,WI5,WI6,WI7")
    a = ap.parse_args()
    us = all_units(a.wi.split(","))
    us += required_fp16(us)
    from collections import Counter
    print("units:", len(us), Counter(u.wi for u in us))
    print("pairs:", len(pairs()), Counter(t for _, _, t in pairs()))
