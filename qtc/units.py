"""Registry of generation units and scoring pairs for the Quantize-then-Correct study (Article 3).

A *unit* is one generated video, identified by

    uid = f"{variant}-{map}-{calib}-{clip}-s{seed}-t{steps}-r{rep}" [+ "-dpre3"]

variant : fp16 | vidit (W8A8, ViDiT-Q/SmoothQuant) | w4a8 (INT4 g128 weights, INT8 per-token activations)
          | w4a8mp (fallback: blocks 1-13 W8A8, blocks 14-40 W4A8)
map     : none (fp16) | full
calib   : own (per-clip calibration file of the Article 2 protocol) | na (fp16)
drive   : raw (DWPose driving pose, the unmodified DynamiCtrl pipeline) | pre (KAPT-corrected driving pose)

The uid of a raw-drive unit is identical to the Article 2 revision uid (revision/units.py), so FP16 and W8A8
videos rendered for either study can be shared when both run on the same volume.

A *pair* (candidate, reference, tag, post) is scored by qtc/score.py. post = "none" scores the skeleton as
extracted; post = "kapt_smoothnet" applies the Article 1 KAPT -> SmoothNet refinement to BOTH videos of the pair
(matched processing), so the deltas measure whether correction brings the quantized skeleton back to the
equally processed FP16 skeleton. Absolute metric levels of every (video, post) combination are also recorded.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

W = Path(os.environ.get("W", "/workspace"))
RUNS = W / "runs"
DRIVE = W / "drive"
QTC = RUNS / "qtc"

SUBJECTS = ["human1", "human2", "human3", "human4"]
MOTIONS = ["motion1", "motion2", "motion3"]
SEEDS = [42, 7]                 # shared with the Article 2 revision (seed 42 = published protocol, 7 = WI-4)
STEPS = 25
QUANT_VARIANTS = ["vidit", "w4a8"]
POSTS = ["none", "kapt_smoothnet"]
PILOT_CLIP = "human2_motion1"
# IJIES R1 revision (R1.2 / R2.2): quantizer calibration held out from the evaluated clip
HELDOUT = "loso"                # calibration = elementwise max of the per-clip statistics of the OTHER subjects
# IJIES R1 revision (R1.5 / R2.4): additional BF16 seeds for the noise floor (reference stays seed 42)
EXTRA_SEEDS = [1, 2, 3]


def clips(subjects=SUBJECTS):
    return [f"{s}_{m}" for s in subjects for m in MOTIONS]


@dataclass(frozen=True)
class Unit:
    variant: str
    clip: str
    seed: int = 42
    drive: str = "raw"
    map: str = "full"
    steps: int = STEPS
    rep: int = 0
    group: str = "core"          # core | optional | pilot | heldout | seeds
    calib_mode: str = "own"      # own (per-clip, submitted version) | loso (held-out subjects, revision)

    @property
    def calib(self) -> str:
        return "na" if self.variant == "fp16" else self.calib_mode

    @property
    def collect_calib(self) -> bool:
        """Article 2 protocol: a clip's ViDiT calibration statistics come from its seed-42 FP16 run.
        They are recorded on the pod during that run when the published files are not available."""
        return self.variant == "fp16" and self.drive == "raw" and self.seed == 42 and self.rep == 0

    @property
    def uid(self) -> str:
        m = "none" if self.variant == "fp16" else self.map
        base = f"{self.variant}-{m}-{self.calib}-{self.clip}-s{self.seed}-t{self.steps}-r{self.rep}"
        return base + ("-dpre3" if self.drive == "pre" else "")        # v3 field (pilot v1: -dpre, v2: -dpre2)

    @property
    def video(self) -> Path:
        return RUNS / "videos" / f"{self.uid}.mp4"

    @property
    def meta(self) -> Path:
        return RUNS / "meta" / f"{self.uid}.json"

    @property
    def motion(self) -> str:
        return self.clip.split("_")[1]

    def quant_key(self) -> str:
        """Units sharing this key share one quantized transformer (seed and drive differ only)."""
        return f"{self.variant}-{self.map}-{self.calib}-{self.clip}"

    def subject(self) -> str:
        return self.clip.split("_")[0]

    def fp16_ref(self) -> "Unit":
        return Unit("fp16", self.clip, self.seed, "raw", "none", self.steps)

    def to_json(self) -> dict:
        d = asdict(self)
        d.update(uid=self.uid, calib=self.calib)
        return d


def pilot_units():
    c = PILOT_CLIP
    return [Unit("fp16", c, 42, group="pilot"), Unit("vidit", c, 42, group="pilot"),
            Unit("w4a8", c, 42, group="pilot"), Unit("w4a8", c, 42, "pre", group="pilot")]


def core_units():
    u = []
    for c in clips():
        for s in SEEDS:
            u.append(Unit("fp16", c, s))
            for v in QUANT_VARIANTS:
                u.append(Unit(v, c, s, "raw"))
                u.append(Unit(v, c, s, "pre"))
    return u


def optional_units():
    """FP16 with the corrected driving pose (isolates the effect of pre-correction at full precision)."""
    return [Unit("fp16", c, 42, "pre", group="optional") for c in clips()]


def fallback_units():
    """Used only if W4A8 collapses in the pilot: mixed precision replaces the w4a8 arm."""
    u = []
    for c in clips():
        for s in SEEDS:
            u.append(Unit("w4a8mp", c, s, "raw"))
            u.append(Unit("w4a8mp", c, s, "pre"))
    return u


def heldout_units():
    """R1.2 / R2.2: W8A8 and W4A8 with calibration statistics and protected-layer set K computed from the
    clips of the three other subjects (the evaluated clip and its subject never enter calibration)."""
    u = []
    for c in clips():
        for s in SEEDS:
            for v in QUANT_VARIANTS:
                u.append(Unit(v, c, s, "raw", group="heldout", calib_mode=HELDOUT))
                u.append(Unit(v, c, s, "pre", group="heldout", calib_mode=HELDOUT))
    return u


def seed_units():
    """R1.5 / R2.4: additional BF16 renders per clip (seed-to-seed noise floor against seed 42)."""
    return [Unit("fp16", c, s, group="seeds") for c in clips() for s in EXTRA_SEEDS]


def ref_units():
    """BF16 references (seeds 42 and 7) of the core set, same uids. Re-rendered on a fresh pod when the
    submitted volume is not reachable; the seed-42 renders also record the per-clip calibration files."""
    return [u for u in core_units() if u.variant == "fp16"]


SETS = {"pilot": pilot_units, "core": core_units, "optional": optional_units, "fallback": fallback_units,
        "heldout": heldout_units, "seeds": seed_units, "refs": ref_units}


def units_for(names: str):
    seen, out = set(), []
    for n in names.split(","):
        for u in SETS[n.strip()]():
            if u.uid not in seen:
                seen.add(u.uid)
                out.append(u)
    # FP16 references first; then group by quantized model so each is built once
    out.sort(key=lambda u: (u.variant != "fp16", u.quant_key(), u.drive, u.seed))
    return out


def pairs(names: str = "pilot,core,optional,fallback"):
    """(candidate uid, reference uid, tag) for every scoring pair implied by the unit sets."""
    P, seen = [], set()

    def add(c, r, t):
        if (c, r) not in seen:
            seen.add((c, r))
            P.append((c, r, t))

    all_u = units_for(names)
    for u in all_u:
        if u.variant == "fp16" and u.drive == "raw":
            continue
        tag = f"{u.variant}_{u.drive}" if u.variant != "fp16" else "fp16_pre"
        if u.calib == HELDOUT:
            tag += "_heldout"
        add(u.uid, u.fp16_ref().uid, tag)
    # FP16 seed-to-seed noise floor (seed 7 vs seed 42, same clip)
    fp = {(u.clip, u.seed): u for u in all_u if u.variant == "fp16" and u.drive == "raw"}
    for c in clips():
        if (c, 7) in fp and (c, 42) in fp:
            add(fp[(c, 7)].uid, fp[(c, 42)].uid, "fp16_seed")
    # additional seeds (revision): each against the seed-42 reference of the same clip
    for u in all_u:
        if u.variant == "fp16" and u.drive == "raw" and u.seed in EXTRA_SEEDS:
            add(u.uid, Unit("fp16", u.clip, 42).uid, "fp16_seedx")
    return P


def all_videos(names: str = "pilot,core,optional,fallback"):
    return {u.uid: u for u in units_for(names)}


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="core")
    a = ap.parse_args()
    us = units_for(a.sets)
    print("units:", len(us), Counter((u.variant, u.drive) for u in us))
    ps = pairs(a.sets)
    print("pairs:", len(ps), Counter(t for *_, t in ps))
