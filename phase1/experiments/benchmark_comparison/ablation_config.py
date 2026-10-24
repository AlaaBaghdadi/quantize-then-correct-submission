"""Ablation variant definitions for KAPT component study (A0–A4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

InputKind = Literal["kalman", "fused", "passthrough"]


@dataclass(frozen=True)
class AblationVariant:
    """One ablation condition."""

    method_id: str
    display_name: str
    input_kind: InputKind
    hard_clamp: bool
    max_drift: float
    run_correction: bool
    output_suffix: str  # filename stem suffix under conditions/{method_id}/

    def output_filename(self, stem: str) -> str:
        return f"{stem}{self.output_suffix}.json"


ABLATION_VARIANTS: List[AblationVariant] = [
    AblationVariant(
        method_id="kapt_full",
        display_name="KAPT (full)",
        input_kind="kalman",
        hard_clamp=False,
        max_drift=0.05,
        run_correction=True,
        output_suffix="_corrected",
    ),
    AblationVariant(
        method_id="ablation_no_kalman",
        display_name="w/o Kalman",
        input_kind="fused",
        hard_clamp=False,
        max_drift=0.05,
        run_correction=True,
        output_suffix="_corrected",
    ),
    AblationVariant(
        method_id="ablation_no_correction",
        display_name="w/o correction (Kalman only)",
        input_kind="passthrough",
        hard_clamp=False,
        max_drift=0.05,
        run_correction=False,
        output_suffix="_kalman_only",
    ),
    AblationVariant(
        method_id="ablation_no_soft_cone",
        display_name="w/o soft cone (hard clamp)",
        input_kind="kalman",
        hard_clamp=True,
        max_drift=0.05,
        run_correction=True,
        output_suffix="_corrected",
    ),
    AblationVariant(
        method_id="ablation_no_length_stab",
        display_name="w/o length stabilizer",
        input_kind="kalman",
        hard_clamp=False,
        max_drift=1.0,
        run_correction=True,
        output_suffix="_corrected",
    ),
]

ABLATION_BY_ID: Dict[str, AblationVariant] = {v.method_id: v for v in ABLATION_VARIANTS}

REFERENCE_METHOD = "kapt_full"
