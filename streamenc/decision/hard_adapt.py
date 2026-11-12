"""Hard-clip score adaptation: regime prefilter + salvage denoise + candidate set.

Goal: raise hard-clip competition score by (1) spending budget under a regime-matched
encoder, (2) applying stronger denoise only when bitrate is otherwise infeasible,
(3) optional short dual-candidate pick among streaming-valid encodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from streamenc.decision.hardpath import (
    DEFAULT_HARD_CONFIG_ID,
    hard_feature_regime,
    select_hard_config,
)
from streamenc.features.extract import FeatureVector


@dataclass(frozen=True)
class HardAdaptPlan:
    regime: str
    primary_config_id: str
    primary_prefilter: str
    salvage_prefilter: str
    alt_config_id: str | None
    alt_prefilter: str


def choose_hard_prefilter(features: FeatureVector) -> str:
    """Prefilter for hard-path primary encode.

    Empirically (hard_adapt A/B): light denoise on noisy_hf *lowers* the only
    clip with score>0 (ai_hard_06). Keep primary clean; use strong denoise only
    for bitrate-incompressible content and as streaming salvage.
    """
    regime = hard_feature_regime(features)
    if regime == "bitrate_incompressible":
        return "hard_denoise"
    return "none"


def hard_adapt_plan(features: FeatureVector) -> HardAdaptPlan:
    """Full adaptation plan for a hard clip."""
    regime = hard_feature_regime(features)
    primary = select_hard_config(features) or DEFAULT_HARD_CONFIG_ID
    primary_pf = choose_hard_prefilter(features)

    # Salvage: if primary fails streaming, force strong denoise + bitrate-friendly arm.
    salvage_pf = "hard_denoise"
    if regime == "bitrate_incompressible":
        alt = "hard_nvenc_p6_noaq_fill" if primary != "hard_nvenc_p6_noaq_fill" else "hevc_nvenc_p6_vbr"
        alt_pf = "hard_denoise"
    elif regime == "edge_heavy":
        alt = "hard_svt_p5_psnr" if primary != "hard_svt_p5_psnr" else "hard_nvenc_p6_noaq_fill"
        alt_pf = "none"
    else:
        # noisy_hf / default: NVENC noaq fill as salvage/alt (no primary denoise).
        alt = "hard_nvenc_p6_noaq_fill" if primary != "hard_nvenc_p6_noaq_fill" else "svt_av1_p7_psnr"
        alt_pf = "none"

    return HardAdaptPlan(
        regime=regime,
        primary_config_id=primary,
        primary_prefilter=primary_pf,
        salvage_prefilter=salvage_pf,
        alt_config_id=alt,
        alt_prefilter=alt_pf,
    )
