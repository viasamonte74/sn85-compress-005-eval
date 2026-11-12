"""Rule-based content → encoder configuration mapping.

Calibrated from Phase 4 stage-2 deltas vs ``av1_nvenc_p6_vbr`` and Phase 5
feature means by kind:

- ``svt_av1_p7_psnr``: small but consistent quality edge (+~0.007 clip_score)
- ``av1_nvenc_p6_noaq``: best salvage VMAF on hard/high-motion floor cases
- ``av1_nvenc_p6_multipass_qres``: fastest NVENC; slight quality regression
- Screen-like content (low chroma, high spatial / static): AQ rarely helps
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from streamenc.decision.catalog import BASELINE_CONFIG_ID, get_config
from streamenc.decision.hard_adapt import choose_hard_prefilter
from streamenc.decision.hardpath import (
    DEFAULT_HARD_CONFIG_ID,
    estimate_difficulty,
    is_hard_content,
    select_hard_config,
)
from streamenc.encoders.generic import EncoderConfig
from streamenc.features.extract import FeatureVector


@dataclass(frozen=True)
class Decision:
    config_id: str
    codec: str
    preset: str | int | None
    rate_control: str
    lookahead: int | None
    spatial_aq: bool | None
    temporal_aq: bool | None
    bit_depth: int
    prefilter: str  # none | hqdn3d_light
    confidence: float
    reason: str
    rule_id: str
    params: dict[str, Any] = field(default_factory=dict)
    hard_path: bool = False
    difficulty_score: float = 0.0
    difficulty_reasons: tuple[str, ...] = ()

    def encoder_config(self) -> EncoderConfig:
        return get_config(self.config_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_config(
    cfg: EncoderConfig,
    *,
    reason: str,
    rule_id: str,
    confidence: float,
    prefilter: str = "none",
    hard_path: bool = False,
    difficulty_score: float = 0.0,
    difficulty_reasons: tuple[str, ...] = (),
) -> Decision:
    p = cfg.params
    bit_depth = 10 if p.get("highbitdepth") or p.get("pix_fmt") == "yuv420p10le" else 8
    return Decision(
        config_id=cfg.config_id,
        codec=cfg.codec,
        preset=p.get("preset"),
        rate_control=str(p.get("rc", "vbr")),
        lookahead=int(p["rc_lookahead"]) if "rc_lookahead" in p else (8 if cfg.codec.endswith("nvenc") else None),
        spatial_aq=None if "spatial_aq" not in p else bool(p["spatial_aq"]),
        temporal_aq=None if "temporal_aq" not in p else bool(p["temporal_aq"]),
        bit_depth=bit_depth,
        prefilter=prefilter,
        confidence=confidence,
        reason=reason,
        rule_id=rule_id,
        params=dict(p),
        hard_path=hard_path,
        difficulty_score=difficulty_score,
        difficulty_reasons=difficulty_reasons,
    )


def decide_rules(
    features: FeatureVector,
    *,
    prefer_quality: bool = True,
    hard_config_id: str | None = None,
) -> Decision:
    """Map a compact feature vector to an encoder configuration."""
    motion = float(features.motion_complexity)
    temporal = float(features.temporal_complexity)
    spatial = float(features.spatial_complexity)
    chroma = float(features.chroma_complexity)
    repeat = float(features.repeated_frame_ratio)
    instability = float(features.temporal_instability)

    # Optional light denoise only for noisy/unstable high-motion sources.
    prefilter = "hqdn3d_light" if instability >= 0.35 and motion >= 0.04 else "none"
    diff = estimate_difficulty(features)

    # Rule 0 / 1: hard / complexity-floor → dedicated hard-path (fuller budget + slower encode).
    # Config from empirical feature→winner map; prefilter from hard_adapt regime rules.
    if diff.is_hard or is_hard_content(features):
        hid = hard_config_id or select_hard_config(features) or DEFAULT_HARD_CONFIG_ID
        try:
            hcfg = get_config(hid)
        except KeyError:
            hcfg = get_config("hard_nvenc_p7_fill")
        hard_pf = choose_hard_prefilter(features) if prefer_quality else "none"
        return _from_config(
            hcfg,
            reason=f"hard_path_full_budget:{hid}:{hard_pf}",
            rule_id="R0_hard_path",
            confidence=min(0.95, 0.55 + 0.4 * diff.score),
            prefilter=hard_pf,
            hard_path=True,
            difficulty_score=diff.score,
            difficulty_reasons=diff.reasons,
        )

    # Rule 2: screen-like (low chroma + sharp/static) → quality SVT; AQ not useful.
    screen_like = chroma < 0.05 and (spatial >= 5.0 or repeat >= 0.45)
    if screen_like:
        if prefer_quality:
            return _from_config(
                get_config("svt_av1_p7_psnr"),
                reason="screen_like_svt_quality",
                rule_id="R2_screen_svt",
                confidence=0.8,
                prefilter="none",
                difficulty_score=diff.score,
                difficulty_reasons=diff.reasons,
            )
        return _from_config(
            get_config("av1_nvenc_p6_multipass_qres"),
            reason="screen_like_nvenc_fast",
            rule_id="R2_screen_fast",
            confidence=0.7,
            prefilter="none",
            difficulty_score=diff.score,
            difficulty_reasons=diff.reasons,
        )

    # Rule 3: smooth synthetic / render (low spatial, richer chroma) → noaq slight edge.
    if spatial < 1.5 and chroma >= 0.10 and motion < 0.03:
        if prefer_quality:
            return _from_config(
                get_config("svt_av1_p7_psnr"),
                reason="smooth_render_svt",
                rule_id="R3_smooth_svt",
                confidence=0.7,
                prefilter=prefilter,
                difficulty_score=diff.score,
                difficulty_reasons=diff.reasons,
            )
        return _from_config(
            get_config("av1_nvenc_p6_noaq"),
            reason="smooth_render_noaq",
            rule_id="R3_smooth_noaq",
            confidence=0.65,
            prefilter=prefilter,
            difficulty_score=diff.score,
            difficulty_reasons=diff.reasons,
        )

    # Rule 4: default — SVT p7 for quality bias, else fixed NVENC baseline.
    if prefer_quality:
        return _from_config(
            get_config("svt_av1_p7_psnr"),
            reason="default_svt_quality",
            rule_id="R4_default_svt",
            confidence=0.55,
            prefilter=prefilter,
            difficulty_score=diff.score,
            difficulty_reasons=diff.reasons,
        )
    return _from_config(
        get_config(BASELINE_CONFIG_ID),
        reason="default_baseline",
        rule_id="R4_default_baseline",
        confidence=0.5,
        prefilter=prefilter,
        difficulty_score=diff.score,
        difficulty_reasons=diff.reasons,
    )
