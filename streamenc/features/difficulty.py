"""Calibrated content difficulty score for hard-path routing."""

from __future__ import annotations

from dataclasses import dataclass

from streamenc.features.extract import FeatureVector


@dataclass(frozen=True)
class DifficultyEstimate:
    score: float  # 0=easy … 1=extremely hard
    is_hard: bool
    reasons: tuple[str, ...]


def difficulty_score(features: FeatureVector) -> DifficultyEstimate:
    """Weighted blend of spatial/temporal/noise cues → [0, 1] difficulty.

    Thresholds tuned against dataset labels + official low VMAF hard clips.
    """
    motion = float(features.motion_complexity)
    temporal = float(features.temporal_complexity)
    spatial = float(features.spatial_complexity)
    texture = float(features.texture_complexity)
    edge = float(features.edge_density)
    instability = float(features.temporal_instability)
    highfreq = float(getattr(features, "highfreq_energy", 0.0) or 0.0)
    noise = float(getattr(features, "noise_estimate", 0.0) or 0.0)
    entropy = float(getattr(features, "entropy", 0.0) or 0.0)
    motion_p95 = float(getattr(features, "motion_p95", motion) or motion)
    spatial_p95 = float(getattr(features, "spatial_p95", spatial) or spatial)

    # Piecewise soft scores in ~[0,1]
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    s_motion = clamp01(motion / 0.08)
    s_temporal = clamp01(temporal / 0.08)
    s_motion_tail = clamp01(motion_p95 / 0.12)
    s_spatial = clamp01(spatial / 16.0)
    s_spatial_peak = clamp01(spatial_p95 / 24.0)
    s_texture = clamp01(texture / 0.7)
    s_edge = clamp01(edge / 0.08)
    s_instab = clamp01(instability / 0.5)
    s_hf = clamp01(highfreq / 25.0)
    s_noise = clamp01(noise / 8.0)
    s_entropy = clamp01((entropy - 4.0) / 3.0)  # typical gray entropy ~5–7

    # Emphasize motion + high-frequency / noise (encode killers at low bpp).
    score = (
        0.22 * s_motion
        + 0.12 * s_temporal
        + 0.14 * s_motion_tail
        + 0.12 * s_spatial
        + 0.08 * s_spatial_peak
        + 0.08 * s_texture
        + 0.06 * s_edge
        + 0.06 * s_instab
        + 0.06 * s_hf
        + 0.04 * s_noise
        + 0.02 * s_entropy
    )
    score = clamp01(score)

    reasons: list[str] = []
    if s_motion >= 0.5:
        reasons.append("motion")
    if s_motion_tail >= 0.55:
        reasons.append("motion_tail")
    if s_spatial >= 0.5 or s_spatial_peak >= 0.55:
        reasons.append("spatial")
    if s_hf >= 0.5 or s_noise >= 0.5:
        reasons.append("highfreq_noise")
    if s_instab >= 0.55:
        reasons.append("temporal_burst")
    if s_texture >= 0.55:
        reasons.append("texture")

    # Hard if score high OR classic rule fire.
    is_hard = score >= 0.42 or motion >= 0.04 or temporal >= 0.04
    if spatial >= 8.0 and motion >= 0.03:
        is_hard = True
        if "spatial" not in reasons:
            reasons.append("spatial_motion")
    return DifficultyEstimate(score=score, is_hard=is_hard, reasons=tuple(reasons))
