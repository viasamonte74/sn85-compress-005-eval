"""Hard-content detection and hard-path encoder selection."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from streamenc.encoders.generic import EncoderConfig, staged_configs
from streamenc.features.difficulty import DifficultyEstimate, difficulty_score
from streamenc.features.extract import FeatureVector

_POLICY_PATH = Path(__file__).with_name("hard_config_policy.json")


def is_hard_content(features: FeatureVector) -> bool:
    """Hard / complexity-floor detector using calibrated difficulty score."""
    return difficulty_score(features).is_hard


def estimate_difficulty(features: FeatureVector) -> DifficultyEstimate:
    return difficulty_score(features)


def hard_path_configs() -> list[EncoderConfig]:
    return [c for c in staged_configs() if c.config_id.startswith("hard_")]


# Fallback before / without empirical policy. Overridden by select_hard_config.
DEFAULT_HARD_CONFIG_ID = "hard_svt_p5_psnr"


@lru_cache(maxsize=1)
def _load_policy() -> dict:
    if not _POLICY_PATH.is_file():
        return {}
    try:
        return json.loads(_POLICY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def hard_feature_regime(features: FeatureVector) -> str:
    """Coarse feature regime used by the empirical hard-config policy.

    ``bitrate_incompressible`` calibrated on generated-hard probe util≥1.15:
    highfreq_energy≥19 and spatial_p95≥95 (F1≈0.82 on the hard_feature_map set).
    """
    noise = float(getattr(features, "noise_estimate", 0.0) or 0.0)
    hf = float(getattr(features, "highfreq_energy", 0.0) or 0.0)
    spatial_p95 = float(getattr(features, "spatial_p95", 0.0) or 0.0)
    motion = float(features.motion_complexity)
    spatial = float(features.spatial_complexity)
    edge = float(features.edge_density)
    if hf >= 19.0 and spatial_p95 >= 95.0:
        return "bitrate_incompressible"
    if noise >= 3.5 or hf >= 12.0:
        return "noisy_hf"
    if edge >= 0.06 and spatial >= 10.0:
        return "edge_heavy"
    if motion >= 0.06:
        return "high_motion"
    if spatial >= 12.0:
        return "spatial_heavy"
    return "generic_hard"


def select_hard_config(features: FeatureVector) -> str:
    """Pick hard-path config_id from features via empirical policy.

    Policy is produced by ``scripts/hard_feature_config_map.py``: for each hard
    clip, features are extracted and encoder/settings are swept for max score;
    winners are aggregated into regime → config rules.
    """
    policy = _load_policy()
    default = str(policy.get("default_hard_config_id") or DEFAULT_HARD_CONFIG_ID)
    regime = hard_feature_regime(features)
    by_regime = policy.get("by_regime") or {}
    if regime in by_regime:
        return str(by_regime[regime])

    for rule in policy.get("rules") or []:
        when = str(rule.get("when") or "")
        cid = str(rule.get("config_id") or default)
        if when.startswith("regime=="):
            if when.split("==", 1)[1] == regime:
                return cid
            continue
        if when == "default_hard":
            return cid
    return default
