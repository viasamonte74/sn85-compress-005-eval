"""Named encoder configs used by the decision engine."""

from __future__ import annotations

from streamenc.encoders.generic import EncoderConfig, staged_configs


def config_catalog() -> dict[str, EncoderConfig]:
    return {c.config_id: c for c in staged_configs()}


def get_config(config_id: str) -> EncoderConfig:
    cat = config_catalog()
    if config_id not in cat:
        raise KeyError(f"unknown config_id: {config_id}")
    return cat[config_id]


BASELINE_CONFIG_ID = "av1_nvenc_p6_vbr"


def baseline_config() -> EncoderConfig:
    return get_config(BASELINE_CONFIG_ID)
