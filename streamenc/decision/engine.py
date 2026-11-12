"""Content-aware encoder selection engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from streamenc.decision.catalog import BASELINE_CONFIG_ID, baseline_config, get_config
from streamenc.decision.rules import Decision, decide_rules
from streamenc.encoders.generic import EncoderConfig
from streamenc.features.analyze import analyze_video
from streamenc.features.extract import FeatureVector


from streamenc.filters.policy import FILTER_CATALOG

PREFILTER_VF = dict(FILTER_CATALOG)


@dataclass(frozen=True)
class EngineResult:
    path: str
    clip_id: str
    features: FeatureVector
    decision: Decision
    feature_seconds: float
    mode: str

    def encoder_config(self) -> EncoderConfig:
        cfg = self.decision.encoder_config()
        vf = PREFILTER_VF.get(self.decision.prefilter)
        if not vf:
            return cfg
        params = dict(cfg.params)
        params["vf"] = vf
        return EncoderConfig(
            config_id=f"{cfg.config_id}+{self.decision.prefilter}",
            codec=cfg.codec,
            stage=cfg.stage,
            params=params,
            notes=f"{cfg.notes}; prefilter={self.decision.prefilter}",
        )

    def to_dict(self) -> dict[str, Any]:
        row = {
            "path": self.path,
            "clip_id": self.clip_id,
            "feature_seconds": self.feature_seconds,
            "mode": self.mode,
        }
        row.update({f"feat_{k}": v for k, v in self.features.to_dict().items()})
        row.update({f"dec_{k}": v for k, v in self.decision.to_dict().items() if k != "params"})
        row["dec_params"] = self.decision.params
        return row


def select_encoder(
    path: str | Path,
    *,
    prefer_quality: bool = True,
    max_samples: int = 24,
    max_width: int = 640,
) -> EngineResult:
    path = Path(path)
    analysis = analyze_video(
        path,
        mode="sampled",
        max_samples=max_samples,
        max_width=max_width,
    )
    decision = decide_rules(analysis.features, prefer_quality=prefer_quality)
    return EngineResult(
        path=str(path),
        clip_id=path.stem,
        features=analysis.features,
        decision=decision,
        feature_seconds=analysis.total_seconds,
        mode="rules",
    )


def fixed_baseline_decision() -> Decision:
    cfg = baseline_config()
    return Decision(
        config_id=cfg.config_id,
        codec=cfg.codec,
        preset=cfg.params.get("preset"),
        rate_control=str(cfg.params.get("rc", "vbr")),
        lookahead=int(cfg.params.get("rc_lookahead", 8)),
        spatial_aq=True,
        temporal_aq=True,
        bit_depth=8,
        prefilter="none",
        confidence=1.0,
        reason="fixed_baseline",
        rule_id="BASELINE",
        params=dict(cfg.params),
    )


__all__ = [
    "BASELINE_CONFIG_ID",
    "EngineResult",
    "fixed_baseline_decision",
    "get_config",
    "select_encoder",
]
