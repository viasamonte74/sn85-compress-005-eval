"""Content-aware encoder decision engine (Phase 6)."""

from streamenc.decision.catalog import BASELINE_CONFIG_ID, baseline_config, get_config
from streamenc.decision.engine import EngineResult, fixed_baseline_decision, select_encoder
from streamenc.decision.hard_adapt import choose_hard_prefilter, hard_adapt_plan
from streamenc.decision.hardpath import (
    DEFAULT_HARD_CONFIG_ID,
    estimate_difficulty,
    is_hard_content,
    select_hard_config,
)
from streamenc.decision.rules import Decision, decide_rules

PHASE = 6

__all__ = [
    "PHASE",
    "BASELINE_CONFIG_ID",
    "DEFAULT_HARD_CONFIG_ID",
    "Decision",
    "EngineResult",
    "baseline_config",
    "decide_rules",
    "estimate_difficulty",
    "fixed_baseline_decision",
    "get_config",
    "is_hard_content",
    "select_encoder",
    "select_hard_config",
    "choose_hard_prefilter",
    "hard_adapt_plan",
]
