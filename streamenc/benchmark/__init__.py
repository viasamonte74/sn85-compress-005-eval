"""Benchmark harness (Phase 4)."""

from streamenc.benchmark.discover import (
    ClipInfo,
    discover_clips,
    enforce_family_split,
    select_benchmark_clips,
)
from streamenc.benchmark.leaderboard import config_leaderboard, experiments_frame, stage1_rank
from streamenc.benchmark.runner import ExperimentResult, attach_validation, encode_with_retries
from streamenc.encoders.generic import EncoderConfig, staged_configs

PHASE = 4

__all__ = [
    "PHASE",
    "ClipInfo",
    "EncoderConfig",
    "ExperimentResult",
    "attach_validation",
    "config_leaderboard",
    "discover_clips",
    "encode_with_retries",
    "enforce_family_split",
    "experiments_frame",
    "select_benchmark_clips",
    "stage1_rank",
    "staged_configs",
]
