"""Schema for encoder-rank dataset artifacts."""

from __future__ import annotations

from typing import Any

LABEL_FIELDS: tuple[str, ...] = (
    "clip_id",
    "path",
    "kind",  # ai | render | screen | generated_hard
    "tier",  # easy | medium | hard | other
    "family",  # leakage group (source stem without augment suffix)
    "split",  # train | val | test
    "winner_config_id",
    "winner_config_index",
    "winner_prefilter",
    "winner_aim",
    "winner_vmaf",
    "winner_clip_score",
    "winner_util",
    "winner_passed",
    "winner_passed_streaming",
    "n_trials",
    "n_streaming_pass",
    "max_vmaf",
    "max_clip_score",
    "difficulty_score",
    "is_hard",
    "regime",
    "label_seconds",
    # Strict supervision
    "trainable",
    "hopeless",
    "winner_confidence",
    "n_gate_valid",
    "reject_reason",
    "tie_with",
)

TRIAL_FIELDS: tuple[str, ...] = (
    "clip_id",
    "config_id",
    "config_index",
    "prefilter",
    "stage",  # A | B
    "ok",
    "passed_streaming",
    "passed",
    "vmaf",
    "vmaf_neg",
    "vmaf_delta",
    "clip_score",
    "util",
    "aim",
    "avg_kbps",
    "peak_kbps",
    "encode_seconds",
    "cpu_seconds",
    "enc_cpu_seconds",
    "crf_estimate",
    "fit_history",
    "attempts",
    "codes",
)


def empty_label(**kwargs: Any) -> dict[str, Any]:
    row = {k: None for k in LABEL_FIELDS}
    row.update(kwargs)
    return row


def empty_trial(**kwargs: Any) -> dict[str, Any]:
    row = {k: None for k in TRIAL_FIELDS}
    row.update(kwargs)
    return row
