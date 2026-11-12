"""Unit tests for Phase 7 correction helpers."""

from __future__ import annotations

from streamenc.correct.controller import _apply_violation_fix
from streamenc.decision.catalog import get_config


def test_peak_fix_tightens_vbv_and_disables_aq():
    cfg = get_config("av1_nvenc_p6_vbr")
    fixed = _apply_violation_fix(cfg, {"STREAMING_PEAK_EXCEEDED"}, attempt_idx=1)
    assert fixed.params["maxrate_fraction"] <= 1.15
    assert fixed.params["bufsize_fraction"] <= 1.50
    assert fixed.params["spatial_aq"] is False


def test_keyframe_fix_sets_strict_gop():
    cfg = get_config("av1_nvenc_p6_vbr")
    fixed = _apply_violation_fix(cfg, {"STREAMING_KEYFRAME_INTERVAL"}, attempt_idx=2)
    assert fixed.params["strict_gop"] is True
    assert fixed.params["no_scenecut"] is True
