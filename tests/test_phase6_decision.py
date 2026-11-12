"""Unit tests for Phase 6 decision rules."""

from __future__ import annotations

from streamenc.decision.rules import decide_rules
from streamenc.features.extract import FeatureVector


def _fv(**kwargs) -> FeatureVector:
    base = dict(
        spatial_complexity=3.0,
        temporal_complexity=0.01,
        edge_density=0.03,
        frame_diff_mean=2.0,
        frame_diff_std=0.5,
        frame_diff_p95=3.0,
        repeated_frame_ratio=0.1,
        motion_complexity=0.01,
        texture_complexity=0.3,
        smooth_gradient_ratio=0.7,
        chroma_complexity=0.12,
        temporal_instability=0.1,
        n_frames_analyzed=24,
        n_pairs=23,
        analysis_width=640,
        analysis_height=360,
    )
    base.update(kwargs)
    return FeatureVector(**base)


def test_high_motion_selects_hard_path():
    d = decide_rules(_fv(motion_complexity=0.09, temporal_complexity=0.08))
    assert d.hard_path is True
    assert d.rule_id == "R0_hard_path"
    assert d.config_id.startswith("hard_")


def test_borderline_hard_motion_triggers_hard_path():
    d = decide_rules(_fv(motion_complexity=0.045, temporal_complexity=0.045))
    assert d.hard_path is True
    assert d.config_id == "hard_svt_p5_psnr"


def test_screen_like_selects_svt():
    d = decide_rules(_fv(chroma_complexity=0.02, spatial_complexity=12.0, repeated_frame_ratio=0.6))
    assert d.config_id == "svt_av1_p7_psnr"
    assert d.rule_id == "R2_screen_svt"


def test_default_quality_svt():
    d = decide_rules(_fv())
    assert d.config_id == "svt_av1_p7_psnr"
    assert d.rule_id == "R4_default_svt"
