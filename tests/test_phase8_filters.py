"""Phase 8 filter policy tests."""

from __future__ import annotations

from streamenc.features.extract import FeatureVector
from streamenc.filters.policy import choose_prefilter, filter_vf


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
    )
    base.update(kwargs)
    return FeatureVector(**base)


def test_default_none():
    assert choose_prefilter(_fv()) == "none"
    assert filter_vf("none") is None


def test_screen_never_filtered():
    assert (
        choose_prefilter(
            _fv(chroma_complexity=0.02, spatial_complexity=12.0, temporal_instability=0.5, motion_complexity=0.1)
        )
        == "none"
    )


def test_noisy_motion_gets_light_denoise():
    assert (
        choose_prefilter(_fv(temporal_instability=0.4, motion_complexity=0.05))
        == "hqdn3d_light"
    )
    assert "hqdn3d" in (filter_vf("hqdn3d_light") or "")
