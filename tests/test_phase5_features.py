"""Unit tests for Phase 5 feature extraction."""

from __future__ import annotations

import numpy as np

from streamenc.features.extract import FEATURE_NAMES, accumulate_frame_stats
from streamenc.features.analyze import _iter_indices, feature_agreement


def test_feature_vector_on_synthetic_frames():
    frames = []
    rng = np.random.default_rng(0)
    for i in range(8):
        # drifting noise + occasional static
        img = np.full((180, 320, 3), 40, dtype=np.uint8)
        if i % 3 != 0:
            noise = rng.integers(0, 40, size=(180, 320, 3), dtype=np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(img)
    fv = accumulate_frame_stats(frames, max_width=320)
    d = fv.to_dict()
    for name in FEATURE_NAMES:
        assert name in d
        assert np.isfinite(d[name])
    assert fv.n_frames_analyzed == 8
    assert fv.n_pairs == 7
    assert 0.0 <= fv.edge_density <= 1.0
    assert 0.0 <= fv.repeated_frame_ratio <= 1.0
    assert fv.highfreq_energy >= 0.0
    assert fv.entropy > 0.0


def test_difficulty_score_separates_static_vs_noisy():
    from streamenc.features.difficulty import difficulty_score

    static = [np.full((120, 160, 3), 80, dtype=np.uint8) for _ in range(6)]
    rng = np.random.default_rng(1)
    noisy = [rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8) for _ in range(6)]
    d_static = difficulty_score(accumulate_frame_stats(static))
    d_noisy = difficulty_score(accumulate_frame_stats(noisy))
    assert d_noisy.score > d_static.score
    assert d_noisy.is_hard is True


def test_iter_indices_sampled_and_full():
    full = _iter_indices(100, mode="full", max_samples=24, stride=1)
    assert full == list(range(100))
    sampled = _iter_indices(100, mode="sampled", max_samples=10, stride=1)
    assert 1 <= len(sampled) <= 10
    assert sampled[0] == 0
    assert sampled == sorted(sampled)


def test_feature_agreement_identical():
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    a = accumulate_frame_stats(frames)
    b = accumulate_frame_stats(frames)
    agr = feature_agreement(a, b)
    assert agr["mae"] == 0.0
    assert agr["pearson_corr"] == 1.0
