from __future__ import annotations

import math


def clip_score(vmaf: float) -> float:
    """Official per-clip quality term: clamp((VMAF - 60) / 40, 0, 1)."""
    return min(1.0, max(0.0, (vmaf - 60.0) / 40.0))


def speed_score(
    total_runtime_seconds: float,
    total_frames: float = 3428.0,
    *,
    fast_spf: float = 1.0 / 30.0,
    slow_spf: float = 2.0,
) -> float:
    """Official speed_score from competition result_rules."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    spf = total_runtime_seconds / total_frames
    if spf <= 0:
        return 1.0
    numer = math.log(slow_spf / spf)
    denom = math.log(slow_spf / fast_spf)
    return min(1.0, max(0.0, numer / denom))


def speed_multiplier(speed: float, weight: float = 0.3) -> float:
    return (1.0 - weight) + weight * speed


def final_score(quality_mean: float, total_runtime_seconds: float, total_frames: float = 3428.0) -> float:
    s = speed_score(total_runtime_seconds, total_frames)
    return quality_mean * speed_multiplier(s)
