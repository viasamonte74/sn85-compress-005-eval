from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingBudget:
    """Competition bitrate budget derived from clip geometry and fps."""

    bits_per_pixel: float
    width: int
    height: int
    fps: float
    peak_ratio: float = 1.5
    average_tolerance: float = 0.01

    @property
    def target_bps(self) -> float:
        return self.bits_per_pixel * self.width * self.height * self.fps

    @property
    def target_kbps(self) -> float:
        return self.target_bps / 1000.0

    @property
    def average_limit_kbps(self) -> float:
        return self.target_kbps * (1.0 + self.average_tolerance)

    @property
    def peak_limit_kbps(self) -> float:
        return self.target_kbps * self.peak_ratio

    def gop_frames(self, max_keyframe_interval_seconds: float = 2.0) -> int:
        return max(1, int(max_keyframe_interval_seconds * self.fps + 1e-9))
