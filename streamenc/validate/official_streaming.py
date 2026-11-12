"""Official streaming packet measurement and delivery-rule gates.

Ported from miner-vidaio ``vidaio/scoring/streaming.py`` and
``vidaio/scoring_worker/streaming_backend.py`` (scoring_version
vidaio-scorer/1+8f10f37c629e / streaming-packets/1).

Keep this logic aligned with upstream; do not "improve" the window math.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

from streamenc.util.subprocess_run import run_command


@dataclass(frozen=True)
class StreamingPacket:
    size_bytes: int
    pts_seconds: float | None
    keyframe: bool


@dataclass(frozen=True)
class StreamingMeasurement:
    avg_kbps: float
    peak_kbps: float
    max_keyframe_interval_seconds: float
    first_frame_is_keyframe: bool
    keyframes: int
    packets: int


@dataclass(frozen=True)
class ValidityViolation:
    code: str
    detail: str
    measured: float | None = None
    limit: float | None = None


def streaming_target_kbps(*, width: int, height: int, fps: float, bits_per_pixel: float) -> float:
    if width <= 0 or height <= 0 or not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"cannot derive a budget for {width}x{height} at {fps} fps")
    return bits_per_pixel * width * height * fps / 1000.0


def measure_streaming(
    packets: Sequence[StreamingPacket],
    *,
    fps: float,
    frames: int,
    peak_window_seconds: float,
) -> StreamingMeasurement:
    """Exact official measurement (decode-order packet sizes + PTS keyframe spans)."""
    if not packets:
        raise ValueError("the candidate has no video packets")
    if frames < 1 or not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"invalid reference timing: {frames} frames at {fps} fps")
    if any(p.pts_seconds is None or not math.isfinite(p.pts_seconds) for p in packets):
        raise ValueError("a video packet has no presentation timestamp")
    sizes = [int(p.size_bytes) for p in packets]
    if any(s < 0 for s in sizes):
        raise ValueError("negative packet size")
    duration = frames / fps
    avg_kbps = 8.0 * sum(sizes) / duration / 1000.0
    window = max(1, round(peak_window_seconds * fps))
    if len(sizes) <= window:
        peak_kbps = 8.0 * sum(sizes) / (len(sizes) / fps) / 1000.0
    else:
        running = sum(sizes[:window])
        best = running
        for i in range(window, len(sizes)):
            running += sizes[i] - sizes[i - window]
            best = max(best, running)
        peak_kbps = 8.0 * best / (window / fps) / 1000.0
    pts = [float(p.pts_seconds) for p in packets]  # type: ignore[arg-type]
    start = min(pts)
    end = max(pts) + 1.0 / fps
    keys = sorted(float(p.pts_seconds) for p in packets if p.keyframe)  # type: ignore[arg-type]
    first_is_key = bool(keys) and abs(keys[0] - start) < 0.5 / fps
    if keys:
        spans = [b - a for a, b in zip(keys, keys[1:])] + [end - keys[-1], keys[0] - start]
        max_interval = max(spans)
    else:
        max_interval = end - start
    return StreamingMeasurement(
        avg_kbps=avg_kbps,
        peak_kbps=peak_kbps,
        max_keyframe_interval_seconds=max_interval,
        first_frame_is_keyframe=first_is_key,
        keyframes=len(keys),
        packets=len(packets),
    )


def streaming_violations(
    measurement: StreamingMeasurement,
    *,
    target_kbps: float,
    peak_ratio: float,
    peak_window_seconds: float,
    max_keyframe_interval_seconds: float,
    fps: float,
    streaming_rate_tolerance: float = 0.01,
) -> list[ValidityViolation]:
    violations: list[ValidityViolation] = []
    limit = target_kbps * (1.0 + streaming_rate_tolerance)
    if measurement.avg_kbps > limit:
        violations.append(
            ValidityViolation(
                code="STREAMING_BITRATE_EXCEEDED",
                detail=(
                    f"average {measurement.avg_kbps:.1f} kbps over the "
                    f"{target_kbps:.1f} kbps budget"
                ),
                measured=measurement.avg_kbps,
                limit=limit,
            )
        )
    peak_limit = target_kbps * peak_ratio
    if measurement.peak_kbps > peak_limit:
        violations.append(
            ValidityViolation(
                code="STREAMING_PEAK_EXCEEDED",
                detail=(
                    f"{measurement.peak_kbps:.1f} kbps over a {peak_window_seconds:g} s window, "
                    f"limit {peak_ratio:g}x the budget"
                ),
                measured=measurement.peak_kbps,
                limit=peak_limit,
            )
        )
    interval_limit = max_keyframe_interval_seconds + 0.5 / fps
    if (not measurement.first_frame_is_keyframe) or (
        measurement.max_keyframe_interval_seconds > interval_limit
    ):
        violations.append(
            ValidityViolation(
                code="STREAMING_KEYFRAME_INTERVAL",
                detail=(
                    "the first frame is not a keyframe"
                    if not measurement.first_frame_is_keyframe
                    else f"{measurement.max_keyframe_interval_seconds:.3f} s between keyframes"
                ),
                measured=measurement.max_keyframe_interval_seconds,
                limit=interval_limit,
            )
        )
    return violations


def read_packets_ffprobe(path: str, *, ffprobe: str = "ffprobe", timeout: float = 300.0) -> list[StreamingPacket]:
    """Official worker packet reader: size, pts_time, flags (keyframe = 'K' in flags)."""
    result = run_command(
        [
            ffprobe,
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=size,pts_time,flags",
            "-print_format",
            "json",
            path,
        ],
        timeout=timeout,
        check=True,
    )
    try:
        doc = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"packets unreadable: {exc}") from exc
    out: list[StreamingPacket] = []
    for packet in doc.get("packets", []):
        pts = packet.get("pts_time")
        try:
            pts_seconds = float(pts) if pts not in (None, "N/A") else None
            size = int(packet["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed packet entry: {packet!r}") from exc
        out.append(
            StreamingPacket(
                size_bytes=size,
                pts_seconds=pts_seconds,
                keyframe="K" in str(packet.get("flags", "")),
            )
        )
    return out
