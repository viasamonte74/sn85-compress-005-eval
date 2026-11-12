"""Video-packet bitrate measurement.

Matches the official streaming_example helper:
average and rolling peak over round(window_seconds * fps) packets,
using ffprobe packet sizes (container overhead excluded).

Phase 3 will cross-check against validator-vidaio streaming.py.
Until then this is the documented example semantics, not a claim of
bit-identical official scorer behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from streamenc.util.subprocess_run import run_command


@dataclass(frozen=True)
class PacketStats:
    avg_kbps: float
    peak_kbps: float
    packet_count: int
    total_bytes: int
    window_packets: int


def measure_packet_kbps(
    path: str,
    *,
    fps: float,
    frames: int,
    window_seconds: float = 2.0,
    timeout: float = 300.0,
) -> PacketStats:
    if fps <= 0 or frames <= 0:
        raise ValueError("fps and frames must be positive")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=size",
            "-of",
            "json",
            path,
        ],
        timeout=timeout,
        check=True,
    )
    packets = json.loads(result.stdout).get("packets") or []
    sizes = [int(p["size"]) for p in packets if "size" in p]
    if not sizes:
        raise RuntimeError(f"no video packets in {path}")
    duration_s = frames / fps
    total_bytes = sum(sizes)
    avg_kbps = 8.0 * total_bytes / duration_s / 1000.0
    window = max(1, round(window_seconds * fps))
    if len(sizes) <= window:
        peak_kbps = 8.0 * total_bytes / (len(sizes) / fps) / 1000.0
    else:
        run = best = sum(sizes[:window])
        for i in range(window, len(sizes)):
            run += sizes[i] - sizes[i - window]
            if run > best:
                best = run
        peak_kbps = 8.0 * best / (window / fps) / 1000.0
    return PacketStats(
        avg_kbps=avg_kbps,
        peak_kbps=peak_kbps,
        packet_count=len(sizes),
        total_bytes=total_bytes,
        window_packets=window,
    )
