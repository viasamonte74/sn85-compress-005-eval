from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from streamenc.util.budget import StreamingBudget
from streamenc.util.subprocess_run import run_command


@dataclass(frozen=True)
class VideoMeta:
    path: str
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    r_frame_rate: str
    avg_frame_rate: str
    nb_frames: int
    color_space: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_range: str | None
    duration: float | None

    @property
    def fps_fraction(self) -> Fraction:
        return Fraction(self.r_frame_rate)

    @property
    def fps(self) -> float:
        return float(self.fps_fraction)

    def budget(
        self,
        bits_per_pixel: float = 0.05,
        peak_ratio: float = 1.5,
        average_tolerance: float = 0.01,
    ) -> StreamingBudget:
        return StreamingBudget(
            bits_per_pixel=bits_per_pixel,
            width=self.width,
            height=self.height,
            fps=self.fps,
            peak_ratio=peak_ratio,
            average_tolerance=average_tolerance,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = self.fps
        d["fps_fraction"] = str(self.fps_fraction)
        return d


def probe_video(path: str | Path, *, count_frames: bool = True, decode_count: bool = False,
                timeout: float = 300.0) -> VideoMeta:
    """Probe exact geometry, rational FPS, and frame count via ffprobe.

    ``count_frames`` counts video packets (``-count_packets``: demux only, ~0.04 s on a 1080p FFV1
    clip, one packet per frame for every codec/container we handle). ``decode_count=True`` decodes
    instead (``-count_frames``, ~8 s on the same clip) — only for a final exactness check.
    """
    path = str(path)
    entries = "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
    entries += "color_space,color_primaries,color_transfer,color_range,duration,nb_frames"
    argv = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames and decode_count:
        argv += ["-count_frames"]
        entries += ",nb_read_frames"
    elif count_frames:
        argv += ["-count_packets"]
        entries += ",nb_read_packets"
    argv += ["-show_entries", entries, "-of", "json", path]
    result = run_command(argv, timeout=timeout, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    if count_frames and decode_count:
        nb_frames = int(stream["nb_read_frames"])
    elif count_frames:
        nb_frames = int(stream["nb_read_packets"])
    else:
        nb = stream.get("nb_frames")
        nb_frames = int(nb) if nb not in (None, "N/A") else 0
    duration = stream.get("duration")
    return VideoMeta(
        path=path,
        codec_name=stream["codec_name"],
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=stream.get("pix_fmt") or "",
        r_frame_rate=stream["r_frame_rate"],
        avg_frame_rate=stream.get("avg_frame_rate") or stream["r_frame_rate"],
        nb_frames=nb_frames,
        color_space=stream.get("color_space"),
        color_primaries=stream.get("color_primaries"),
        color_transfer=stream.get("color_transfer"),
        color_range=stream.get("color_range"),
        duration=float(duration) if duration not in (None, "N/A") else None,
    )


def packet_pts(path: str | Path, *, timeout: float = 300.0) -> list[float]:
    """Sorted presentation timestamps (seconds) of the video packets (demux only, no decode)."""
    argv = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
            "-of", "csv=p=0", str(path)]
    result = run_command(argv, timeout=timeout, check=True)
    out: list[float] = []
    for line in result.stdout.splitlines():
        s = line.strip().rstrip(",")
        if not s or s == "N/A":
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return sorted(out)


def gop_for_timestamps(pts: list[float], fps: float, max_interval: float = 2.0) -> int:
    """Largest fixed GOP (frames) whose keyframe spans, measured on timestamps like the official
    streaming check (key->key and last key->end, end = last pts + 1/fps), all stay <= max_interval
    + 0.25/fps (the official limit is max_interval + 0.5/fps).

    Equals int(max_interval * fps) for constant-frame-rate input; smaller when the source has
    timestamp gaps (dropped frames / VFR), where a frame-count GOP would span more than 2 s.
    """
    nominal = max(1, int(max_interval * fps + 1e-9))
    if len(pts) < 2 or not fps > 0:
        return nominal
    end = pts[-1] + 1.0 / fps
    lim = max_interval + 0.25 / fps  # official check allows +0.5/fps; keep half of it for pts rounding
    for n in range(nominal, 0, -1):
        keys = pts[::n]
        spans = [b - a for a, b in zip(keys, keys[1:])] + [end - keys[-1]]
        if max(spans) <= lim:
            return n
    return 1


def safe_gop(path: str | Path, fps: float, max_interval: float = 2.0) -> int:
    """GOP for ``path`` that keeps every keyframe span <= max_interval on the real timestamps."""
    nominal = max(1, int(max_interval * fps + 1e-9))
    try:
        return gop_for_timestamps(packet_pts(path), fps, max_interval)
    except Exception:
        return nominal
