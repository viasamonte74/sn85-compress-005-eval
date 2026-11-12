"""Staged encoder benchmark runner."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streamenc.benchmark.discover import ClipInfo
from streamenc.encoders.generic import EncoderConfig, build_encode_argv
from streamenc.probe import probe_video
from streamenc.probe.ffprobe_meta import safe_gop
from streamenc.score import clip_score, final_score
from streamenc.util.logging import get_logger
from streamenc.util.subprocess_run import run_command
from streamenc.validate.official_streaming import (
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)
from streamenc.validate.validate_clip import validate_clip

log = get_logger("streamenc.benchmark")

RETRY_FRACTIONS = (0.975, 0.94, 0.90, 0.85, 0.78, 0.70)


@dataclass
class ExperimentResult:
    clip_id: str
    family: str
    kind: str
    split: str
    resolution_tag: str
    width: int
    height: int
    fps: float
    frames: int
    config_id: str
    codec: str
    config_params: dict[str, Any]
    target_kbps: float
    aim_fraction: float
    avg_kbps: float | None
    peak_kbps: float | None
    avg_ok: bool | None
    peak_ok: bool | None
    keyframe_ok: bool | None
    encode_seconds: float
    attempts: int
    encode_ok: bool
    encode_error: str = ""
    vmaf: float | None = None
    vmaf_neg: float | None = None
    vmaf_delta: float | None = None
    clip_score: float | None = None
    validation_passed: bool | None = None
    violation_codes: list[str] = field(default_factory=list)
    estimated_final_score: float | None = None
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["violation_codes"] = ",".join(self.violation_codes)
        d["config_params"] = json.dumps(self.config_params, sort_keys=True)
        return d


def _ensure_nvenc_lib_path() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if enc not in parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([enc, *parts])


def encode_with_retries(
    clip: ClipInfo,
    cfg: EncoderConfig,
    output_path: str,
    *,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    peak_window: float = 2.0,
    peak_ratio: float = 1.5,
) -> ExperimentResult:
    _ensure_nvenc_lib_path()
    meta = probe_video(clip.path, count_frames=True)
    target = streaming_target_kbps(
        width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=bits_per_pixel
    )
    gop = safe_gop(clip.path, meta.fps, max_keyframe_interval)
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=".bench-partial-", suffix=".mp4", dir=out_dir)
    os.close(fd)

    attempts = 0
    encode_seconds = 0.0
    last_avg = last_peak = None
    last_aim = RETRY_FRACTIONS[0]
    avg_ok = peak_ok = key_ok = None
    encode_ok = False
    encode_error = ""

    try:
        for aim in RETRY_FRACTIONS:
            attempts += 1
            last_aim = aim
            kbps = target * aim
            argv = build_encode_argv(
                cfg=cfg,
                input_path=clip.path,
                output_path=partial,
                kbps=kbps,
                gop=gop,
                pix_fmt=meta.pix_fmt if meta.pix_fmt in {"yuv420p", "yuv420p10le", "p010le"} else "yuv420p",
                color_space=meta.color_space or "bt709",
                color_primaries=meta.color_primaries or "bt709",
                color_transfer=meta.color_transfer or "bt709",
                color_range=meta.color_range or "tv",
            )
            t0 = time.perf_counter()
            proc = run_command(argv, timeout=3600)
            encode_seconds += time.perf_counter() - t0
            if not proc.ok or not os.path.exists(partial) or os.path.getsize(partial) == 0:
                encode_error = (proc.stderr or proc.stdout or "empty encode")[:400]
                continue
            try:
                packets = read_packets_ffprobe(partial)
                meas = measure_streaming(
                    packets, fps=meta.fps, frames=meta.nb_frames, peak_window_seconds=peak_window
                )
            except Exception as exc:  # noqa: BLE001
                encode_error = f"packet measure failed: {exc}"
                continue
            last_avg, last_peak = meas.avg_kbps, meas.peak_kbps
            vs = streaming_violations(
                meas,
                target_kbps=target,
                peak_ratio=peak_ratio,
                peak_window_seconds=peak_window,
                max_keyframe_interval_seconds=max_keyframe_interval,
                fps=meta.fps,
            )
            codes = {v.code for v in vs}
            avg_ok = "STREAMING_BITRATE_EXCEEDED" not in codes
            peak_ok = "STREAMING_PEAK_EXCEEDED" not in codes
            key_ok = "STREAMING_KEYFRAME_INTERVAL" not in codes
            if avg_ok and peak_ok and key_ok:
                os.chmod(partial, 0o644)
                os.replace(partial, output_path)
                partial = ""
                encode_ok = True
                encode_error = ""
                break
        else:
            if not encode_error:
                encode_error = (
                    f"no aim fits: avg={last_avg} peak={last_peak} target={target}"
                )
    finally:
        if partial and os.path.exists(partial):
            try:
                os.unlink(partial)
            except OSError:
                pass

    return ExperimentResult(
        clip_id=clip.clip_id,
        family=clip.family,
        kind=clip.kind,
        split=clip.split,
        resolution_tag=clip.resolution_tag,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        frames=meta.nb_frames,
        config_id=cfg.config_id,
        codec=cfg.codec,
        config_params=dict(cfg.params),
        target_kbps=target,
        aim_fraction=last_aim,
        avg_kbps=last_avg,
        peak_kbps=last_peak,
        avg_ok=avg_ok,
        peak_ok=peak_ok,
        keyframe_ok=key_ok,
        encode_seconds=encode_seconds,
        attempts=attempts,
        encode_ok=encode_ok,
        encode_error=encode_error,
        output_path=output_path if encode_ok else "",
    )


def attach_validation(result: ExperimentResult, reference: str, *, run_vmaf: bool = True) -> ExperimentResult:
    if not result.encode_ok or not result.output_path:
        result.validation_passed = False
        result.violation_codes = ["ENCODE_FAILED"]
        result.estimated_final_score = 0.0
        return result
    report = validate_clip(
        reference,
        result.output_path,
        run_vmaf=run_vmaf,
        run_perceptual=run_vmaf,  # keep paired with quality stage
    )
    result.validation_passed = report.passed
    result.violation_codes = [v.code for v in report.violations]
    if "vmaf" in report.metrics:
        result.vmaf = float(report.metrics["vmaf"])
        result.vmaf_neg = float(report.metrics["vmaf_neg"])
        result.vmaf_delta = float(report.metrics["vmaf_delta"])
        result.clip_score = float(report.metrics["clip_score"])
        # Estimate competition final using this clip's encode time as if it were the whole batch
        # is wrong; instead store clip_score and leave aggregate to leaderboard with total time.
        result.estimated_final_score = result.clip_score if report.passed else 0.0
    else:
        result.estimated_final_score = 0.0 if not report.passed else None
    return result
