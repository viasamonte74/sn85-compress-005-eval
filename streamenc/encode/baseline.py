"""Phase 2 baseline: one FFV1 input -> one NVENC AV1 MP4 output.

Usage:
  python -m streamenc.encode.baseline INPUT_PATH OUTPUT_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from streamenc.encode.packets import measure_packet_kbps
from streamenc.encoders.nvenc_av1 import NvencAv1Config, build_ffmpeg_argv
from streamenc.probe import probe_video
from streamenc.util.logging import get_logger
from streamenc.util.subprocess_run import run_command

log = get_logger("streamenc.encode")


@dataclass
class EncodeResult:
    input: str
    output: str
    width: int
    height: int
    fps: float
    r_frame_rate: str
    frames_in: int
    frames_out: int
    budget_kbps: float
    aim_fraction: float
    target_kbps: float
    avg_kbps: float
    peak_kbps: float
    avg_limit_kbps: float
    peak_limit_kbps: float
    gop: int
    encode_seconds: float
    attempts: int
    preset: str
    ok: bool
    reason: str = ""


def _ensure_nvenc_lib_path() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if enc not in parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([enc, *parts])


def _count_frames(path: str, timeout: float = 300.0) -> int:
    r = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            path,
        ],
        timeout=timeout,
        check=True,
    )
    return int(r.stdout.strip())


def _first_packet_is_keyframe(path: str, timeout: float = 120.0) -> bool:
    r = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=flags",
            "-of",
            "csv=p=0",
            path,
        ],
        timeout=timeout,
        check=True,
    )
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # flags look like "K_" for keyframe
        return "K" in line.split(",")[0]
    return False


def encode_baseline(
    input_path: str,
    output_path: str,
    *,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    peak_window: float = 2.0,
    peak_ratio: float = 1.5,
    average_tolerance: float = 0.01,
    cfg: NvencAv1Config | None = None,
    verify_frames: bool = True,
) -> EncodeResult:
    _ensure_nvenc_lib_path()
    cfg = cfg or NvencAv1Config()
    meta = probe_video(input_path, count_frames=True)
    budget = meta.budget(
        bits_per_pixel=bits_per_pixel,
        peak_ratio=peak_ratio,
        average_tolerance=average_tolerance,
    )
    gop = budget.gop_frames(max_keyframe_interval)
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, partial = tempfile.mkstemp(prefix=".streaming-partial-", suffix=".mp4", dir=out_dir)
    os.close(fd)
    attempts = 0
    last_avg = 0.0
    last_peak = 0.0
    last_aim = cfg.target_fraction
    encode_seconds = 0.0
    frames_out = 0
    reason = ""

    try:
        for aim in cfg.retry_fractions:
            attempts += 1
            last_aim = aim
            target = budget.target_kbps * aim
            argv = build_ffmpeg_argv(
                input_path=input_path,
                output_path=partial,
                kbps=target,
                gop=gop,
                cfg=cfg,
                pix_fmt=meta.pix_fmt if meta.pix_fmt in {"yuv420p", "yuv444p", "nv12", "p010le"} else "yuv420p",
                color_space=meta.color_space or "bt709",
                color_primaries=meta.color_primaries or "bt709",
                color_transfer=meta.color_transfer or "bt709",
                color_range=meta.color_range or "tv",
            )
            t0 = time.perf_counter()
            proc = run_command(argv, timeout=3600)
            encode_seconds += time.perf_counter() - t0
            if not proc.ok:
                reason = f"ffmpeg failed: {proc.stderr.strip() or proc.stdout.strip()}"
                log.error("%s", reason)
                break
            if not os.path.exists(partial) or os.path.getsize(partial) == 0:
                reason = "empty encode output"
                break

            stats = measure_packet_kbps(
                partial,
                fps=meta.fps,
                frames=meta.nb_frames,
                window_seconds=peak_window,
            )
            last_avg, last_peak = stats.avg_kbps, stats.peak_kbps
            under_avg = stats.avg_kbps <= budget.average_limit_kbps
            under_peak = stats.peak_kbps <= budget.peak_limit_kbps
            log.info(
                "attempt %s aim=%.3f target=%.1f avg=%.1f peak=%.1f limits=%.1f/%.1f",
                attempts,
                aim,
                target,
                stats.avg_kbps,
                stats.peak_kbps,
                budget.average_limit_kbps,
                budget.peak_limit_kbps,
            )
            if under_avg and under_peak:
                if verify_frames:
                    frames_out = _count_frames(partial)
                    if frames_out != meta.nb_frames:
                        reason = f"frame count mismatch in={meta.nb_frames} out={frames_out}"
                        break
                    if not _first_packet_is_keyframe(partial):
                        reason = "first packet is not a keyframe"
                        break
                else:
                    frames_out = meta.nb_frames
                os.chmod(partial, 0o644)
                os.replace(partial, output_path)
                partial = ""
                return EncodeResult(
                    input=input_path,
                    output=output_path,
                    width=meta.width,
                    height=meta.height,
                    fps=meta.fps,
                    r_frame_rate=meta.r_frame_rate,
                    frames_in=meta.nb_frames,
                    frames_out=frames_out,
                    budget_kbps=budget.target_kbps,
                    aim_fraction=aim,
                    target_kbps=target,
                    avg_kbps=last_avg,
                    peak_kbps=last_peak,
                    avg_limit_kbps=budget.average_limit_kbps,
                    peak_limit_kbps=budget.peak_limit_kbps,
                    gop=gop,
                    encode_seconds=encode_seconds,
                    attempts=attempts,
                    preset=cfg.preset,
                    ok=True,
                )
        if not reason:
            reason = (
                f"no aim fits budget: last avg={last_avg:.1f} peak={last_peak:.1f} "
                f"limits={budget.average_limit_kbps:.1f}/{budget.peak_limit_kbps:.1f}"
            )
    finally:
        if partial and os.path.exists(partial):
            try:
                os.unlink(partial)
            except OSError:
                pass

    return EncodeResult(
        input=input_path,
        output=output_path,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        r_frame_rate=meta.r_frame_rate,
        frames_in=meta.nb_frames,
        frames_out=frames_out,
        budget_kbps=budget.target_kbps,
        aim_fraction=last_aim,
        target_kbps=budget.target_kbps * last_aim,
        avg_kbps=last_avg,
        peak_kbps=last_peak,
        avg_limit_kbps=budget.average_limit_kbps,
        peak_limit_kbps=budget.peak_limit_kbps,
        gop=gop,
        encode_seconds=encode_seconds,
        attempts=attempts,
        preset=cfg.preset,
        ok=False,
        reason=reason,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits-per-pixel", type=float, default=0.05)
    parser.add_argument("--max-keyframe-interval", type=float, default=2.0)
    parser.add_argument("--peak-window", type=float, default=2.0)
    parser.add_argument("--peak-ratio", type=float, default=1.5)
    parser.add_argument("--preset", default="p6")
    parser.add_argument("--json", action="store_true", help="print EncodeResult JSON on stdout")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)

    cfg = NvencAv1Config(preset=args.preset)
    result = encode_baseline(
        args.input,
        args.output,
        bits_per_pixel=args.bits_per_pixel,
        max_keyframe_interval=args.max_keyframe_interval,
        peak_window=args.peak_window,
        peak_ratio=args.peak_ratio,
        cfg=cfg,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        if result.ok:
            log.info(
                "ok %s -> %s avg=%.1f/%.1f peak=%.1f/%.1f attempts=%s t=%.2fs",
                result.input,
                result.output,
                result.avg_kbps,
                result.avg_limit_kbps,
                result.peak_kbps,
                result.peak_limit_kbps,
                result.attempts,
                result.encode_seconds,
            )
        else:
            log.error("FAILED %s: %s", result.input, result.reason)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
