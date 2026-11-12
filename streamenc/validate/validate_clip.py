"""Standalone clip validator for sn85-compression-005.

Matches official streaming packet math and VMAF model/delta gates where available.
Sandbox output caps (512 MiB / file, 2 GiB / batch) are enforced as local gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from streamenc.probe import probe_video
from streamenc.score import clip_score
from streamenc.util.logging import get_logger
from streamenc.util.subprocess_run import run_command
from streamenc.validate.official_streaming import (
    ValidityViolation,
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)
from streamenc.validate.perceptual import run_perceptual_gates
from streamenc.validate.report import GateResult, ValidationReport
from streamenc.validate.vmaf import VMAF_MODEL_DELTA_MAX, canonicalize_pair, compute_vmaf_pair, default_ffmpeg_with_vmaf

log = get_logger("streamenc.validate")

MAX_OUTPUT_FILE_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_BATCH_BYTES = 2 * 1024 * 1024 * 1024
CODEC_ALLOWLIST = {"h264", "hevc", "vp9", "av1"}


def _probe_candidate(path: str, *, count_frames: bool = True):
    return probe_video(path, count_frames=count_frames)


def _decodable(path: str, frames: int, *, ffmpeg: str, timeout: float = 300.0) -> tuple[bool, str]:
    # Decode all frames to null; failure => undecodable.
    r = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        timeout=timeout,
    )
    if not r.ok:
        return False, (r.stderr or r.stdout or "decode failed")[:300]
    return True, f"decoded OK (ref_frames={frames})"


def validate_clip(
    reference: str,
    candidate: str,
    *,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    peak_window: float = 2.0,
    peak_ratio: float = 1.5,
    streaming_rate_tolerance: float = 0.01,
    run_vmaf: bool = True,
    run_perceptual: bool = True,
    ffmpeg: str | None = None,
    ffprobe: str = "ffprobe",
    batch_total_bytes: int | None = None,
) -> ValidationReport:
    ffmpeg = ffmpeg or default_ffmpeg_with_vmaf()
    gates: list[GateResult] = []
    violations: list[ValidityViolation] = []
    metrics: dict = {}
    unverified: list[str] = []

    # --- structural ---
    try:
        ref = _probe_candidate(reference)
        cand = _probe_candidate(candidate)
    except Exception as exc:  # noqa: BLE001
        report = ValidationReport(
            reference=reference,
            candidate=candidate,
            passed=False,
            gates=[GateResult("probe", "fail", str(exc))],
            violations=[ValidityViolation(code="UNDECODABLE", detail=str(exc))],
        )
        return report

    metrics["reference"] = ref.to_dict()
    metrics["candidate"] = cand.to_dict()

    if cand.codec_name.lower() not in CODEC_ALLOWLIST:
        violations.append(
            ValidityViolation(
                code="ENCODING_NOT_ALLOWED",
                detail=f"codec {cand.codec_name!r} not in allowlist {sorted(CODEC_ALLOWLIST)}",
            )
        )
        gates.append(GateResult("encoding", "fail", cand.codec_name))
    else:
        gates.append(GateResult("encoding", "pass", cand.codec_name))

    if (cand.width, cand.height) != (ref.width, ref.height):
        violations.append(
            ValidityViolation(
                code="STREAM_DIMENSIONS_MISMATCH",
                detail=f"candidate {cand.width}x{cand.height} vs reference {ref.width}x{ref.height}",
            )
        )
        gates.append(GateResult("dimensions", "fail", f"{cand.width}x{cand.height}"))
    else:
        gates.append(GateResult("dimensions", "pass", f"{cand.width}x{cand.height}"))

    if cand.nb_frames != ref.nb_frames:
        violations.append(
            ValidityViolation(
                code="FRAME_COUNT_MISMATCH",
                detail=f"candidate has {cand.nb_frames} frames, reference has {ref.nb_frames}",
                measured=float(cand.nb_frames),
                limit=float(ref.nb_frames),
            )
        )
        gates.append(GateResult("frame_count", "fail", f"{cand.nb_frames}!={ref.nb_frames}"))
    else:
        gates.append(GateResult("frame_count", "pass", str(cand.nb_frames)))

    # Competition requires preserving exact frame rate (rational).
    if Fraction(cand.r_frame_rate) != Fraction(ref.r_frame_rate):
        violations.append(
            ValidityViolation(
                code="STREAM_FRAME_RATE_MISMATCH",
                detail=f"candidate {cand.r_frame_rate} vs reference {ref.r_frame_rate}",
            )
        )
        gates.append(GateResult("frame_rate", "fail", cand.r_frame_rate))
    else:
        gates.append(GateResult("frame_rate", "pass", cand.r_frame_rate))

    ok_dec, dec_detail = _decodable(candidate, ref.nb_frames, ffmpeg=ffmpeg)
    if not ok_dec:
        violations.append(ValidityViolation(code="UNDECODABLE", detail=dec_detail))
        gates.append(GateResult("decodable", "fail", dec_detail))
    else:
        gates.append(GateResult("decodable", "pass", dec_detail))

    cand_bytes = Path(candidate).stat().st_size
    metrics["candidate_bytes"] = cand_bytes
    if cand_bytes > MAX_OUTPUT_FILE_BYTES:
        violations.append(
            ValidityViolation(
                code="OUTPUT_FILE_TOO_LARGE",
                detail=f"{cand_bytes} bytes exceeds {MAX_OUTPUT_FILE_BYTES}",
                measured=float(cand_bytes),
                limit=float(MAX_OUTPUT_FILE_BYTES),
            )
        )
        gates.append(GateResult("file_size_cap", "fail", f"{cand_bytes}"))
    else:
        gates.append(GateResult("file_size_cap", "pass", f"{cand_bytes}"))

    if batch_total_bytes is not None:
        if batch_total_bytes > MAX_OUTPUT_BATCH_BYTES:
            violations.append(
                ValidityViolation(
                    code="OUTPUT_BATCH_TOO_LARGE",
                    detail=f"batch {batch_total_bytes} bytes exceeds {MAX_OUTPUT_BATCH_BYTES}",
                    measured=float(batch_total_bytes),
                    limit=float(MAX_OUTPUT_BATCH_BYTES),
                )
            )
            gates.append(GateResult("batch_size_cap", "fail", str(batch_total_bytes)))
        else:
            gates.append(GateResult("batch_size_cap", "pass", str(batch_total_bytes)))
    else:
        gates.append(GateResult("batch_size_cap", "skip", "no batch total provided"))

    # --- streaming packet rules (official) ---
    target = streaming_target_kbps(
        width=ref.width, height=ref.height, fps=ref.fps, bits_per_pixel=bits_per_pixel
    )
    metrics["target_kbps"] = target
    try:
        packets = read_packets_ffprobe(candidate, ffprobe=ffprobe)
        measurement = measure_streaming(
            packets, fps=ref.fps, frames=ref.nb_frames, peak_window_seconds=peak_window
        )
        metrics["streaming"] = asdict(measurement)
        stream_vs = streaming_violations(
            measurement,
            target_kbps=target,
            peak_ratio=peak_ratio,
            peak_window_seconds=peak_window,
            max_keyframe_interval_seconds=max_keyframe_interval,
            fps=ref.fps,
            streaming_rate_tolerance=streaming_rate_tolerance,
        )
        violations.extend(stream_vs)
        if any(v.code == "STREAMING_BITRATE_EXCEEDED" for v in stream_vs):
            gates.append(GateResult("avg_bitrate", "fail", f"{measurement.avg_kbps:.2f}"))
        else:
            gates.append(
                GateResult(
                    "avg_bitrate",
                    "pass",
                    f"{measurement.avg_kbps:.2f}",
                    measured=measurement.avg_kbps,
                    limit=target * (1.0 + streaming_rate_tolerance),
                )
            )
        if any(v.code == "STREAMING_PEAK_EXCEEDED" for v in stream_vs):
            gates.append(GateResult("peak_bitrate", "fail", f"{measurement.peak_kbps:.2f}"))
        else:
            gates.append(
                GateResult(
                    "peak_bitrate",
                    "pass",
                    f"{measurement.peak_kbps:.2f}",
                    measured=measurement.peak_kbps,
                    limit=target * peak_ratio,
                )
            )
        if any(v.code == "STREAMING_KEYFRAME_INTERVAL" for v in stream_vs):
            gates.append(
                GateResult(
                    "keyframes",
                    "fail",
                    f"first={measurement.first_frame_is_keyframe} max_interval={measurement.max_keyframe_interval_seconds:.3f}",
                )
            )
        else:
            gates.append(
                GateResult(
                    "keyframes",
                    "pass",
                    f"first={measurement.first_frame_is_keyframe} max_interval={measurement.max_keyframe_interval_seconds:.3f}",
                    measured=measurement.max_keyframe_interval_seconds,
                    limit=max_keyframe_interval + 0.5 / ref.fps,
                )
            )
    except Exception as exc:  # noqa: BLE001
        violations.append(ValidityViolation(code="STREAMING_PACKETS_UNREADABLE", detail=str(exc)))
        gates.append(GateResult("avg_bitrate", "fail", str(exc)))
        gates.append(GateResult("peak_bitrate", "fail", str(exc)))
        gates.append(GateResult("keyframes", "fail", str(exc)))

    # --- canonicalize once (validator recipe), shared by VMAF and perceptual gates ---
    canon_ref, canon_cand = reference, candidate
    canon_tmp = None
    if run_vmaf or run_perceptual:
        try:
            import tempfile

            canon_tmp = tempfile.TemporaryDirectory(prefix="canon-")
            canon_ref, canon_cand = canonicalize_pair(reference, candidate, canon_tmp.name, ffmpeg=ffmpeg)
            metrics["canonicalized"] = True
        except Exception as exc:  # noqa: BLE001
            metrics["canonicalized"] = False
            unverified.append(f"canonicalization_failed: {exc}")
            canon_ref, canon_cand = reference, candidate

    # --- VMAF / VMAF-NEG ---
    if run_vmaf:
        try:
            pair = compute_vmaf_pair(
                canon_ref, canon_cand, ffmpeg=ffmpeg, canonicalize_inputs=not metrics.get("canonicalized", False)
            )
            metrics["vmaf"] = pair.vmaf
            metrics["vmaf_neg"] = pair.vmaf_neg
            metrics["vmaf_delta"] = pair.delta
            metrics["clip_score"] = clip_score(pair.vmaf)
            metrics["libvmaf_version"] = pair.libvmaf_version
            gates.append(GateResult("vmaf", "pass", f"{pair.vmaf:.6f}"))
            if pair.delta > VMAF_MODEL_DELTA_MAX:
                violations.append(
                    ValidityViolation(
                        code="VMAF_MODEL_DELTA_EXCEEDED",
                        detail="two VMAF model runs disagree beyond tolerance",
                        measured=pair.delta,
                        limit=VMAF_MODEL_DELTA_MAX,
                    )
                )
                gates.append(GateResult("vmaf_model_delta", "fail", f"{pair.delta:.6f}"))
            else:
                gates.append(
                    GateResult(
                        "vmaf_model_delta",
                        "pass",
                        f"{pair.delta:.6f}",
                        measured=pair.delta,
                        limit=VMAF_MODEL_DELTA_MAX,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            gates.append(GateResult("vmaf", "fail", str(exc)))
            gates.append(GateResult("vmaf_model_delta", "fail", str(exc)))
            violations.append(ValidityViolation(code="VMAF_UNAVAILABLE", detail=str(exc)))
            unverified.append("vmaf_measurement_failed")
    else:
        gates.append(GateResult("vmaf", "skip", "disabled"))
        gates.append(GateResult("vmaf_model_delta", "skip", "disabled"))
        unverified.append("vmaf_skipped")

    # --- tone / colour / chroma (official formulas) ---
    if run_perceptual:
        try:
            results, perc_vs = run_perceptual_gates(canon_ref, canon_cand)
            metrics["perceptual"] = [asdict(r) for r in results]
            violations.extend(perc_vs)
            for r in results:
                gates.append(
                    GateResult(
                        r.name,
                        "pass" if r.passed else "fail",
                        r.detail,
                        measured=r.measure,
                        limit=r.limit,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            for name in ("tone_manipulation", "color_grayscale", "chroma_uv"):
                gates.append(GateResult(name, "unverified", str(exc)))
            unverified.extend(["tone_manipulation", "color_grayscale", "chroma_uv"])
    else:
        for name in ("tone_manipulation", "color_grayscale", "chroma_uv"):
            gates.append(GateResult(name, "skip", "disabled"))
            unverified.append(name)

    # pix_fmt note: official canonicalize often forces yuv420p before compare.
    # We record mismatch diagnostically but do not invent a competition zero unless
    # the scorer's pre-canonical path rejects it. Label as informational.
    if cand.pix_fmt != ref.pix_fmt:
        gates.append(
            GateResult(
                "pix_fmt",
                "unverified",
                f"candidate {cand.pix_fmt!r} vs reference {ref.pix_fmt!r} "
                "(official path compares after canonicalization to 8-bit 4:2:0)",
            )
        )
        unverified.append("pix_fmt_pre_canonicalize")
    else:
        gates.append(GateResult("pix_fmt", "pass", cand.pix_fmt))

    hard_fail_codes = {
        "ENCODING_NOT_ALLOWED",
        "STREAM_DIMENSIONS_MISMATCH",
        "FRAME_COUNT_MISMATCH",
        "STREAM_FRAME_RATE_MISMATCH",
        "UNDECODABLE",
        "OUTPUT_FILE_TOO_LARGE",
        "OUTPUT_BATCH_TOO_LARGE",
        "STREAMING_BITRATE_EXCEEDED",
        "STREAMING_PEAK_EXCEEDED",
        "STREAMING_KEYFRAME_INTERVAL",
        "STREAMING_PACKETS_UNREADABLE",
        "VMAF_MODEL_DELTA_EXCEEDED",
        "VMAF_UNAVAILABLE",
        "TONE_MANIPULATION",
        "COLOR_GRAYSCALE",
        "CHROMA_UV_MANIPULATION",
    }
    passed = not any(v.code in hard_fail_codes for v in violations)
    if canon_tmp is not None:
        canon_tmp.cleanup()
    return ValidationReport(
        reference=reference,
        candidate=candidate,
        passed=passed,
        gates=gates,
        violations=violations,
        metrics=metrics,
        unverified=unverified,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--bits-per-pixel", type=float, default=0.05)
    parser.add_argument("--max-keyframe-interval", type=float, default=2.0)
    parser.add_argument("--peak-window", type=float, default=2.0)
    parser.add_argument("--peak-ratio", type=float, default=1.5)
    parser.add_argument("--no-vmaf", action="store_true")
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    report = validate_clip(
        args.reference,
        args.candidate,
        bits_per_pixel=args.bits_per_pixel,
        max_keyframe_interval=args.max_keyframe_interval,
        peak_window=args.peak_window,
        peak_ratio=args.peak_ratio,
        run_vmaf=not args.no_vmaf,
        run_perceptual=not args.no_perceptual,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        status = "PASS" if report.passed else "FAIL"
        print(f"{status} {args.candidate}")
        for g in report.gates:
            print(f"  [{g.status}] {g.name}: {g.detail}")
        for v in report.violations:
            print(f"  violation {v.code}: {v.detail}")
        if report.unverified:
            print(f"  unverified: {', '.join(report.unverified)}")
        if "vmaf" in report.metrics:
            print(
                f"  vmaf={report.metrics['vmaf']:.4f} "
                f"neg={report.metrics['vmaf_neg']:.4f} "
                f"delta={report.metrics['vmaf_delta']:.4f} "
                f"clip_score={report.metrics['clip_score']:.4f}"
            )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
