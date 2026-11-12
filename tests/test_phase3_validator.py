from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_ENC = "/opt/nvidia-drivers/lib64"
if Path(_ENC).is_dir():
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if _ENC not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_ENC}:{cur}" if cur else _ENC

from streamenc.validate import (
    StreamingPacket,
    measure_streaming,
    streaming_target_kbps,
    streaming_violations,
    validate_clip,
)
from streamenc.validate.vmaf import default_ffmpeg_with_vmaf

FPS = 30.0
SAMPLE = Path("/workspace/dataset/ai/ai_easy_01.mkv")
GOOD = ROOT / "work" / "phase2_pilot" / "ai_easy_01"


def _packets(sizes: list[int], keyframes: set[int]) -> list[StreamingPacket]:
    return [
        StreamingPacket(size_bytes=size, pts_seconds=i / FPS, keyframe=i in keyframes)
        for i, size in enumerate(sizes)
    ]


def test_official_budget_formula():
    assert streaming_target_kbps(width=1920, height=1080, fps=30.0, bits_per_pixel=0.05) == pytest.approx(
        3110.4
    )


def test_official_peak_window_math():
    sizes = [1250] * 120
    sizes[60:120] = [2500] * 60
    m = measure_streaming(_packets(sizes, {0, 60}), fps=FPS, frames=120, peak_window_seconds=2.0)
    assert m.avg_kbps == pytest.approx(8 * (60 * 1250 + 60 * 2500) / 4.0 / 1000)
    assert m.peak_kbps == pytest.approx(8 * 60 * 2500 / 2.0 / 1000)
    assert m.max_keyframe_interval_seconds == pytest.approx(2.0)
    assert m.first_frame_is_keyframe and m.keyframes == 2


def test_official_violation_thresholds():
    from streamenc.validate.official_streaming import StreamingMeasurement

    def m(**overrides):
        base = dict(
            avg_kbps=3000.0,
            peak_kbps=4000.0,
            max_keyframe_interval_seconds=2.0,
            first_frame_is_keyframe=True,
            keyframes=3,
            packets=180,
        )
        base.update(overrides)
        return StreamingMeasurement(**base)

    target = 3110.4
    kwargs = dict(
        target_kbps=target,
        peak_ratio=1.5,
        peak_window_seconds=2.0,
        max_keyframe_interval_seconds=2.0,
        fps=FPS,
        streaming_rate_tolerance=0.01,
    )
    assert streaming_violations(m(), **kwargs) == []
    assert streaming_violations(m(avg_kbps=target * 1.009), **kwargs) == []
    assert [v.code for v in streaming_violations(m(avg_kbps=target * 1.02), **kwargs)] == [
        "STREAMING_BITRATE_EXCEEDED"
    ]
    assert [v.code for v in streaming_violations(m(peak_kbps=target * 1.51), **kwargs)] == [
        "STREAMING_PEAK_EXCEEDED"
    ]
    assert [v.code for v in streaming_violations(m(first_frame_is_keyframe=False), **kwargs)] == [
        "STREAMING_KEYFRAME_INTERVAL"
    ]
    assert streaming_violations(m(max_keyframe_interval_seconds=2.0 + 0.4 / FPS), **kwargs) == []


@pytest.mark.skipif(not SAMPLE.exists() or not GOOD.exists(), reason="phase2 pilot encode missing")
def test_validate_good_baseline_encode():
    report = validate_clip(str(SAMPLE), str(GOOD), run_vmaf=True, run_perceptual=True)
    assert report.passed, json.dumps(report.to_dict(), indent=2)
    assert report.metrics["vmaf"] > 60
    assert report.metrics["vmaf_delta"] <= 3.0


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset missing")
def test_invalid_wrong_resolution_fails(tmp_path: Path):
    ffmpeg = default_ffmpeg_with_vmaf()
    out = tmp_path / "bad_res.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=24:duration=1",
            "-frames:v",
            "24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "48",
            str(out),
        ],
        check=True,
    )
    # Use a real reference with different geometry
    report = validate_clip(str(SAMPLE), str(out), run_vmaf=False, run_perceptual=False)
    assert not report.passed
    assert any(v.code == "STREAM_DIMENSIONS_MISMATCH" for v in report.violations)


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset missing")
def test_invalid_long_gop_fails(tmp_path: Path):
    ffmpeg = default_ffmpeg_with_vmaf()
    # Encode reference geometry with huge GOP.
    out = tmp_path / "long_gop.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(SAMPLE),
            "-an",
            "-c:v",
            "libx264",
            "-b:v",
            "800k",
            "-maxrate",
            "1000k",
            "-bufsize",
            "2000k",
            "-g",
            "300",
            "-keyint_min",
            "300",
            "-sc_threshold",
            "0",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    report = validate_clip(str(SAMPLE), str(out), run_vmaf=False, run_perceptual=False)
    assert not report.passed
    assert any(v.code == "STREAMING_KEYFRAME_INTERVAL" for v in report.violations)


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset missing")
def test_invalid_over_budget_fails(tmp_path: Path):
    ffmpeg = default_ffmpeg_with_vmaf()
    out = tmp_path / "fat.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(SAMPLE),
            "-an",
            "-c:v",
            "libx264",
            "-b:v",
            "8000k",
            "-maxrate",
            "12000k",
            "-bufsize",
            "16000k",
            "-g",
            "48",
            "-sc_threshold",
            "0",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    report = validate_clip(str(SAMPLE), str(out), run_vmaf=False, run_perceptual=False)
    assert not report.passed
    codes = {v.code for v in report.violations}
    assert "STREAMING_BITRATE_EXCEEDED" in codes or "STREAMING_PEAK_EXCEEDED" in codes


def test_junk_file_fails(tmp_path: Path):
    ref = SAMPLE if SAMPLE.exists() else None
    if ref is None:
        pytest.skip("dataset missing")
    junk = tmp_path / "junk"
    junk.write_bytes(b"not a video")
    report = validate_clip(str(ref), str(junk), run_vmaf=False, run_perceptual=False)
    assert not report.passed
