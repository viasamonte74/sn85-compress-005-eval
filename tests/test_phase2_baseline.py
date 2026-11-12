from __future__ import annotations

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

from streamenc.encode.baseline import encode_baseline
from streamenc.encode.packets import measure_packet_kbps
from streamenc.encoders import NvencAv1Config, build_ffmpeg_argv
from streamenc.probe import probe_video


SAMPLE = Path("/workspace/dataset/ai/ai_easy_01.mkv")
HARD = Path("/workspace/dataset/ai/ai_hard_01.mkv")
SCREEN = Path("/workspace/dataset/screen/scr_easy_01.mkv")


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset sample missing")
def test_build_argv_uses_matroska_and_mp4():
    argv = build_ffmpeg_argv(
        input_path="/tmp/deadbeef",
        output_path="/tmp/out",
        kbps=1000.0,
        gop=48,
        cfg=NvencAv1Config(),
    )
    assert argv[argv.index("-f") + 1] == "matroska"
    assert argv.count("-f") == 2
    assert argv[-2] == "mp4"
    assert "av1_nvenc" in argv
    assert "-rc" in argv and argv[argv.index("-rc") + 1] == "vbr"


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset sample missing")
def test_encode_extensionless_input_and_output(tmp_path: Path):
    # Competition naming: no extensions.
    src = tmp_path / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    dst = tmp_path / "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    src.write_bytes(SAMPLE.read_bytes())
    result = encode_baseline(str(src), str(dst), cfg=NvencAv1Config(preset="p6"))
    assert result.ok, result.reason
    assert dst.is_file() and dst.stat().st_size > 0
    assert result.frames_out == result.frames_in
    assert result.avg_kbps <= result.avg_limit_kbps
    assert result.peak_kbps <= result.peak_limit_kbps
    assert result.width == 1280 and result.height == 720


@pytest.mark.skipif(not HARD.exists(), reason="hard sample missing")
def test_encode_hard_clip_stays_in_budget(tmp_path: Path):
    dst = tmp_path / "hard_out"
    result = encode_baseline(str(HARD), str(dst), cfg=NvencAv1Config(preset="p6"))
    assert result.ok, result.reason
    assert result.avg_kbps <= result.avg_limit_kbps
    assert result.peak_kbps <= result.peak_limit_kbps


@pytest.mark.skipif(not SCREEN.exists(), reason="screen sample missing")
def test_encode_screen_1080p(tmp_path: Path):
    dst = tmp_path / "screen_out"
    result = encode_baseline(str(SCREEN), str(dst), cfg=NvencAv1Config(preset="p6"))
    assert result.ok, result.reason
    meta = probe_video(dst, count_frames=True)
    assert meta.width == 1920 and meta.height == 1080
    assert meta.nb_frames == result.frames_in


def test_run_sh_batch_contract(tmp_path: Path):
    if not SAMPLE.exists():
        pytest.skip("dataset sample missing")
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    name = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    (inp / name).write_bytes(SAMPLE.read_bytes())
    proc = subprocess.run(
        ["bash", str(ROOT / "run.sh"), str(inp), str(out)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "ITEM_TIMEOUT_SECONDS": "180",
            "BATCH_BUDGET_SECONDS": "300",
        },
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / name).is_file() and (out / name).stat().st_size > 0
    assert "done" in proc.stderr
