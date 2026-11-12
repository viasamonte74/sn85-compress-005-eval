from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure NVENC encode libs are visible before encoder smokes.
_ENC = "/opt/nvidia-drivers/lib64"
if Path(_ENC).is_dir():
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if _ENC not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_ENC}:{cur}" if cur else _ENC

from streamenc.env_check import collect_env_report
from streamenc.probe import probe_video
from streamenc.score import clip_score, final_score, speed_multiplier, speed_score
from streamenc.util.budget import StreamingBudget


def test_competition_config_matches_official_manifest():
    cfg = json.loads((ROOT / "configs" / "competition_005.json").read_text())
    assert cfg["competition_id"] == "sn85-compression-005"
    rules = cfg["streaming_rules"]
    assert rules["bits_per_pixel"] == 0.05
    assert rules["max_keyframe_interval_seconds"] == 2.0
    assert rules["peak_ratio"] == 1.5
    assert rules["peak_window_seconds"] == 2.0
    assert cfg["result_rules"]["speed_inserted_frames"] == 3428
    assert cfg["run_contract"]["invocation"] == "/bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR"


def test_budget_formula_720p24():
    b = StreamingBudget(0.05, 1280, 720, 24.0)
    assert b.target_bps == 0.05 * 1280 * 720 * 24
    assert math.isclose(b.target_kbps, b.target_bps / 1000.0)
    assert b.gop_frames(2.0) == 48


def test_budget_formula_1080p30():
    b = StreamingBudget(0.05, 1920, 1080, 30.0)
    # 0.05 * 1920 * 1080 * 30 = 3_110_400 bps = 3110.4 kbps
    assert math.isclose(b.target_bps, 3_110_400.0)
    assert b.gop_frames(2.0) == 60


def test_score_helpers():
    assert clip_score(60) == 0.0
    assert clip_score(100) == 1.0
    assert clip_score(80) == 0.5
    assert speed_score(3428 / 30.0, 3428) == pytest.approx(1.0, abs=1e-9)
    assert speed_score(3428 * 2.0, 3428) == pytest.approx(0.0, abs=1e-9)
    assert speed_multiplier(1.0) == pytest.approx(1.0)
    assert speed_multiplier(0.0) == pytest.approx(0.7)
    assert final_score(1.0, 3428 / 30.0) == pytest.approx(1.0, abs=1e-9)


def test_probe_sample_ffv1():
    sample = Path("/workspace/dataset/ai/ai_easy_01.mkv")
    if not sample.exists():
        pytest.skip("dataset sample missing")
    meta = probe_video(sample, count_frames=True)
    assert meta.codec_name == "ffv1"
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.r_frame_rate == "24/1"
    assert meta.nb_frames > 0
    budget = meta.budget()
    assert budget.target_kbps > 0


def test_env_report_required_checks():
    report = collect_env_report(work_dir=ROOT / "work" / "pytest_env")
    failed = [c for c in report.checks if c.required_for_phase1 and not c.ok]
    assert not failed, json.dumps([c.__dict__ for c in failed], indent=2)


def test_run_sh_is_executable_contract_shell():
    run_sh = ROOT / "run.sh"
    assert run_sh.is_file()
    text = run_sh.read_text()
    assert "INPUT_DIR" in text and "OUTPUT_DIR" in text
    assert "BITS_PER_PIXEL=${BITS_PER_PIXEL:-0.05}" in text
