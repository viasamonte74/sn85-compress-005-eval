"""rc4 (2026-11-12): in-pipeline VMAF check settings (threads, validator-equivalent frame pairing)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.validate import vmaf  # noqa: E402


def test_default_threads(monkeypatch):
    monkeypatch.delenv("STREAMENC_QUICK_VMAF_THREADS", raising=False)
    assert vmaf._quick_threads() == (8, 4)
    monkeypatch.setenv("STREAMENC_QUICK_VMAF_THREADS", "1,1")
    assert vmaf._quick_threads() == (1, 1)
    monkeypatch.setenv("STREAMENC_QUICK_VMAF_THREADS", "junk")
    assert vmaf._quick_threads() == (1, 1)


def test_pairing_uses_the_reference_rate(monkeypatch):
    monkeypatch.delenv("STREAMENC_QUICK_VMAF_PAIRING", raising=False)
    assert vmaf._quick_pairing(SimpleNamespace(fps_fraction="30000/1001", fps=29.97)) == "settb=1/30000,setpts=N*1001"
    assert vmaf._quick_pairing(SimpleNamespace(fps_fraction="50", fps=50.0)) == "settb=1/50,setpts=N*1"
    assert vmaf._quick_pairing(None) == "setpts=N/FRAME_RATE/TB"
    monkeypatch.setenv("STREAMENC_QUICK_VMAF_PAIRING", "old")
    assert vmaf._quick_pairing(SimpleNamespace(fps_fraction="24", fps=24.0)) == "setpts=N/FRAME_RATE/TB"
    assert vmaf._quick_pairing(SimpleNamespace(fps_fraction="24", fps=24.0), force=True) == "settb=1/24,setpts=N*1"
