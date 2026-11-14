"""1.0.0rc16: raw-cache prefetch (same calls, same results), P2 exact pick, P3 /tmp reservation."""
import importlib
import inspect
import shutil
import subprocess
from collections import namedtuple

import pytest


def _tiny_clip(tmp_path):
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg not on PATH")
    p = tmp_path / "tiny.mkv"
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=0.5",
                    "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_switches_default_on():
    import streamenc.encode.crf_pipeline as cp
    cp = importlib.reload(cp)
    assert cp.RAW_PREFETCH is True and cp.NEG_EXACT_PICK is True
    assert not hasattr(cp, "NEG_FINAL")            # rc15's P1 (time cost) is not part of rc16


def test_prefetch_matches_direct_calls(tmp_path, monkeypatch):
    import streamenc.encode.crf_pipeline as cp
    clip = _tiny_clip(tmp_path)
    monkeypatch.setattr(cp.tempfile, "gettempdir", lambda: str(tmp_path))
    meta, pts, raw, t_probe, t_raw = cp._prefetch_source(str(clip))
    assert meta == cp.probe_video(str(clip), count_frames=True)
    assert pts == cp.packet_pts(str(clip))
    path, d, _ = raw
    assert path is not None and d is not None and cp._RAW_RESERVED[0] == 0.0
    shutil.rmtree(d)


def test_prefetch_wired_into_both_functions():
    import streamenc.encode.crf_pipeline as cp
    enc = inspect.getsource(cp.encode_item_crf)
    assert "y4m, raw_dir, raw_cpu = pre[2]" in enc and "_raw_source(input_path, meta, pts)" in enc   # fallback kept
    item = inspect.getsource(cp.process_item_crf)
    assert "pre = pool_.submit(_prefetch_source, input_path)" in item and "prefetch=pre" in item
    assert "shutil.rmtree(rd, ignore_errors=True)" in item


def test_exact_pick_wired():
    import streamenc.encode.crf_pipeline as cp
    src = inspect.getsource(cp.encode_item_crf)
    assert "if NEG_EXACT_PICK and len(tried) > 1 and min(r[0] for r in tried) > NEG_SAFE:" in src


def test_raw_cache_counts_reserved_space(tmp_path, monkeypatch):
    import streamenc.encode.crf_pipeline as cp
    clip = _tiny_clip(tmp_path)
    Meta = namedtuple("Meta", "pix_fmt width height nb_frames")
    meta = Meta("yuv420p", 64, 48, 12)
    monkeypatch.setattr(cp.tempfile, "gettempdir", lambda: str(tmp_path))
    real = shutil.disk_usage(str(tmp_path))
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(cp.shutil, "disk_usage", lambda _p: Usage(real.total, real.used, int(1.5e9)))
    monkeypatch.setattr(cp, "_RAW_RESERVED", [1.0e9])      # another clip is writing 1 GB right now
    raw, d, _ = cp._raw_source(str(clip), meta, [])
    assert raw is None and cp._RAW_RESERVED[0] == 1.0e9
