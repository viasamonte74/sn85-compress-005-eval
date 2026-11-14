"""2026-11-13 colour fix: untagged sources are relabelled as BT.709 (like the validator's canonicalization), never
converted by ffmpeg; tagged sources keep exactly the old commands."""
import os
import subprocess
import tempfile

import pytest

from streamenc.encoders.arms import FFMPEG, build_commands, get_arm
from streamenc.validate.vmaf import build_canonicalization_plan, relabel_vf, untagged_color_fill


def test_relabel_only_for_untagged_yuv():
    assert relabel_vf("yuv420p", None, "tv") == "setparams=colorspace=bt709"
    assert relabel_vf("yuv420p", "unknown", None) == "setparams=colorspace=bt709:range=tv"
    assert relabel_vf("yuv420p", "bt709", "tv") is None
    assert relabel_vf("yuv420p", "bt470bg", "tv") is None          # a stated matrix is never overridden
    assert relabel_vf("rgb24", None, None) is None


def test_scoring_fills_missing_tags_like_the_validator():
    # validator probe: a missing tag is reported as "unknown" -> the reference is relabelled, not converted
    fill = untagged_color_fill("yuv420p", None or "unknown", "tv")
    plan = build_canonicalization_plan("in.mkv", "o.y4m", **fill)
    assert plan[plan.index("-vf") + 1].startswith("setparams=colorspace=bt709,")


def test_tagged_commands_unchanged():
    a = build_commands(get_arm("svt_crf_p4_qm_la"), "in.mkv", "o.mp4", 1000.0, 48, "/tmp/x", "boxblur=1", crf=40)[0]
    b = build_commands(get_arm("svt_crf_p4_qm_la"), "in.mkv", "o.mp4", 1000.0, 48, "/tmp/x", "boxblur=1", crf=40, pre_vf=None)[0]
    assert a == b and a[a.index("-vf") + 1] == "boxblur=1"
    c = build_commands(get_arm("svt_crf_p4_qm_la"), "in.mkv", "o.mp4", 1000.0, 48, "/tmp/x", "boxblur=1", crf=40,
                       pre_vf="setparams=colorspace=bt709")[0]
    assert c[c.index("-vf") + 1] == "setparams=colorspace=bt709,boxblur=1"


@pytest.mark.skipif(not os.path.isfile(FFMPEG) and FFMPEG == "ffmpeg", reason="needs the pinned ffmpeg")
def test_untagged_encode_keeps_pixels():
    """An untagged source encoded losslessly-ish with the relabel keeps its luma; without it ffmpeg converts."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.mkv")
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=24:duration=0.5", "-pix_fmt", "yuv420p", "-c:v", "ffv1", src], check=True)
        def luma(vf_pre):
            out = os.path.join(d, f"o{bool(vf_pre)}.mp4")
            cmd = build_commands(get_arm("svt_crf_p8_t1"), src, out, 500.0, 12, os.path.join(d, "p"), None, crf=1, pre_vf=vf_pre)[0]
            subprocess.run(cmd, check=True, capture_output=True)
            r = subprocess.run([FFMPEG, "-hide_banner", "-i", out, "-i", src, "-lavfi",
                                "[0:v]extractplanes=y[a];[1:v]extractplanes=y[b];[a][b]psnr", "-f", "null", "-"],
                               capture_output=True, text=True).stderr
            return float(r.split("average:")[1].split()[0].replace("inf", "99"))
        assert luma("setparams=colorspace=bt709") > luma(None) + 3.0
