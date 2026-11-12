#!/usr/bin/env python3
"""Feasibility of per-GOP CRF mixing (2026-11-11): q08 encoded (a) whole at CRF 55.5, (b) as 3 independent segments
(frames 0-47, 48-95, 96-120, each starting with a keyframe) at CRF 55.5 joined into one AV1 stream, (c) segments at
mixed CRFs 55.25/55.5. Validated like the labels (frame count, keyframes, bitrate, VMAF, NEG, chroma/tone)."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS, FFMPEG
from streamenc.validate.validate_clip import validate_clip
SRC = "/workspace/dataset/practice005/pp005_q08.mkv"; W = ROOT / "results/user_tests/gopmix"; W.mkdir(parents=True, exist_ok=True)
arm = ARMS["svt_crf_p2_qm_la"]; svt = {"keyint": 48, "scd": 0, "lp": 8, **arm["svt"]}
def enc(inp_args, crf, out):
    p = ":".join(f"{k}={v}" for k, v in {**svt, "crf": f"{crf:.2f}"}.items())
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *inp_args, "-an", "-c:v", "libsvtav1", "-preset", "2",
                    "-g", "48", "-svtav1-params", p, "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-color_range", "tv", str(out)], check=True)
def seg_encode(crfs, tag):
    parts = []
    for i, (a, b) in enumerate(((0, 48), (48, 96), (96, 121))):
        o = W / f"{tag}_s{i}.ivf"
        enc(["-i", SRC, "-vf", f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS", "-r", "24"], crfs[i], o)
        parts.append(o)
    lst = W / f"{tag}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = W / f"{tag}.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-video_track_timescale", "24", str(out)], check=True)
    return out
def report(out, tag):
    r = validate_clip(SRC, str(out), bits_per_pixel=0.05); m = r.metrics
    print(json.dumps({"tag": tag, "passed": r.passed, "vmaf": round(m.get("vmaf", 0), 3), "neg": round(m.get("vmaf_neg", 0), 3),
                      "avg_kbps": round(m.get("avg_kbps") or m.get("streaming_avg_kbps") or 0, 1),
                      "violations": [v.code for v in r.violations], "metrics_keys": sorted(k for k in m if "kbps" in k or "util" in k)}), flush=True)
whole = W / "whole_55.5.mp4"; enc(["-i", SRC], 55.5, whole); report(whole, "whole crf 55.5")
report(seg_encode([55.5, 55.5, 55.5], "seg_55.5"), "3 segments crf 55.5")
report(seg_encode([55.25, 55.5, 55.5], "seg_mix1"), "segments 55.25/55.5/55.5")
report(seg_encode([55.25, 55.25, 55.5], "seg_mix2"), "segments 55.25/55.25/55.5")
whole2 = W / "whole_55.25.mp4"; enc(["-i", SRC], 55.25, whole2); report(whole2, "whole crf 55.25")
