#!/usr/bin/env python3
"""2026-11-11: do BT.601-tagged references distort our labels? scrr_003 (screen_real, tagged smpte170m) vs the
same pixels re-tagged BT.709 (lossless FFV1, pixels unchanged). One encode (tags BT.709, like every miner output)
is validated against both references; any VMAF difference is caused by the reference's colour tags alone."""
import subprocess, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import FFMPEG, ARMS, build_commands
from streamenc.validate.validate_clip import validate_clip
W = ROOT / "results/user_tests/colour"; W.mkdir(parents=True, exist_ok=True)
src = "/workspace/dataset/screen_real/scrr_003.mkv"
tag709 = W / "scrr_003_tag709.mkv"
subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src, "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709",
                "-c:v", "ffv1", "-level", "3", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(tag709)], check=True)
# same pixels? (decode both raw, ignore tags)
h = [subprocess.run([FFMPEG, "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], capture_output=True).stdout for p in (src, str(tag709))]
print("pixels identical:", h[0] == h[1], len(h[0]))
out = W / "scrr_003_p8_crf38.mp4"
for c in build_commands(ARMS["svt_crf_p8_t1"], src, str(out), 1152.0, 60, str(W / "pl"), crf=38.0):
    subprocess.run(c, check=True)
for name, ref in (("reference tagged BT.601 (as labelled)", src), ("reference re-tagged BT.709", str(tag709))):
    r = validate_clip(ref, str(out), bits_per_pixel=1.0)  # bpp 1.0: only the quality / colour gates matter here
    m = r.metrics
    print(json.dumps({"case": name, "vmaf": round(m.get("vmaf", 0), 3), "vmaf_neg": round(m.get("vmaf_neg", 0), 3),
                      "violations": [v.code for v in r.violations]}), flush=True)
