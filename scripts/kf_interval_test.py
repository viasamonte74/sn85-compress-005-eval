#!/usr/bin/env python3
"""User-requested test (2026-11-11): keyframes every 1 s / 1.5 s instead of 2 s (allowed: rule is "at most 2 s").
q08 and q07, p2, rate fit to the 0.05 bpp budget + full validation; counts keyframes in the output."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS, FFPROBE
from streamenc.encode.ratefit import encode_rate_fit
from streamenc.validate.validate_clip import validate_clip
P2 = ARMS["svt_crf_p2_qm_la"]
W = ROOT / "results/user_tests/kf"; W.mkdir(parents=True, exist_ok=True)
for clip, hint in (("pp005_q08", 55.5), ("pp005_q07", 57.25)):
    src = f"/workspace/dataset/practice005/{clip}.mkv"
    for name, ki in (("2.0s (48 frames)", None), ("1.5s (36 frames)", 36), ("1.0s (24 frames)", 24)):
        arm = {**P2, "svt": {**P2["svt"], **({"keyint": ki} if ki else {})}}
        out = W / f"{clip}_{ki or 48}.mp4"
        enc = encode_rate_fit(arm, src, str(out), crf_hint=hint + (0 if ki is None else 1.0))
        if not enc["ok"]:
            print(clip, name, "no valid encode", enc.get("codes"), enc.get("error"), flush=True); continue
        pk = json.loads(subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=flags",
                                        "-of", "json", str(out)], capture_output=True, text=True).stdout)["packets"]
        r = validate_clip(src, str(out)); m = r.metrics
        print(json.dumps({"clip": clip, "keyframes every": name, "keyframes": sum("K" in p["flags"] for p in pk), "crf": enc["aim"],
                          "use": round(enc["util"], 4), "vmaf": round(m["vmaf"], 3), "gap": round(m["vmaf"] - m["vmaf_neg"], 2),
                          "passed": r.passed, "violations": [v.code for v in r.violations]}), flush=True)
