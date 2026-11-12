#!/usr/bin/env python3
"""User-requested test (2026-11-11): ai_hard_19_1080 with SVT-AV1 preset 8 + tune 5 (VMAF), GOP 48.
Tune 5 only runs in CRF mode (SVT: "Tune VMAF only supports CRF rate control mode") and applies unsharp masking.
Variants: (a) CRF fitted to the clip's 0.05 bpp budget (2,488 kbps) -> mp4 and the same stream remuxed to MKV;
(b) CRF fitted to the user's 4,275 kbps target (1.72x budget, informational: breaks the average-bitrate rule).
Each output validated like the labels (canonical VMAF + VMAF-NEG, streaming rules, tone/chroma gates)."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encode.ratefit import encode_rate_fit
from streamenc.encoders.arms import FFMPEG
from streamenc.validate.validate_clip import validate_clip
SRC = "/workspace/dataset/ai1080/ai_hard_19_1080.mkv"
OUT = ROOT / "results/user_tests"
ARM = {"codec": "libsvtav1", "rc": "crf", "preset": 8, "svt": {"tune": 5}}
BUDGET = 0.05 * 1920 * 1080 * 24 / 1000

def val(path, tag, enc):
    r = validate_clip(SRC, str(path), bits_per_pixel=0.05, run_vmaf=True, run_perceptual=True)
    m = r.metrics
    row = {"variant": tag, "file": str(path), "crf": enc.get("aim"), "avg_kbps": enc.get("avg_kbps"),
           "budget_kbps": BUDGET, "use": (enc.get("avg_kbps") or 0) / BUDGET, "peak_kbps": enc.get("peak_kbps"),
           "vmaf": m.get("vmaf"), "vmaf_neg": m.get("vmaf_neg"), "gap": m.get("vmaf_delta"),
           "passed": r.passed, "clip_score": m.get("clip_score") if r.passed else 0.0,
           "violations": [v.code for v in r.violations], "enc_cpu_s": enc.get("enc_cpu_seconds"),
           "history": enc.get("history")}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "history"}), flush=True)
    with open(OUT / "t5_ai_hard_19_1080.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

for tag, bpp in (("p8_tune5_budget", 0.05), ("p8_tune5_4275kbps", 4275 / (1920 * 1080 * 24 / 1000))):
    out = OUT / f"ai_hard_19_1080_{tag}.mp4"
    enc = encode_rate_fit(ARM, SRC, str(out), bits_per_pixel=bpp, crf_hint=60.0 if bpp == 0.05 else 55.0)
    print(tag, "gop/fit:", enc.get("history"), enc.get("codes"), enc.get("error"), flush=True)
    if not enc["ok"]:
        print(tag, "no valid encode at this target", flush=True); continue
    val(out, tag, enc)
    if tag == "p8_tune5_budget":
        mkv = out.with_suffix(".mkv")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(out), "-c", "copy", str(mkv)], check=True)
        val(mkv, tag + "_mkv", enc)
