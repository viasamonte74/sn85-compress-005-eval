#!/usr/bin/env python3
"""Per-frame VMAF / VMAF-NEG and packet sizes of q08 (p2, CRF 55.5 = its label fit), canonicalised the validator's
way, to see where VMAF is lost (keyframes at 0/48/96, motion bursts, ...). -> results/user_tests/q08_frames.json"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS, FFMPEG, FFPROBE, build_commands
from streamenc.validate.vmaf import canonicalize_pair, default_ffmpeg_with_vmaf
SRC = "/workspace/dataset/practice005/pp005_q08.mkv"
W = ROOT / "results/user_tests"; out = W / "q08_p2_crf55.5.mp4"
for c in build_commands(ARMS["svt_crf_p2_qm_la"], SRC, str(out), 1105.92, 48, str(W / "pl08"), crf=55.5):
    subprocess.run(c, check=True)
pk = json.loads(subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size,flags,pts_time",
                                "-of", "json", str(out)], capture_output=True, text=True).stdout)["packets"]
ff = default_ffmpeg_with_vmaf(); res = {}
with tempfile.TemporaryDirectory() as tmp:
    ref, cand = canonicalize_pair(SRC, str(out), tmp)
    for name, model in (("vmaf", "version=vmaf_v0.6.1"), ("neg", "version=vmaf_v0.6.1neg")):
        log = os.path.join(tmp, f"{name}.json")
        subprocess.run([ff, "-v", "error", "-i", cand, "-i", ref, "-filter_complex",
                        f"[0:v:0][1:v:0]libvmaf=model={model}:log_fmt=json:log_path={log}:n_threads=4", "-f", "null", "-"], check=True)
        res[name] = [f["metrics"]["vmaf"] for f in json.load(open(log))["frames"]]
        res[name + "_features"] = [{k: round(v, 4) for k, v in f["metrics"].items()} for f in json.load(open(log))["frames"]][:3]
pk.sort(key=lambda p: float(p["pts_time"]))
rows = [{"i": i, "key": "K" in p["flags"], "bytes": int(p["size"]), "vmaf": round(res["vmaf"][i], 2), "neg": round(res["neg"][i], 2)} for i, p in enumerate(pk)]
json.dump({"rows": rows, "features_first3": res["vmaf_features"]}, open(W / "q08_frames.json", "w"), indent=1)
import statistics as st
print("mean vmaf", round(st.mean(res["vmaf"]), 3), "neg", round(st.mean(res["neg"]), 3), "kbps", round(sum(r["bytes"] for r in rows) * 8 / (len(rows) / 24) / 1000, 1))
for r in rows:
    if r["key"] or r["i"] % 6 == 0: print(r)
