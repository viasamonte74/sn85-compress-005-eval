#!/usr/bin/env python3
"""q07 / q08 (practice pack AI clips) preprocessing + encoder study, 2026-11-11 (user's list: spatial / temporal
denoise, sharpening (measurement only: MINING.md calls it metric gaming), chroma filtering) + slower presets and
in-encoder temporal filter. Every trial = label trial (rate fit at 0.05 bpp + full validation)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS
from streamenc.ml.encoder_rank.labeling import run_label_trial
P2 = dict(ARMS["svt_crf_p2_qm_la"])
for name, extra in (("tf2", {"tf-strength": 2}), ("tf3", {"tf-strength": 3}), ("tf4", {"tf-strength": 4})):
    ARMS[f"q_p2_{name}"] = {**P2, "svt": {**P2["svt"], **extra}}
ARMS["q_p2_lqb"] = {**P2, "svt": {**P2["svt"], "luminance-qp-bias": 5}}
ARMS["q_p1_lqb"] = {**ARMS["svt_crf_p1_qm_la"], "svt": {**ARMS["svt_crf_p1_qm_la"]["svt"], "luminance-qp-bias": 5}}
PRE = ["cblur4", "cblur8", "cblur4x2", "c410", "cblur4_ds85", "bm3d_s3"]   # round 2 (round 1 list: see results file)
CFG = [f"svt_crf_p2_qm_la+{p}" for p in PRE] + [f"svt_crf_p1_qm_la+{p}" for p in ("cblur4", "cblur8", "c410", "cblur4_ds85")] \
      + ["svt_crf_p2_qm_la"]
OUT = ROOT / "results/user_tests/q0708_study2.jsonl"
done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
for clip, hint in (("pp005_q08", 55.5), ("pp005_q07", 57.25)):
    path = Path(f"/workspace/dataset/practice005/{clip}.mkv")
    todo = [c for c in CFG if (clip, c) not in done]
    with ThreadPoolExecutor(6) as ex:
        for t in ex.map(lambda c: run_label_trial(path, c, ROOT / "work/q0708", crf_hint=hint), todo):
            with open(OUT, "a") as f: f.write(json.dumps(t) + "\n")
            g = (t["vmaf"] - t["vmaf_neg"]) if t.get("vmaf") else None
            print(f"{clip} {t['config_id']:36} pass={t['passed']} vmaf={t['vmaf']} gap={g and round(g,2)} use={t.get('util')} cpu={t.get('enc_cpu_seconds')} {t['codes']}", flush=True)
