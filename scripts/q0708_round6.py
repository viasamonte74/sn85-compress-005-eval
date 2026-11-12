#!/usr/bin/env python3
"""Round 6 (2026-11-12, user: "solve the q08 issue first"; q08 = 59.70 VMAF, score 0).
Round 5 showed cblur8_ds85 +0.21 / +0.03 / +0.22 / +0.24 budget-adjusted over cblur4_ds85 on q08 / q07 /
ai_hard_12_1080 / ai_hard_21_1080 (consistent, not noise). Here, on the same arm p3_qm_la_lqb (zero time cost vs the
shipped hard path): luma low-pass strength 80 / 90 % and shape (lanczos down, luma-only gaussian 0.6) under cblur8,
and cblur16_ds85 (chroma gates are in validate_clip). Compare against round 5's cblur4_ds85 / cblur8_ds85 rows.
-> results/user_tests/q0708_round6.jsonl"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.ml.encoder_rank.labeling import run_label_trial
A = "svt_crf_p3_qm_la_lqb"
CFG = [f"{A}+{f}" for f in ("cblur8_ds80", "cblur8_ds90", "cblur8_ds85l", "cblur8_gb6", "cblur16_ds85")]
CLIPS = [("pp005_q08", "/workspace/dataset/practice005/pp005_q08.mkv", 57.0),
         ("pp005_q07", "/workspace/dataset/practice005/pp005_q07.mkv", 54.75),
         ("ai_hard_12_1080", "/workspace/dataset/ai1080/ai_hard_12_1080.mkv", None),
         ("ai_hard_21_1080", "/workspace/dataset/ai1080/ai_hard_21_1080.mkv", None)]
OUT = ROOT / "results/user_tests/q0708_round6.jsonl"
done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
jobs = [(c, p, h, cfg) for c, p, h in CLIPS for cfg in CFG if (Path(p).stem, cfg) not in done and (c, cfg) not in done]
def one(j):
    c, p, h, cfg = j
    return run_label_trial(Path(p), cfg, ROOT / "work/q0708r6", crf_hint=h)
with ThreadPoolExecutor(6) as ex:
    for t in ex.map(one, jobs):
        with open(OUT, "a") as f: f.write(json.dumps(t) + "\n")
        g = (t["vmaf"] - t["vmaf_neg"]) if t.get("vmaf") else None
        print(f"{t['clip_id']} {t['config_id']:34} pass={t['passed']} vmaf={t['vmaf'] and round(t['vmaf'],2)} gap={g and round(g,2)} "
              f"use={t.get('util') and round(t['util'],3)} crf={t.get('aim')} cpu={t.get('enc_cpu_seconds') and round(t['enc_cpu_seconds'])} {t['codes']}", flush=True)
