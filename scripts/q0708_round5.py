#!/usr/bin/env python3
"""Round 5 (2026-11-11, user: "reduce bitrate on factors the scoring does not reflect, spend it on VMAF").
Validator facts (validator-vidaio scoring/backends_real.py): VMAF v0.6.1, every frame (n_subsample=1), mean pooled,
luma only; NEG only gates the gap. Chroma is already mostly spent: on a 24-frame q08 cut at CRF 50 chroma costs 5.0 %
of the bits, 3.5 % after cblur4, 2.5 % after cblur8; SVT 4.3 chroma qindex offsets are no-ops (even with fixed
offsets), chroma-qm-min/max nearly so. Tested here, all on the current hard-clip arm p3_lqb at budget, against
cblur4_ds85 (the shipped-candidate hard path): in-loop superres (codes a narrower frame, decoder upscales; VMAF gives
little weight to the finest luma detail), cblur8_ds85, AQ mode 0 / 1 (default 2), hierarchical levels 4.
Clips: practice q07 / q08 + held-out ai_hard_12_1080 / ai_hard_21_1080 (guard against fitting two clips).
-> results/user_tests/q0708_round5.jsonl"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS
from streamenc.ml.encoder_rank.labeling import run_label_trial
B = ARMS["svt_crf_p3_qm_la_lqb"]
def v(**kw):
    return {**B, "svt": {**B["svt"], **{k.replace("_", "-"): x for k, x in kw.items()}}}
ARMS.update({
    "r5_sr9": v(superres_mode=1, superres_denom=9, superres_kf_denom=9),
    "r5_sr10": v(superres_mode=1, superres_denom=10, superres_kf_denom=10),
    "r5_aq0": v(aq_mode=0), "r5_aq1": v(aq_mode=1),
    "r5_hl4": v(hierarchical_levels=4),
})
CFG = ["svt_crf_p3_qm_la_lqb+cblur4_ds85", "svt_crf_p3_qm_la_lqb+cblur4", "r5_sr9+cblur4", "r5_sr10+cblur4",
       "svt_crf_p3_qm_la_lqb+cblur8_ds85", "r5_aq0+cblur4_ds85", "r5_aq1+cblur4_ds85", "r5_hl4+cblur4_ds85"]
CLIPS = [("pp005_q08", "/workspace/dataset/practice005/pp005_q08.mkv", 57.0),
         ("pp005_q07", "/workspace/dataset/practice005/pp005_q07.mkv", 54.75),
         ("ai_hard_12_1080", "/workspace/dataset/ai1080/ai_hard_12_1080.mkv", None),
         ("ai_hard_21_1080", "/workspace/dataset/ai1080/ai_hard_21_1080.mkv", None)]
OUT = ROOT / "results/user_tests/q0708_round5.jsonl"
done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
jobs = [(c, p, h, cfg) for c, p, h in CLIPS for cfg in CFG if (Path(p).stem, cfg) not in done and (c, cfg) not in done]
def one(j):
    c, p, h, cfg = j
    return run_label_trial(Path(p), cfg, ROOT / "work/q0708r5", crf_hint=h)
with ThreadPoolExecutor(6) as ex:
    for t in ex.map(one, jobs):
        with open(OUT, "a") as f: f.write(json.dumps(t) + "\n")
        g = (t["vmaf"] - t["vmaf_neg"]) if t.get("vmaf") else None
        print(f"{t['clip_id']} {t['config_id']:34} pass={t['passed']} vmaf={t['vmaf'] and round(t['vmaf'],2)} gap={g and round(g,2)} "
              f"use={t.get('util') and round(t['util'],3)} crf={t.get('aim')} cpu={t.get('enc_cpu_seconds') and round(t['enc_cpu_seconds'])} {t['codes']}", flush=True)
