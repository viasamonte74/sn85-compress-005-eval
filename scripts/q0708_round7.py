#!/usr/bin/env python3
"""Round 7 (2026-11-12, user: solve q08 via the NW-AV1 plan's luma/chroma-separated preprocessing, safely).
cblur16_ds85 helped most hard AI clips (+0.34..+0.56) but hurt anim_hard_007 (-0.78 / -0.39) and pushed the
hfr_rnd_hard_03 NEG gap 2.40 -> 3.06. Test edge-preserving chroma smoothing (smartblur with a flat-area threshold,
chroma-plane bilateral; luma untouched) against cblur4_ds85 (shipped) and cblur16_ds85, on the clips cblur16 helped
AND hurt, each on the arm the pipeline chose for it. -> results/user_tests/q0708_round7.jsonl"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.ml.encoder_rank.labeling import run_label_trial
FILTERS = ("cblur4_ds85", "cblur16_ds85", "csm5_8_ds85", "csm5_20_ds85", "cbil8_ds85")
CLIPS = [("pp005_q08", "/workspace/dataset/practice005/pp005_q08.mkv", "svt_crf_p3_qm_la_lqb", 57.0),
         ("pp005_q07", "/workspace/dataset/practice005/pp005_q07.mkv", "svt_crf_p3_qm_la", 54.75),
         ("ai_hard_12_1080", "/workspace/dataset/ai1080/ai_hard_12_1080.mkv", "svt_crf_p3_qm_la_lqb", None),
         ("ai_hard_21_1080", "/workspace/dataset/ai1080/ai_hard_21_1080.mkv", "svt_crf_p3_qm_la", None),
         ("anim_hard_007_i60_1080", "/workspace/dataset/hfr_ai/anim_hard_007_i60_1080.mkv", "svt_crf_p4_qm_la_lqb", 53.5),
         ("anim_hard_007_1080", "/workspace/dataset/ai_gen2/anim_hard_007_1080.mkv", "svt_crf_p5_qm_la", None),
         ("hfr_rnd_hard_03", "/workspace/dataset/hfr_render/hfr_rnd_hard_03.mkv", "svt_crf_p2_qm_la", 57.75),
         ("aig3_medium_004_1080", "/workspace/dataset/ai_gen3/aig3_medium_004_1080.mkv", "svt_crf_p5_qm_la", None)]
OUT = ROOT / "results/user_tests/q0708_round7.jsonl"
done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
jobs = [(c, p, a, h, f"{a}+{f}") for c, p, a, h in CLIPS for f in FILTERS]
jobs = [j for j in jobs if (Path(j[1]).stem, j[4]) not in done and (j[0], j[4]) not in done]
def one(j):
    c, p, a, h, cfg = j
    return run_label_trial(Path(p), cfg, ROOT / "work/q0708r7", crf_hint=h)
with ThreadPoolExecutor(6) as ex:
    for t in ex.map(one, jobs):
        with open(OUT, "a") as f: f.write(json.dumps(t) + "\n")
        g = (t["vmaf"] - t["vmaf_neg"]) if t.get("vmaf") else None
        print(f"{t['clip_id']} {t['config_id']:40} pass={t['passed']} vmaf={t['vmaf'] and round(t['vmaf'],2)} gap={g and round(g,2)} "
              f"use={t.get('util') and round(t['util'],3)} crf={t.get('aim')} cpu={t.get('enc_cpu_seconds') and round(t['enc_cpu_seconds'])} {t['codes']}", flush=True)
