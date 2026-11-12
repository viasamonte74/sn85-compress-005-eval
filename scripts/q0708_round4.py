#!/usr/bin/env python3
"""q07 / q08 round 4 (2026-11-11): per-layer quantiser allocation. q08 per-frame VMAF: top-layer frames (68 % of
frames) avg 54.6-59.8 vs reference frames 66-69 -> flatten the layer QP spread so the mean rises at equal bits.
SVT-AV1 4.3: use-fixed-qindex-offsets=1 + qindex-offsets=[L0..L5] (qindex units, 0..255), key-frame-qindex-offset,
qp-scale-compress-strength 1..3. All p2 (hierarchical-levels 5 default). -> results/user_tests/q0708_round4.jsonl"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS
from streamenc.ml.encoder_rank.labeling import run_label_trial
P2 = ARMS["svt_crf_p2_qm_la"]
def v(**kw):
    return {**P2, "svt": {**P2["svt"], **{k.replace("_", "-"): x for k, x in kw.items()}}}
F = dict(use_fixed_qindex_offsets=1)
ARMS.update({
    "q_p2_fq_flat": v(**F, qindex_offsets="[0,0,0,0,0,0]"),
    "q_p2_fq_1": v(**F, qindex_offsets="[0,1,2,3,4,5]"),
    "q_p2_fq_2": v(**F, qindex_offsets="[0,2,4,6,8,10]"),
    "q_p2_fq_3": v(**F, qindex_offsets="[0,3,6,9,12,15]"),
    "q_p2_fq_5": v(**F, qindex_offsets="[0,5,10,15,20,25]"),
    "q_p2_fq_inv": v(**F, qindex_offsets="[4,3,2,1,0,0]"),
    "q_p2_fq_1_kf8": v(**F, qindex_offsets="[0,1,2,3,4,5]", key_frame_qindex_offset=8),
    "q_p2_fq_1_kf16": v(**F, qindex_offsets="[0,1,2,3,4,5]", key_frame_qindex_offset=16),
    "q_p2_qpc1": v(qp_scale_compress_strength=1), "q_p2_qpc2": v(qp_scale_compress_strength=2),
    "q_p2_qpc3": v(qp_scale_compress_strength=3),
})
CFG = list(k for k in ARMS if k.startswith("q_p2_fq") or k.startswith("q_p2_qpc"))
OUT = ROOT / "results/user_tests/q0708_round4.jsonl"
done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
for clip, crf in (("pp005_q08", 55.5), ("pp005_q07", 57.25)):
    path = Path(f"/workspace/dataset/practice005/{clip}.mkv")
    todo = [c for c in CFG if (clip, c) not in done]
    with ThreadPoolExecutor(6) as ex:
        for t in ex.map(lambda c: run_label_trial(path, c, ROOT / "work/q0708r4", crf_hint=crf), todo):
            with open(OUT, "a") as f: f.write(json.dumps(t) + "\n")
            g = (t["vmaf"] - t["vmaf_neg"]) if t.get("vmaf") else None
            print(f"{clip} {t['config_id']:18} pass={t['passed']} vmaf={t['vmaf'] and round(t['vmaf'],2)} gap={g and round(g,2)} "
                  f"use={t.get('util') and round(t['util'],3)} crf={t.get('aim')} cpu={t.get('enc_cpu_seconds') and round(t['enc_cpu_seconds'])} {t['codes']}", flush=True)
