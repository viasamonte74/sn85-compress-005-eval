#!/usr/bin/env python3
"""q07 / q08 round 3 (2026-11-11): 10-bit, capped-CRF exact budget fill, libaom slow presets, hierarchical levels,
combined with the chroma blur (cblur4) / 85 % low-pass that helped in rounds 1-2. Each = label trial (rate fit +
full validation incl. NEG gap and chroma / tone gates). Results: results/user_tests/q0708_round3.jsonl."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS
from streamenc.ml.encoder_rank.labeling import run_label_trial
P2, P1 = ARMS["svt_crf_p2_qm_la"], ARMS["svt_crf_p1_qm_la"]
def v(base, **kw):
    a = {**base, "svt": {**base["svt"]}}
    for k, val in kw.items():
        if k in ("pix_fmt", "cap"): a[k] = val
        else: a["svt"][k.replace("_", "-")] = val
    return a
ARMS.update({
    "q_p2_b10": v(P2, pix_fmt="yuv420p10le"), "q_p1_b10": v(P1, pix_fmt="yuv420p10le"),
    "q_p2_cap": v(P2, cap=1.0), "q_p2_cap_b10": v(P2, cap=1.0, pix_fmt="yuv420p10le"),
    "q_p2_hl3": v(P2, hierarchical_levels=3), "q_p2_hl4": v(P2, hierarchical_levels=4),
    "aom_c1_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 1, "tune": "psnr"},
})
# (config, crf hint): capped CRF starts 2 CRF below the uncapped fit so the cap (1.0 x budget) binds
CFG = [("q_p2_b10", None), ("q_p2_b10+cblur4", None), ("q_p1_b10+cblur4_ds85", None), ("q_p1_b10+cblur4", None),
       ("q_p2_cap", -2.0), ("q_p2_cap+cblur4", -2.0), ("q_p2_cap_b10+cblur4", -2.0),
       ("q_p2_hl3", None), ("q_p2_hl4", None), ("q_p2_hl3+cblur4", None),
       ("aom_c2_psnr", None), ("aom_c1_psnr", None), ("aom_c2_psnr+cblur4", None), ("aom_c2_psnr+ds85", None)]
OUT = ROOT / "results/user_tests/q0708_round3.jsonl"
done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
for clip, crf in (("pp005_q08", 55.5), ("pp005_q07", 57.25)):
    path = Path(f"/workspace/dataset/practice005/{clip}.mkv")
    todo = [(c, h) for c, h in CFG if (clip, c) not in done]
    def one(ch):
        c, h = ch
        return run_label_trial(path, c, ROOT / "work/q0708r3", crf_hint=(crf + h) if h is not None else crf)
    with ThreadPoolExecutor(6) as ex:
        for t in ex.map(one, todo):
            with open(OUT, "a") as f: f.write(json.dumps(t) + "\n")
            g = (t["vmaf"] - t["vmaf_neg"]) if t.get("vmaf") else None
            print(f"{clip} {t['config_id']:28} pass={t['passed']} vmaf={t['vmaf'] and round(t['vmaf'],2)} gap={g and round(g,2)} "
                  f"use={t.get('util') and round(t['util'],3)} crf={t.get('aim')} cpu={t.get('enc_cpu_seconds') and round(t['enc_cpu_seconds'])} {t['codes']}", flush=True)
