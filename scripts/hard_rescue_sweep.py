#!/usr/bin/env python3
"""Hard-clip rescue sweep (2026-11-10): which encoder tools let clips that no v8 catalog arm could score
(no gate-valid trial = over budget even at CRF 63, or best VMAF < 60 -> clip score 0) reach a score.

Each (clip, config) is a normal label trial (labeling.run_label_trial: rate fit at 0.05 bpp + full competition
validation incl. VMAF-NEG gate), so results compare directly with the existing labels. Resumable: appends to
results/hard_rescue/trials.jsonl and skips (clip, config) pairs already there.
  nice -n 19 /venv/main/bin/python scripts/hard_rescue_sweep.py [--clips a,b] [--configs x,y] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.ml.encoder_rank.labeling import run_label_trial, source_cache  # noqa: E402

SAMPLE = [
    # no gate-valid trial (over budget at CRF 63)
    "rnd_hard_03", "rnd_hard_04_1080", "gen_hard_19_edge_boost_ai_hard_19", "gen_hard_21_edge_boost_ai_hard_25",
    # gate-valid but best VMAF < 60 (score 0), spread over kinds / resolutions / how far below 60
    "ai_hard_20_1080", "gen_hard_09_grain_ai_hard_20", "ai_medium_02_1080", "anim_hard_011_1080",
    "aix_hard_012", "scr_hard_09", "ai_hard_25", "rnd_hard_02",
]
CONFIGS = [
    "svt_crf_p4_qm_la",                                   # baseline (v8 catalog arm)
    "svt_crf_p4_qm_sr12", "svt_crf_p4_qm_sr16",           # AV1 superres 1.33x / 2x horizontal
    "svt_crf_p4_qm_la_vb", "svt_crf_p4_qm_la_tf3", "svt_crf_p4_qm_la_fg", "svt_crf_p4_qm_la_qpc",
    "svt_crf_p4_qm_la+hqdn3d_hard", "svt_crf_p4_qm_la+hard_denoise",
    "svt_crf_p4_qm_la+ds85", "svt_crf_p4_qm_la+ds67", "svt_crf_p4_qm_la+ds50",
    "nv_av1_p7_uhq",                                      # NVENC AV1 reference (VBR)
]
OUT = ROOT / "results/hard_rescue/trials.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    clips = [c for c in (a.clips.split(",") if a.clips else SAMPLE) if c]
    configs = [c for c in (a.configs.split(",") if a.configs else CONFIGS) if c]
    labels = {}
    extra = sorted((ROOT / "data").glob("encoder_rank_v14*/shard*"))  # label sets not merged into _all yet
    for f in [ROOT / "data/encoder_rank_all/labels.jsonl"] + [d / "labels.jsonl" for d in extra]:
        for l in (open(f) if f.exists() else []):
            r = json.loads(l); labels[r["clip_id"]] = r
    pilot = {}
    for f in [ROOT / "data/encoder_rank_all/trials.jsonl"] + [d / "trials.jsonl" for d in extra]:
        for l in (open(f) if f.exists() else []):
            t = json.loads(l)
            if t["config_id"] == "svt_crf_p5_qm_la" and t.get("crf_estimate") is not None:
                pilot[t["clip_id"]] = float(t["crf_estimate"])
    done = set()
    if OUT.exists():
        for l in open(OUT):
            t = json.loads(l); done.add((t["clip_id"], t["config_id"]))
    work = ROOT / "work/hard_rescue"
    for cid in clips:
        path = Path(labels[cid]["path"])
        todo = [c for c in configs if (cid, c) not in done]
        if not todo:
            continue
        raw, raw_dir = source_cache(path)
        hint = pilot.get(cid)
        print(f"{cid}: {len(todo)} configs, pilot CRF hint {hint}", flush=True)
        try:
            with ThreadPoolExecutor(a.workers) as ex:
                for t in ex.map(lambda c: run_label_trial(path, c, work, crf_hint=hint if c.startswith("svt") else None,
                                                          encode_input=raw), todo):
                    with open(OUT, "a") as f:
                        f.write(json.dumps(t) + "\n")
                    print(f"  {t['config_id']:34} ok={t['ok']} pass={t['passed']} vmaf={t['vmaf']} neg={t.get('vmaf_neg')} "
                          f"score={t['clip_score']} util={t['util']} enc={t['encode_seconds']} codes={t['codes'][:60]}", flush=True)
        finally:
            if raw_dir:
                shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
