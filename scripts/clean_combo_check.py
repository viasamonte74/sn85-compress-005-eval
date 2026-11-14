#!/usr/bin/env python3
"""Fast-preset check of the clean-content search winners (2026-11-13). The runtime cannot afford preset 1 on most
clips, so: do the cheap single tools (chroma QP, start-up QP offset, screen-content mode, AC bias, chroma blur,
10-bit) add up at the presets the runtime actually ships (p2 / p4 / p6)? Same label trials as clean_config_search.py,
appended to results/clean_search/trials.jsonl (resumable).
  python scripts/clean_combo_check.py --clips rnd_hard_06,... --workers 8
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
sys.path.insert(0, str(ROOT / "scripts"))
import clean_config_search as ccs  # noqa: E402  (registers chq32 / chq63 in hcs.TOOLS)
import hard_config_search as hcs  # noqa: E402
from streamenc.ml.encoder_rank.labeling import run_label_trial, source_cache  # noqa: E402

hcs.TOOLS.update({"p6": ("preset", {"preset": 6})})
COMBOS = [(), ("chq63",), ("sqo8",), ("scm1",), ("chq63", "sqo8"), ("chq63", "sqo8", "scm1"),
          ("chq63", "sqo8", "scm1", "acb1"), ("f_cblur", "chq63", "sqo8"), ("b10", "chq63", "sqo8")]
PRESETS = [("p6",), ()]   # () = the p4 base; p2 dropped after the first 6 clips (slow, least used at run time)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    labels = {json.loads(l)["clip_id"]: json.loads(l) for l in open(ROOT / "data/encoder_rank_all/labels.jsonl")}
    hint = {}
    for l in open(ROOT / "data/encoder_rank_all/trials.jsonl"):
        t = json.loads(l)
        if t["config_id"] == "svt_crf_p4_qm_la" and t.get("crf_estimate") is not None:
            hint[t["clip_id"]] = float(t["crf_estimate"])
    done = set()
    if ccs.OUT.exists():
        done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(ccs.OUT)}
    work = ROOT / "work/clean_combo"
    for cid in a.clips.split(","):
        assert labels[cid]["split"] != "test"
        path = Path(labels[cid]["path"])
        raw, raw_dir = source_cache(path)
        lids = [hcs.label_id(p + c) for p in PRESETS for c in COMBOS]
        todo = [l for l in lids if (cid, l) not in done]
        print(f"== {cid}: {len(todo)} of {len(lids)} trials to run", flush=True)
        try:
            with ThreadPoolExecutor(a.workers) as ex:
                for t in ex.map(lambda l: run_label_trial(path, l, work, crf_hint=hint.get(cid), encode_input=raw), todo):
                    with open(ccs.OUT, "a") as f:
                        f.write(json.dumps(t) + "\n")
                    print(f"  {t['config_id']:34} pass={t['passed']} vmaf={t['vmaf']} util={t.get('util') and round(t['util'], 4)} "
                          f"norm={ccs.metric(t) and round(ccs.metric(t), 3)} cpu={t.get('enc_cpu_seconds')}", flush=True)
        finally:
            if raw_dir:
                shutil.rmtree(raw_dir, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
