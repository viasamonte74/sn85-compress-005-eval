#!/usr/bin/env python3
"""Preset headroom (2026-11-13, user: "review current version and find gap of the performance"): for evaluation clips
that score below the cap, how much VMAF do slower presets buy at the same budget? Every trial is a normal label trial
(rate fit at 0.05 bpp + full validator-equivalent validation) with rc10's chroma QP, base svt p4 qm la120 tune 1.
  python scripts/preset_headroom.py --list data/headroom_clips.txt --workers 8
Output: results/headroom/trials.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import clean_config_search  # noqa: E402,F401  (registers chq63)
import hard_config_search as hcs  # noqa: E402
from streamenc.ml.encoder_rank.labeling import run_label_trial, source_cache  # noqa: E402

PRESETS = [("chq63",), ("p3", "chq63"), ("p2", "chq63"), ("p1", "chq63")]
OUT = ROOT / "results/headroom/trials.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["clip_id"], json.loads(l)["config_id"]) for l in open(OUT)} if OUT.exists() else set()
    work = ROOT / "work/headroom"
    for path in [Path(l.strip()) for l in open(a.list) if l.strip() and not l.startswith("#")]:
        cid = path.parent.name if path.name == "clip.mkv" else path.stem
        lids = [hcs.label_id(t) for t in PRESETS]
        todo = [l for l in lids if (cid, l) not in done]
        if not todo:
            continue
        raw, raw_dir = source_cache(path)
        print(f"== {cid}: {len(todo)} trials", flush=True)
        try:
            with ThreadPoolExecutor(a.workers) as ex:
                for t in ex.map(lambda l: run_label_trial(path, l, work, encode_input=raw), todo):
                    t["clip_id"] = cid
                    with open(OUT, "a") as f:
                        f.write(json.dumps(t) + "\n")
                    print(f"  {t['config_id']:16} pass={t['passed']} vmaf={t['vmaf']} util={t.get('util') and round(t['util'], 4)} "
                          f"cpu={t.get('enc_cpu_seconds')} {t['codes'][:40]}", flush=True)
        finally:
            if raw_dir:
                shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
