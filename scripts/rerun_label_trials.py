#!/usr/bin/env python3
"""Re-run selected label trials with the current rate search (label policy 8) and fully relabel
selected clips, sharded. Output: <out>/shard<i>/{trials.jsonl, labels.jsonl, features.jsonl, done.txt}.
scripts/merge_label_fix.py folds the results into a merged label directory.

  python scripts/rerun_label_trials.py --labels data/encoder_rank_v7/merged/labels.jsonl \
      --trials data/encoder_rank_v7/merged/trials.jsonl --pairs pairs.json --full id1,id2 \
      --out data/encoder_rank_v7/fix --shard 0 --nshards 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.features.extract import FEATURE_NAMES  # noqa: E402,F401
from streamenc.ml.encoder_rank.catalog import PILOT_LABEL_ID  # noqa: E402
from streamenc.ml.encoder_rank.labeling import (  # noqa: E402
    ARM_WORKERS, PILOT_MAX_ATTEMPTS, label_clip, run_label_trial)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--pairs", default="", help="json list of [clip_id, config_id] to re-run")
    ap.add_argument("--full", default="", help="comma list of clip ids to relabel completely")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=str(ROOT / "work" / "label_fix"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    labels = {json.loads(l)["clip_id"]: json.loads(l) for l in open(args.labels) if l.strip()}
    old_trials: dict[tuple[str, str], dict] = {}
    for l in open(args.trials):
        if l.strip():
            t = json.loads(l)
            old_trials[(t["clip_id"], t["config_id"])] = t
    todo: dict[str, list[str]] = {}
    if args.pairs:
        for cid, lid in json.load(open(args.pairs)):
            todo.setdefault(cid, []).append(lid)
    full = [c for c in args.full.split(",") if c]
    for c in full:
        todo[c] = ["__full__"]
    clips = sorted(todo)
    mine = [c for i, c in enumerate(clips) if i % args.nshards == args.shard]
    out = Path(args.out) / f"shard{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    done_f = out / "done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    work = Path(args.work_dir) / f"shard{args.shard}"
    print(f"shard {args.shard}: {len(mine)} clips ({len(done)} done)", flush=True)
    for k, cid in enumerate(mine, 1):
        if cid in done:
            continue
        t0 = time.time()
        path = Path(labels[cid]["path"])
        if todo[cid] == ["__full__"]:
            label, trials, meta = label_clip(path, work_dir=work)
            with open(out / "labels.jsonl", "a") as f:
                f.write(json.dumps(label) + "\n")
            with open(out / "features.jsonl", "a") as f:
                f.write(json.dumps({"clip_id": cid, "path": str(path), "split": label["split"], "tier": label["tier"],
                                    "kind": label["kind"], "family": label["family"],
                                    "difficulty_score": label["difficulty_score"], "is_hard": label["is_hard"],
                                    "regime": label["regime"], **meta["features"]}) + "\n")
            what = f"full relabel winner={label['winner_config_id']} vmaf={label['winner_vmaf']}"
        else:
            lids = list(todo[cid])
            trials = []
            hint = (old_trials.get((cid, PILOT_LABEL_ID)) or {}).get("crf_estimate")
            if PILOT_LABEL_ID in lids:
                p = run_label_trial(path, PILOT_LABEL_ID, work, crf_hint=None, max_attempts=PILOT_MAX_ATTEMPTS)
                trials.append(p)
                hint = p.get("crf_estimate") if p.get("crf_estimate") is not None else hint
                lids.remove(PILOT_LABEL_ID)
            with ThreadPoolExecutor(ARM_WORKERS) as ex:
                trials += list(ex.map(lambda lid: run_label_trial(path, lid, work, crf_hint=hint), lids))
            what = " ".join(f"{t['config_id'][8:]}:{'ok' if t['passed'] else 'x'}:{t['vmaf'] and round(t['vmaf'], 2)}"
                            f"(was {'ok' if old_trials[(cid, t['config_id'])]['passed'] else 'x'}:"
                            f"{old_trials[(cid, t['config_id'])]['vmaf'] and round(old_trials[(cid, t['config_id'])]['vmaf'], 2)})"
                            for t in trials)
        with open(out / "trials.jsonl", "a") as f:
            for t in trials:
                f.write(json.dumps({**t, "rerun": True}) + "\n")
        with open(done_f, "a") as f:
            f.write(cid + "\n")
        print(f"[{k}/{len(mine)}] {cid} {time.time() - t0:.0f}s {what}", flush=True)
    print(f"shard {args.shard} finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
