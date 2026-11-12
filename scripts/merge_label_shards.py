#!/usr/bin/env python3
"""Merge data/encoder_rank_<ver>/shard*/ into data/encoder_rank_<ver>/merged/ (labels.jsonl, trials.jsonl,
features.csv, splits.json, catalog.json) so train_encoder_rank.py --data can point at one directory."""
import glob, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ver = sys.argv[1] if len(sys.argv) > 1 else "v6"
base = ROOT / "data" / f"encoder_rank_{ver}"; out = base / "merged"; out.mkdir(exist_ok=True)
shards = sorted(p for p in base.glob("shard*") if p.is_dir())
for name in ("labels.jsonl", "trials.jsonl"):
    rows = []
    for s in shards:
        f = s / name
        if f.exists(): rows += [l for l in f.read_text().splitlines() if l.strip()]
    (out / name).write_text("\n".join(rows) + ("\n" if rows else ""))
    print(name, len(rows), "rows")
feat = []
for s in shards:
    f = s / "features.csv"
    if f.exists():
        lines = f.read_text().splitlines()
        if not feat: feat.append(lines[0])
        feat += lines[1:]
(out / "features.csv").write_text("\n".join(feat) + "\n"); print("features.csv", len(feat) - 1, "rows")
splits = {}
for s in shards:
    f = s / "splits.json"
    if f.exists(): splits.update(json.loads(f.read_text()))
(out / "splits.json").write_text(json.dumps(splits, indent=2)); print("splits", len(splits))
shutil.copy(shards[0] / "catalog.json", out / "catalog.json")
