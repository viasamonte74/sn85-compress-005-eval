#!/usr/bin/env python3
"""Concatenate merged label dirs (labels.jsonl, trials.jsonl, features.csv, splits.json) into one dir.
Later dirs win on duplicate clip ids.  Usage: combine_label_dirs.py --out DIR DIR1 DIR2 ..."""
import argparse, csv, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
labels, trials, feats, splits, header = {}, {}, {}, {}, None
for d in map(Path, a.dirs):
    lab = [json.loads(l) for l in (d / "labels.jsonl").read_text().splitlines() if l.strip()]
    ids = {r["clip_id"] for r in lab}
    for r in lab: labels[r["clip_id"]] = r
    for k in list(trials):
        if k[0] in ids: del trials[k]
    for l in (d / "trials.jsonl").read_text().splitlines():
        if l.strip():
            t = json.loads(l); trials[(t["clip_id"], t["config_id"], t.get("stage"), t.get("prefilter"))] = t
    with open(d / "features.csv") as f:
        rd = csv.DictReader(f)
        if header is None: header = rd.fieldnames
        elif rd.fieldnames != header: raise SystemExit(f"feature columns differ in {d}")
        for r in rd: feats[r["clip_id"]] = r
    if (d / "splits.json").exists(): splits.update(json.loads((d / "splits.json").read_text()))
(out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels.values()))
(out / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials.values()))
with open(out / "features.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header); w.writeheader(); [w.writerow(r) for r in feats.values()]
(out / "splits.json").write_text(json.dumps(splits, indent=2))
if (Path(a.dirs[-1]) / "catalog.json").exists(): shutil.copy(Path(a.dirs[-1]) / "catalog.json", out / "catalog.json")
print(f"labels {len(labels)} trials {len(trials)} features {len(feats)} splits {len(splits)} "
      f"missing features {sorted(set(labels) - set(feats))[:5]}")
