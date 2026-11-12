#!/usr/bin/env python3
"""Fold re-run label trials (scripts/rerun_label_trials.py) into a merged label directory.

  python scripts/merge_label_fix.py --base data/encoder_rank_v7/merged --fix data/encoder_rank_v7/fix \
      --out data/encoder_rank_v7/merged_fix

Re-run (clip, config) trials replace the old rows; fully relabelled clips replace all their rows and
their label. Labels of clips with re-run trials are recomputed with labeling.select_winner (same rule
as label_clip). ``kind`` is re-derived for every clip with the current splits.clip_kind.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.ml.encoder_rank.catalog import LABEL_POLICY_VERSION, write_catalog  # noqa: E402
from streamenc.ml.encoder_rank.labeling import select_winner  # noqa: E402
from streamenc.ml.encoder_rank.splits import clip_kind  # noqa: E402


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, fix, out = Path(args.base), Path(args.fix), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = {l["clip_id"]: l for l in jl(base / "labels.jsonl")}
    trials = jl(base / "trials.jsonl")
    new_trials, new_labels, new_feats = [], {}, []
    for s in sorted(fix.glob("shard*")):
        new_trials += jl(s / "trials.jsonl")
        new_labels.update({l["clip_id"]: l for l in jl(s / "labels.jsonl")})
        new_feats += jl(s / "features.jsonl")
    full = set(new_labels)
    rerun_pairs = {(t["clip_id"], t["config_id"]) for t in new_trials if t["clip_id"] not in full}
    kept = [t for t in trials if t["clip_id"] not in full and (t["clip_id"], t["config_id"]) not in rerun_pairs]
    merged_trials = kept + new_trials
    by_clip: dict[str, list[dict]] = {}
    for t in merged_trials:
        by_clip.setdefault(t["clip_id"], []).append(t)

    changed = 0
    for cid in {c for c, _ in rerun_pairs}:
        lab = labels[cid]
        tr = by_clip[cid]
        winner, q = select_winner(tr)
        valid = [t for t in tr if t.get("passed") and t.get("passed_streaming")]
        before = (lab.get("winner_config_id"), lab.get("winner_vmaf"))
        lab.update({
            "winner_config_id": winner["config_id"] if winner else None,
            "winner_config_index": winner["config_index"] if winner else None,
            "winner_prefilter": winner["prefilter"] if winner else None,
            "winner_aim": winner.get("aim") if winner else None,
            "winner_vmaf": winner.get("vmaf") if winner else None,
            "winner_clip_score": float(winner.get("clip_score") or 0.0) if winner else 0.0,
            "winner_util": winner.get("util") if winner else None,
            "winner_passed": bool(winner.get("passed")) if winner else False,
            "winner_passed_streaming": bool(winner.get("passed_streaming")) if winner else False,
            "n_trials": len(tr),
            "n_streaming_pass": sum(1 for t in tr if t.get("passed_streaming")),
            "max_vmaf": max((t["vmaf"] for t in valid if t.get("vmaf") is not None), default=None),
            "max_clip_score": max((float(t.get("clip_score") or 0.0) for t in valid), default=0.0),
            "trainable": bool(q["trainable"]),
            "winner_confidence": float(q["winner_confidence"]),
            "n_gate_valid": int(q["n_valid"]),
            "reject_reason": q.get("reject_reason"),
            "tie_with": q.get("tie_with"),
            "efficient_config_id": q.get("efficient_config_id"),
            "hopeless": bool(q.get("hopeless", False)),
            "label_policy_version": LABEL_POLICY_VERSION,
            "fix_rerun": True,
        })
        changed += before != (lab["winner_config_id"], lab["winner_vmaf"])
    labels.update(new_labels)
    for lab in labels.values():
        lab["kind"] = clip_kind(Path(lab["path"]))

    with open(out / "labels.jsonl", "w") as f:
        for lab in labels.values():
            f.write(json.dumps(lab) + "\n")
    with open(out / "trials.jsonl", "w") as f:
        for t in merged_trials:
            f.write(json.dumps(t) + "\n")
    with open(out / "labels_trainable.jsonl", "w") as f:
        for lab in labels.values():
            if lab.get("trainable"):
                f.write(json.dumps(lab) + "\n")
    rows = list(csv.DictReader(open(base / "features.csv")))
    cols = list(rows[0].keys())
    have = {r["clip_id"] for r in rows if r["clip_id"] not in full}
    rows = [r for r in rows if r["clip_id"] in have]
    for r in new_feats:
        rows.append({k: r.get(k, "") for k in cols})
    for r in rows:
        if r["clip_id"] in labels:
            r["kind"] = labels[r["clip_id"]]["kind"]
    with open(out / "features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    splits = json.loads((base / "splits.json").read_text())
    for lab in new_labels.values():
        splits[lab["family"]] = lab["split"]
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    write_catalog(out / "catalog.json")
    missing = [c for c in labels if c not in {r["clip_id"] for r in rows}]
    print(f"labels {len(labels)} (full relabel {len(full)}, recomputed {len({c for c, _ in rerun_pairs})}, "
          f"winner changed {changed}); trials {len(merged_trials)} (re-run {len(new_trials)}); "
          f"features {len(rows)}; clips without features {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
