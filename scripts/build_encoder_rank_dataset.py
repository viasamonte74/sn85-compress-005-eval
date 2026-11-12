#!/usr/bin/env python3
"""Build encoder-rank dataset: videos → best encoder/settings (Phase-1 labeling).

Writes under data/encoder_rank/:
  catalog.json       frozen action space
  labels.jsonl       one winner row per clip
  trials.csv         all (clip, config) scores
  features.csv       Phase-5b features per clip
  splits.json        family → split map
  manifest.json      run metadata
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.ml.encoder_rank.catalog import write_catalog
from streamenc.ml.encoder_rank.labeling import collect_clip_paths, label_clip
from streamenc.util.logging import get_logger

log = get_logger("build_encoder_rank")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "encoder_rank_v3"))
    parser.add_argument("--work-dir", default=str(ROOT / "work" / "encoder_rank_v3"))
    parser.add_argument("--per-tier", type=int, default=8, help="Max clips per tier (easy/medium/hard); 0=all")
    parser.add_argument("--limit", type=int, default=0, help="Hard cap after stratification")
    parser.add_argument("--no-generated", action="store_true")
    parser.add_argument("--with-1080", action="store_true", help="Also label ai1080/ clips")
    parser.add_argument("--extra-dirs", default="", help="Comma list of extra dataset subdirs (ai_gen2,open_clips)")
    parser.add_argument("--include", default="", help="Regex on clip stem; labeled first, ignores --per-tier")
    parser.add_argument("--clip-list", default="", help="File with one clip path per line: label exactly these, in order")
    parser.add_argument("--resume", action="store_true", help="Skip clip_ids already in labels.jsonl")
    parser.add_argument(
        "--trainable-only-export",
        action="store_true",
        help="Also write labels_trainable.jsonl with gate-valid winners only",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = write_catalog(out_dir / "catalog.json")
    log.info("Wrote catalog %s", catalog_path)

    per_tier = None if args.per_tier <= 0 else args.per_tier
    extra_dirs = tuple(d for d in args.extra_dirs.split(",") if d)
    paths = collect_clip_paths(
        Path(args.dataset),
        include_generated=not args.no_generated,
        per_tier=per_tier,
        exclude_1080=not args.with_1080,
        extra_dirs=extra_dirs,
    )
    if args.clip_list:
        paths = [Path(l.strip()) for l in Path(args.clip_list).read_text().splitlines() if l.strip()]
        paths = [p for p in paths if p.exists()]
    if args.include:
        pat = re.compile(args.include)
        everything = collect_clip_paths(
            Path(args.dataset),
            include_generated=not args.no_generated,
            exclude_1080=not args.with_1080,
            extra_dirs=extra_dirs,
        )
        priority = [p for p in everything if pat.search(p.stem)]
        paths = priority + [p for p in paths if p not in priority]
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    log.info("Labeling %d clips", len(paths))

    labels_path = out_dir / "labels.jsonl"
    done: set[str] = set()
    if args.resume and labels_path.is_file():
        for line in labels_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip_id"])
        log.info("Resume: %d already labeled", len(done))

    label_rows: list[dict] = []
    trial_rows: list[dict] = []
    feature_rows: list[dict] = []
    splits: dict[str, str] = {}

    # Reload prior if resume
    if args.resume and labels_path.is_file():
        for line in labels_path.read_text().splitlines():
            if line.strip():
                label_rows.append(json.loads(line))
        trials_csv = out_dir / "trials.csv"
        feats_csv = out_dir / "features.csv"
        if trials_csv.is_file():
            trial_rows.extend(pd.read_csv(trials_csv).to_dict(orient="records"))
        if feats_csv.is_file():
            feature_rows.extend(pd.read_csv(feats_csv).to_dict(orient="records"))

    mode = "a" if args.resume else "w"
    with labels_path.open(mode) as lf:
        for i, path in enumerate(paths, 1):
            if path.stem in done:
                log.info("[%d/%d] skip %s", i, len(paths), path.name)
                continue
            log.info("[%d/%d] label %s", i, len(paths), path.name)
            try:
                label, trials, meta = label_clip(path, work_dir=work_dir)
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed %s: %s", path.name, exc)
                continue
            label_rows.append(label)
            trial_rows.extend(trials)
            with (out_dir / "trials.jsonl").open("a") as tjf:
                for t in trials:
                    tjf.write(json.dumps(t) + "\n")
            feature_rows.append(
                {
                    "clip_id": path.stem,
                    "path": str(path),
                    "split": label["split"],
                    "tier": label["tier"],
                    "kind": label["kind"],
                    "family": label["family"],
                    "difficulty_score": label["difficulty_score"],
                    "is_hard": label["is_hard"],
                    "regime": label["regime"],
                    **meta["features"],
                }
            )
            splits[label["family"]] = label["split"]
            lf.write(json.dumps(label) + "\n")
            lf.flush()
            log.info(
                "  winner=%s score=%.4f vmaf=%s valid=%s trainable=%s conf=%.2f split=%s (%.1fs)%s",
                label["winner_config_id"],
                float(label["winner_clip_score"] or 0),
                label["winner_vmaf"],
                label.get("n_gate_valid"),
                label.get("trainable"),
                float(label.get("winner_confidence") or 0),
                label["split"],
                float(label["label_seconds"] or 0),
                f" reject={label.get('reject_reason')}" if label.get("reject_reason") else "",
            )

    pd.DataFrame(trial_rows).to_csv(out_dir / "trials.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out_dir / "features.csv", index=False)
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    # Summary stats
    winners = {}
    trainable_rows = [r for r in label_rows if r.get("trainable")]
    for r in trainable_rows:
        cid = r.get("winner_config_id") or "none"
        winners[cid] = winners.get(cid, 0) + 1
    split_counts = {}
    for r in label_rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    tier_counts = {}
    for r in label_rows:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

    if args.trainable_only_export or True:
        tpath = out_dir / "labels_trainable.jsonl"
        with tpath.open("w") as tf:
            for r in trainable_rows:
                tf.write(json.dumps(r) + "\n")

    manifest = {
        "label_policy_version": 3,
        "n_clips": len(label_rows),
        "n_trainable": len(trainable_rows),
        "n_rejected": len(label_rows) - len(trainable_rows),
        "n_trials": len(trial_rows),
        "winner_counts_trainable": winners,
        "split_counts": split_counts,
        "tier_counts": tier_counts,
        "mean_winner_score_trainable": float(
            sum(float(r.get("winner_clip_score") or 0) for r in trainable_rows) / max(len(trainable_rows), 1)
        ),
        "fraction_score_gt0_trainable": float(
            sum(1 for r in trainable_rows if float(r.get("winner_clip_score") or 0) > 0)
            / max(len(trainable_rows), 1)
        ),
        "policy": {
            "full_catalog_sweep": True,
            "requires_gate_pass": True,
            "encoder": "encode_rate_fit (all encoder families, closed loop to ~99% budget)",
            "winner": "max(clip_score, VMAF, util) among gate-valid",
        },
        "catalog": str(catalog_path),
        "labels": str(labels_path),
        "labels_trainable": str(out_dir / "labels_trainable.jsonl"),
        "trials": str(out_dir / "trials.csv"),
        "features": str(out_dir / "features.csv"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %s", json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
