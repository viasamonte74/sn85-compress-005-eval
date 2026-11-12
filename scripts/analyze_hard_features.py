#!/usr/bin/env python3
"""Extract improved features on labeled + generated hard clips; evaluate detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.features.analyze import analyze_video
from streamenc.features.difficulty import difficulty_score
from streamenc.features.extract import FEATURE_NAMES
from streamenc.util.logging import get_logger

log = get_logger("analyze_hard")


def _tier_from_name(stem: str) -> str:
    if "_hard_" in stem or stem.startswith("gen_hard_"):
        return "hard"
    if "_medium_" in stem:
        return "medium"
    if "_easy_" in stem:
        return "easy"
    return "other"


def collect_paths(dataset: Path, generated: Path, *, per_tier: int) -> list[Path]:
    paths: list[Path] = []
    for tier, needle in (("easy", "*easy*.mkv"), ("medium", "*medium*.mkv"), ("hard", "*hard*.mkv")):
        bucket: list[Path] = []
        for sub in ("ai", "render", "screen"):
            bucket.extend(sorted((dataset / sub).glob(needle)))
        bucket = [p for p in bucket if "_1080" not in p.stem]
        # Spread selection
        if len(bucket) > per_tier:
            step = max(1, len(bucket) // per_tier)
            bucket = bucket[::step][:per_tier]
        paths.extend(bucket)
    if generated.is_dir():
        paths.extend(sorted(generated.glob("gen_hard_*.mkv")))
    # dedupe
    seen = set()
    out = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--generated", default="/workspace/dataset/generated_hard")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "phase5b"))
    parser.add_argument("--per-tier", type=int, default=12)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_paths(Path(args.dataset), Path(args.generated), per_tier=args.per_tier)
    log.info("Analyzing %d clips", len(paths))

    official = {}
    off_path = Path(args.dataset) / "official_scores.jsonl"
    if off_path.exists():
        for line in off_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                official[row["clip"]] = row

    rows = []
    for i, path in enumerate(paths, 1):
        log.info("[%d/%d] %s", i, len(paths), path.name)
        res = analyze_video(path, mode="sampled", max_samples=24, max_width=640)
        diff = difficulty_score(res.features)
        tier = _tier_from_name(path.stem)
        off = official.get(path.stem.replace("_1080", ""), {})
        row = {
            "clip_id": path.stem,
            "path": str(path),
            "tier_label": tier,
            "generated": path.stem.startswith("gen_hard_"),
            "difficulty_score": diff.score,
            "is_hard_pred": diff.is_hard,
            "hard_reasons": ",".join(diff.reasons),
            "official_score": off.get("score"),
            "official_vmaf": off.get("vmaf"),
            "feature_seconds": res.total_seconds,
            **{k: getattr(res.features, k) for k in FEATURE_NAMES},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "features_expanded.csv", index=False)
    try:
        df.to_parquet(results_dir / "features_expanded.parquet", index=False)
    except Exception:  # noqa: BLE001
        pass

    # Detector metrics vs tier labels (easy vs hard; medium counted as not-hard target false)
    labeled = df[df["tier_label"].isin(["easy", "medium", "hard"])]
    y_true = labeled["tier_label"] == "hard"
    y_pred = labeled["is_hard_pred"]
    tp = int(((y_true) & (y_pred)).sum())
    tn = int(((~y_true) & (~y_pred)).sum())
    fp = int(((~y_true) & (y_pred)).sum())
    fn = int(((y_true) & (~y_pred)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    # Separation: mean difficulty by tier
    tier_means = (
        labeled.groupby("tier_label")["difficulty_score"].agg(["mean", "median", "count"]).reset_index()
    )
    gen = df[df["generated"]]
    summary = {
        "n_clips": len(df),
        "n_generated": int(gen.shape[0]),
        "detector": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "difficulty_by_tier": tier_means.to_dict(orient="records"),
        "generated_difficulty": {
            "mean": float(gen["difficulty_score"].mean()) if len(gen) else None,
            "median": float(gen["difficulty_score"].median()) if len(gen) else None,
            "hard_pred_rate": float(gen["is_hard_pred"].mean()) if len(gen) else None,
        },
        "feature_names": list(FEATURE_NAMES),
        # Correlation with official score when available (lower official → higher difficulty)
        "corr_difficulty_vs_official_score": (
            float(labeled.dropna(subset=["official_score"])[["difficulty_score", "official_score"]].corr().iloc[0, 1])
            if labeled["official_score"].notna().sum() >= 5
            else None
        ),
    }
    (results_dir / "phase5b_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase5b summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
