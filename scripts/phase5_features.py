#!/usr/bin/env python3
"""Phase 5: extract content features; compare sampled vs full; bench overhead."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.benchmark.discover import discover_clips, select_benchmark_clips
from streamenc.features.analyze import agreement_row, analyze_video
from streamenc.features.extract import FEATURE_NAMES
from streamenc.util.logging import get_logger

log = get_logger("phase5")


def _save(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet failed: %s", exc)


def _discover_paths(
    dataset: Path,
    *,
    limit: int,
    corpus: str,
    train_per_kind: int,
    test_per_kind: int,
) -> list[Path]:
    if corpus == "all":
        paths: list[Path] = []
        for sub in ("ai", "render", "screen", "ai1080"):
            d = dataset / sub
            if d.is_dir():
                paths.extend(sorted(d.glob("*.mkv")))
    else:
        all_clips = discover_clips(dataset)
        train, test = select_benchmark_clips(
            all_clips,
            train_per_kind=train_per_kind,
            test_per_kind=test_per_kind,
        )
        # Prefer 720p for speed; still cover kinds. Deduplicate by path.
        seen: set[str] = set()
        paths = []
        for c in train + test:
            if c.path in seen:
                continue
            seen.add(c.path)
            paths.append(Path(c.path))
    if limit > 0:
        paths = paths[:limit]
    return paths

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "phase5"))
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="limit clips (0=all selected)")
    parser.add_argument(
        "--corpus",
        choices=("bench", "all"),
        default="bench",
        help="family-safe bench subset (default) or every mkv",
    )
    parser.add_argument("--train-per-kind", type=int, default=4)
    parser.add_argument("--test-per-kind", type=int, default=2)
    parser.add_argument(
        "--compare-limit",
        type=int,
        default=12,
        help="how many clips get full-frame comparison (expensive)",
    )
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    paths = _discover_paths(
        dataset,
        limit=args.limit,
        corpus=args.corpus,
        train_per_kind=args.train_per_kind,
        test_per_kind=args.test_per_kind,
    )
    if not paths:
        log.error("no clips found under %s", dataset)
        return 1

    log.info(
        "Phase5: %d clips, sampled max_samples=%d max_width=%d compare_limit=%d",
        len(paths),
        args.max_samples,
        args.max_width,
        args.compare_limit,
    )

    sampled_results = []
    for i, path in enumerate(paths, 1):
        log.info("[%d/%d] sampled %s", i, len(paths), path.name)
        res = analyze_video(
            path,
            mode="sampled",
            max_samples=args.max_samples,
            max_width=args.max_width,
        )
        sampled_results.append(res)
        log.info(
            "  frames=%d/%d t=%.3fs spatial=%.3f temporal=%.4f edge=%.4f motion=%.4f repeat=%.3f",
            res.features.n_frames_analyzed,
            res.frames_total,
            res.total_seconds,
            res.features.spatial_complexity,
            res.features.temporal_complexity,
            res.features.edge_density,
            res.features.motion_complexity,
            res.features.repeated_frame_ratio,
        )

    sampled_df = pd.DataFrame([r.to_row() for r in sampled_results])
    _save(sampled_df, results_dir / "features_sampled")

    agreement_rows = []
    full_rows = []
    sampled_by_id = {r.clip_id: r for r in sampled_results}
    if not args.skip_full:
        compare_paths = paths[: max(0, args.compare_limit)]
        for i, path in enumerate(compare_paths, 1):
            log.info("[%d/%d] full %s", i, len(compare_paths), path.name)
            full = analyze_video(
                path,
                mode="full",
                max_samples=args.max_samples,
                max_width=args.max_width,
            )
            full_rows.append(full.to_row())
            samp = sampled_by_id[path.stem]
            agr = agreement_row(samp, full)
            agreement_rows.append(agr)
            log.info(
                "  full_t=%.3fs sampled_t=%.3fs speedup=%.2fx corr=%.4f mae=%.4f",
                full.total_seconds,
                samp.total_seconds,
                agr["speedup"] or 0.0,
                agr["pearson_corr"],
                agr["mae"],
            )

        if full_rows:
            _save(pd.DataFrame(full_rows), results_dir / "features_full")
        if agreement_rows:
            agr_df = pd.DataFrame(agreement_rows)
            _save(agr_df, results_dir / "sampled_vs_full")

    # Overhead summary
    spf = sampled_df["total_seconds"] / sampled_df["frames_total"].clip(lower=1)
    summary = {
        "n_clips_sampled": int(len(sampled_df)),
        "n_clips_full_compare": int(len(agreement_rows)),
        "max_samples": args.max_samples,
        "max_width": args.max_width,
        "feature_names": list(FEATURE_NAMES),
        "sampled_runtime": {
            "mean_seconds": float(sampled_df["total_seconds"].mean()),
            "p50_seconds": float(sampled_df["total_seconds"].median()),
            "p95_seconds": float(sampled_df["total_seconds"].quantile(0.95)),
            "mean_seconds_per_source_frame": float(spf.mean()),
            "extrapolated_3428_frames_seconds": float(spf.mean() * 3428.0),
        },
    }
    if agreement_rows:
        agr_df = pd.DataFrame(agreement_rows)
        summary["sampled_vs_full"] = {
            "mean_pearson_corr": float(agr_df["pearson_corr"].mean()),
            "mean_mae": float(agr_df["mae"].mean()),
            "mean_rel_err": float(agr_df["mean_rel_err"].mean()),
            "mean_speedup": float(agr_df["speedup"].mean()),
            "min_pearson_corr": float(agr_df["pearson_corr"].min()),
            "worth_full_frame": bool(agr_df["pearson_corr"].mean() < 0.95),
            "recommendation": (
                "prefer_sampled"
                if agr_df["pearson_corr"].mean() >= 0.95 and agr_df["speedup"].mean() >= 2.0
                else "consider_full_or_more_samples"
            ),
        }
        # Per-feature mean abs error
        err_cols = [c for c in agr_df.columns if c.startswith("err_")]
        summary["sampled_vs_full"]["mean_abs_err_by_feature"] = {
            c.replace("err_", ""): float(agr_df[c].mean()) for c in err_cols
        }

    # Quick kind-level feature means (useful for Phase 6 rules)
    if "clip_id" in sampled_df.columns:
        kind_map = {}
        for p in paths:
            parent = p.parent.name
            kind = "ai" if parent in {"ai", "ai1080"} else parent
            kind_map[p.stem] = kind
        sampled_df = sampled_df.copy()
        sampled_df["kind"] = sampled_df["clip_id"].map(kind_map)
        kind_means = (
            sampled_df.groupby("kind")[list(FEATURE_NAMES)]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        summary["feature_means_by_kind"] = kind_means
        _save(sampled_df, results_dir / "features_sampled")  # rewrite with kind

    (results_dir / "phase5_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase5 summary:\n%s", json.dumps(summary, indent=2))
    log.info("Phase5 complete → %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
