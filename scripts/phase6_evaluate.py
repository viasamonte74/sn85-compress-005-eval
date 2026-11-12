#!/usr/bin/env python3
"""Phase 6: evaluate content-aware decisions vs fixed NVENC p6 baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.benchmark.discover import ClipInfo, discover_clips, select_benchmark_clips
from streamenc.benchmark.runner import attach_validation, encode_with_retries
from streamenc.decision.catalog import BASELINE_CONFIG_ID, baseline_config
from streamenc.decision.engine import select_encoder
from streamenc.util.logging import get_logger

log = get_logger("phase6")


def _ensure_nvenc() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if enc not in parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([enc, *parts])


def _save(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet failed: %s", exc)


def held_out_clips(dataset: Path, *, n_per_kind: int = 3) -> list[ClipInfo]:
    """Family-safe held-out set: prefer manifest test, else hash-test families."""
    clips = discover_clips(dataset)
    # Exclude Phase-4 train families to keep the eval honest.
    phase4_train = {
        "ai_easy_01",
        "ai_easy_05",
        "rnd_easy_01",
        "rnd_easy_02",
        "scr_easy_05",
        "scr_easy_06",
    }
    test_pool = [
        c
        for c in clips
        if c.split == "test" and c.family not in phase4_train and c.resolution_tag != "1080"
    ]
    # If too few manifest-test clips, add other non-train families (prefer hard/medium).
    if len({c.family for c in test_pool}) < n_per_kind * 3:
        extras = [
            c
            for c in clips
            if c.family not in phase4_train
            and c.resolution_tag != "1080"
            and c.split != "train"  # keep leakage-safe when possible
        ]
        # Fall back to any non-phase4-train family if still short.
        if len({c.family for c in test_pool} | {c.family for c in extras}) < n_per_kind * 3:
            extras = [
                c
                for c in clips
                if c.family not in phase4_train and c.resolution_tag != "1080"
            ]
        for c in sorted(extras, key=lambda x: (0 if "hard" in x.clip_id else 1, x.clip_id)):
            if all(c.family != x.family for x in test_pool):
                test_pool.append(c)

    # Diversify by kind
    out: list[ClipInfo] = []
    for kind in ("ai", "render", "screen"):
        pool = sorted([c for c in test_pool if c.kind == kind], key=lambda c: c.clip_id)
        seen = set()
        n = 0
        for c in pool:
            if c.family in seen:
                continue
            seen.add(c.family)
            out.append(c)
            n += 1
            if n >= n_per_kind:
                break
    return out


def _run_arm(
    clip: ClipInfo,
    *,
    label: str,
    cfg,
    out_path: Path,
    feature_seconds: float = 0.0,
    decision_meta: dict | None = None,
) -> dict:
    t0 = time.perf_counter()
    enc = encode_with_retries(clip, cfg, str(out_path))
    enc = attach_validation(enc, clip.path, run_vmaf=True)
    wall = time.perf_counter() - t0
    row = enc.to_dict()
    row["arm"] = label
    row["feature_seconds"] = feature_seconds
    row["arm_wall_seconds"] = wall + feature_seconds
    row["encode_only_seconds"] = enc.encode_seconds
    if decision_meta:
        row.update(decision_meta)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "phase6"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "phase6"))
    parser.add_argument("--per-kind", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prefer-quality", action="store_true", default=True)
    parser.add_argument("--prefer-speed", action="store_true", help="disable quality bias")
    args = parser.parse_args()
    _ensure_nvenc()

    prefer_quality = not args.prefer_speed
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    clips = held_out_clips(dataset, n_per_kind=args.per_kind)
    if args.limit > 0:
        clips = clips[: args.limit]
    (results_dir / "eval_clips.json").write_text(
        json.dumps([c.to_dict() for c in clips], indent=2)
    )
    log.info("Phase6 held-out eval: %d clips", len(clips))
    for c in clips:
        log.info("  %s kind=%s fam=%s split=%s", c.clip_id, c.kind, c.family, c.split)

    rows: list[dict] = []
    decisions: list[dict] = []
    baseline_cfg = baseline_config()

    for i, clip in enumerate(clips, 1):
        log.info("[%d/%d] decide %s", i, len(clips), clip.clip_id)
        eng = select_encoder(clip.path, prefer_quality=prefer_quality)
        dec = eng.decision
        adaptive_cfg = eng.encoder_config()
        decisions.append(eng.to_dict())
        log.info(
            "  rule=%s → %s conf=%.2f prefilter=%s feat_t=%.3fs",
            dec.rule_id,
            dec.config_id,
            dec.confidence,
            dec.prefilter,
            eng.feature_seconds,
        )

        # Adaptive arm
        adapt_out = out_dir / "adaptive" / f"{clip.clip_id}.mp4"
        adapt_out.parent.mkdir(parents=True, exist_ok=True)
        log.info("  encode adaptive %s", adaptive_cfg.config_id)
        adapt_row = _run_arm(
            clip,
            label="adaptive",
            cfg=adaptive_cfg,
            out_path=adapt_out,
            feature_seconds=eng.feature_seconds,
            decision_meta={
                "decision_config_id": dec.config_id,
                "decision_rule_id": dec.rule_id,
                "decision_reason": dec.reason,
                "decision_confidence": dec.confidence,
                "decision_prefilter": dec.prefilter,
            },
        )
        rows.append(adapt_row)
        log.info(
            "  adaptive pass=%s vmaf=%s score=%s t=%.2fs",
            adapt_row.get("validation_passed"),
            adapt_row.get("vmaf"),
            adapt_row.get("clip_score"),
            adapt_row["arm_wall_seconds"],
        )

        # Baseline arm
        base_out = out_dir / "baseline" / f"{clip.clip_id}.mp4"
        base_out.parent.mkdir(parents=True, exist_ok=True)
        log.info("  encode baseline %s", BASELINE_CONFIG_ID)
        base_row = _run_arm(
            clip,
            label="baseline",
            cfg=baseline_cfg,
            out_path=base_out,
            feature_seconds=0.0,
            decision_meta={
                "decision_config_id": BASELINE_CONFIG_ID,
                "decision_rule_id": "BASELINE",
                "decision_reason": "fixed_baseline",
                "decision_confidence": 1.0,
                "decision_prefilter": "none",
            },
        )
        rows.append(base_row)
        log.info(
            "  baseline pass=%s vmaf=%s score=%s t=%.2fs",
            base_row.get("validation_passed"),
            base_row.get("vmaf"),
            base_row.get("clip_score"),
            base_row["arm_wall_seconds"],
        )

    df = pd.DataFrame(rows)
    _save(df, results_dir / "eval_experiments")
    _save(pd.DataFrame(decisions), results_dir / "decisions")

    # Pairwise comparison
    adapt = df[df.arm == "adaptive"].set_index("clip_id")
    base = df[df.arm == "baseline"].set_index("clip_id")
    common = sorted(set(adapt.index) & set(base.index))
    cmp_rows = []
    for cid in common:
        a, b = adapt.loc[cid], base.loc[cid]
        a_score = float(a["clip_score"]) if pd.notna(a["clip_score"]) and a["validation_passed"] else 0.0
        b_score = float(b["clip_score"]) if pd.notna(b["clip_score"]) and b["validation_passed"] else 0.0
        cmp_rows.append(
            {
                "clip_id": cid,
                "kind": a["kind"],
                "rule_id": a.get("decision_rule_id"),
                "adaptive_config": a.get("decision_config_id"),
                "adaptive_vmaf": a.get("vmaf"),
                "baseline_vmaf": b.get("vmaf"),
                "adaptive_score": a_score,
                "baseline_score": b_score,
                "delta_score": a_score - b_score,
                "adaptive_pass": bool(a["validation_passed"]),
                "baseline_pass": bool(b["validation_passed"]),
                "adaptive_wall_s": float(a["arm_wall_seconds"]),
                "baseline_wall_s": float(b["arm_wall_seconds"]),
                "delta_wall_s": float(a["arm_wall_seconds"]) - float(b["arm_wall_seconds"]),
            }
        )
    cmp = pd.DataFrame(cmp_rows)
    _save(cmp, results_dir / "adaptive_vs_baseline")

    mean_delta = float(cmp["delta_score"].mean()) if not cmp.empty else 0.0
    wins = int((cmp["delta_score"] > 1e-6).sum()) if not cmp.empty else 0
    losses = int((cmp["delta_score"] < -1e-6).sum()) if not cmp.empty else 0
    ties = int(len(cmp) - wins - losses)
    adaptive_mean = float(cmp["adaptive_score"].mean()) if not cmp.empty else 0.0
    baseline_mean = float(cmp["baseline_score"].mean()) if not cmp.empty else 0.0
    beats = bool(mean_delta > 0.001 and wins >= losses)

    summary = {
        "n_clips": len(common),
        "prefer_quality": prefer_quality,
        "baseline_config": BASELINE_CONFIG_ID,
        "adaptive_mean_clip_score": adaptive_mean,
        "baseline_mean_clip_score": baseline_mean,
        "mean_delta_clip_score": mean_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "adaptive_mean_wall_s": float(cmp["adaptive_wall_s"].mean()) if not cmp.empty else None,
        "baseline_mean_wall_s": float(cmp["baseline_wall_s"].mean()) if not cmp.empty else None,
        "beats_baseline": beats,
        "ml_trained": False,
        "ml_skipped_reason": (
            None
            if not beats
            else "rules already beat baseline; Phase-4 labelled set too small for robust LightGBM/CatBoost"
        ),
        "rule_counts": cmp["rule_id"].value_counts().to_dict() if not cmp.empty else {},
        "recommendation": (
            "ship_rules"
            if beats
            else "keep_fixed_baseline"
        ),
    }
    (results_dir / "phase6_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase6 summary:\n%s", json.dumps(summary, indent=2))
    if not cmp.empty:
        log.info("Per-clip deltas:\n%s", cmp.to_string(index=False))
    log.info("Phase6 complete → %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
