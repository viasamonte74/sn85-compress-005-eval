#!/usr/bin/env python3
"""A/B score hard-clip adaptation: baseline hard-path vs adapted (+ optional dual)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.decision.hard_adapt import hard_adapt_plan
from streamenc.decision.hardpath import hard_feature_regime
from streamenc.features.analyze import analyze_video
from streamenc.pipeline.item import process_item
from streamenc.util.logging import get_logger
from streamenc.validate.validate_clip import validate_clip

log = get_logger("score_hard_adapt")

CLIPS = [
    Path("/workspace/dataset/ai/ai_hard_06.mkv"),
    Path("/workspace/dataset/ai/ai_hard_03.mkv"),
    Path("/workspace/dataset/ai/ai_hard_12.mkv"),
    Path("/workspace/dataset/ai/ai_hard_01.mkv"),
    Path("/workspace/dataset/render/rnd_hard_01.mkv"),
    Path("/workspace/dataset/render/rnd_hard_06.mkv"),
    Path("/workspace/dataset/generated_hard/gen_hard_01_grain_ai_hard_13.mkv"),
    Path("/workspace/dataset/generated_hard/gen_hard_02_sharp_grain_ai_hard_13.mkv"),
    Path("/workspace/dataset/generated_hard/gen_hard_18_edge_boost_ai_hard_13.mkv"),
    Path("/workspace/dataset/generated_hard/gen_hard_26_shake_grain_ai_hard_13.mkv"),
]


def _ensure_nvenc() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc


def score_one(path: Path, out: Path, **kwargs) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    item = process_item(str(path), str(out), **kwargs)
    row = {
        "clip_id": path.stem,
        "ok": item.ok,
        "config_id": item.config_id,
        "prefilter": item.prefilter,
        "rule_id": item.rule_id,
        "encode_seconds": item.encode_seconds,
        "attempts": item.attempts,
        "avg_kbps": item.avg_kbps,
        "passed": False,
        "vmaf": None,
        "clip_score": 0.0,
        "winner_tag": (item.extras.get("hard_adapt") or {}).get("winner_tag"),
        "regime": (item.extras.get("hard_adapt") or {}).get("regime"),
    }
    if item.ok and out.exists():
        report = validate_clip(str(path), str(out), run_vmaf=True, run_perceptual=True)
        row["passed"] = report.passed
        if "vmaf" in report.metrics:
            row["vmaf"] = float(report.metrics["vmaf"])
            row["clip_score"] = float(report.metrics.get("clip_score", 0.0)) if report.passed else 0.0
        if not report.passed:
            row["clip_score"] = 0.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "hard_adapt"))
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "hard_adapt"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    _ensure_nvenc()

    clips = [p for p in CLIPS if p.exists()]
    if args.limit and args.limit > 0:
        clips = clips[: args.limit]

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    arms = [
        ("baseline", {"use_decision": True, "hard_adapt": False, "allow_filter": False}),
        ("adapted", {"use_decision": True, "hard_adapt": True, "hard_dual": False, "allow_filter": True}),
        ("adapted_dual", {"use_decision": True, "hard_adapt": True, "hard_dual": True, "allow_filter": True}),
    ]

    rows = []
    for path in clips:
        feats = analyze_video(path, mode="sampled", max_samples=24, max_width=640).features
        plan = hard_adapt_plan(feats)
        regime = hard_feature_regime(feats)
        log.info("clip %s regime=%s plan=%s/%s", path.stem, regime, plan.primary_config_id, plan.primary_prefilter)
        for arm_name, kwargs in arms:
            log.info("  arm %s", arm_name)
            out = out_dir / arm_name / f"{path.stem}.mp4"
            row = score_one(path, out, **kwargs)
            row["arm"] = arm_name
            row["regime_feat"] = regime
            row["plan_config"] = plan.primary_config_id
            row["plan_prefilter"] = plan.primary_prefilter
            rows.append(row)
            log.info(
                "    ok=%s pass=%s vmaf=%s score=%s pf=%s cfg=%s t=%.1fs",
                row["ok"],
                row["passed"],
                f"{row['vmaf']:.2f}" if row["vmaf"] is not None else None,
                f"{row['clip_score']:.4f}",
                row["prefilter"],
                row["config_id"],
                row["encode_seconds"] or 0.0,
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "hard_adapt_ab.csv", index=False)

    # Per-arm summary
    summary_arms = (
        df.groupby("arm")
        .agg(
            mean_vmaf=("vmaf", "mean"),
            mean_score=("clip_score", "mean"),
            pass_rate=("passed", "mean"),
            ok_rate=("ok", "mean"),
            mean_encode_s=("encode_seconds", "mean"),
            n=("arm", "count"),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    # Paired deltas vs baseline
    piv = df.pivot_table(index="clip_id", columns="arm", values=["clip_score", "vmaf", "ok"])
    deltas = []
    for arm in ("adapted", "adapted_dual"):
        if ("clip_score", arm) not in piv.columns:
            continue
        score_d = (piv[("clip_score", arm)] - piv[("clip_score", "baseline")]).mean()
        vmaf_d = (piv[("vmaf", arm)] - piv[("vmaf", "baseline")]).mean()
        ok_d = (piv[("ok", arm)].astype(float) - piv[("ok", "baseline")].astype(float)).mean()
        deltas.append({"arm": arm, "delta_score": float(score_d), "delta_vmaf": float(vmaf_d) if pd.notna(vmaf_d) else None, "delta_ok_rate": float(ok_d)})

    # Best per clip
    winners = []
    for clip_id, g in df.groupby("clip_id"):
        best = g.sort_values(["clip_score", "vmaf", "ok"], ascending=False).iloc[0]
        winners.append(
            {
                "clip_id": clip_id,
                "best_arm": best["arm"],
                "best_score": best["clip_score"],
                "best_vmaf": best["vmaf"],
                "regime": best["regime_feat"],
            }
        )

    summary = {
        "n_clips": len(clips),
        "arms": summary_arms.to_dict(orient="records"),
        "deltas_vs_baseline": deltas,
        "per_clip_winners": winners,
        "verdict": None,
    }
    # Verdict
    adapted = next((a for a in deltas if a["arm"] == "adapted"), None)
    dual = next((a for a in deltas if a["arm"] == "adapted_dual"), None)
    best_delta = max(
        [d for d in deltas if d.get("delta_score") is not None],
        key=lambda d: (d["delta_score"], d.get("delta_vmaf") or -999),
        default=None,
    )
    if best_delta and (best_delta["delta_score"] > 0.001 or (best_delta.get("delta_vmaf") or 0) > 0.5):
        summary["verdict"] = f"{best_delta['arm']} improves hard clips (Δscore={best_delta['delta_score']:.4f}, ΔVMAF={best_delta.get('delta_vmaf')})"
    elif adapted and (adapted.get("delta_ok_rate") or 0) > 0.05:
        summary["verdict"] = "adapted raises streaming pass rate; VMAF/score still near floor"
    else:
        summary["verdict"] = "adaptation does not materially raise hard clip_score on this set"

    (results_dir / "hard_adapt_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
