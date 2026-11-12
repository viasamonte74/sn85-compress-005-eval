#!/usr/bin/env python3
"""A/B hard-path configs on hard clips: VMAF / clip_score / util / time."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.benchmark.discover import ClipInfo
from streamenc.correct.controller import HARD_AIM_LADDER, CorrectionController
from streamenc.decision.catalog import get_config
from streamenc.decision.hardpath import hard_path_configs, is_hard_content
from streamenc.features.analyze import analyze_video
from streamenc.util.logging import get_logger
from streamenc.validate.validate_clip import validate_clip

log = get_logger("hardpath_ab")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "hardpath"))
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "hardpath"))
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    # Candidate hard clips (known difficult + feature-flagged).
    candidates = [
        Path("/workspace/dataset/ai/ai_hard_03.mkv"),
        Path("/workspace/dataset/ai/ai_hard_06.mkv"),
        Path("/workspace/dataset/ai/ai_hard_12.mkv"),
        Path("/workspace/dataset/render/rnd_hard_01.mkv"),
        Path("/workspace/dataset/render/rnd_hard_06.mkv"),
        Path("/workspace/dataset/ai/ai_hard_01.mkv"),
    ]
    clips = []
    for p in candidates:
        if not p.exists():
            continue
        feats = analyze_video(p, mode="sampled", max_samples=24).features
        clips.append((p, feats, is_hard_content(feats)))
        if len(clips) >= args.limit:
            break

    # Arms: current production-ish baseline for hard + hard catalogue.
    arms = [
        ("baseline_noaq", get_config("av1_nvenc_p6_noaq"), False),
        ("baseline_p6", get_config("av1_nvenc_p6_vbr"), False),
        ("baseline_svt_p7", get_config("svt_av1_p7_psnr"), False),
    ]
    for cfg in hard_path_configs():
        arms.append((cfg.config_id, cfg, True))

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, feats, flagged in clips:
        log.info("clip %s hard_flag=%s motion=%.3f spatial=%.2f", path.stem, flagged, feats.motion_complexity, feats.spatial_complexity)
        for arm_name, cfg, hard_mode in arms:
            out = out_dir / arm_name / f"{path.stem}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            ctrl = CorrectionController(
                aim_ladder=HARD_AIM_LADDER if hard_mode else None,
                fill_up=hard_mode,
                fill_up_util_threshold=0.92,
            )
            log.info("  arm %s", arm_name)
            outcome = ctrl.encode(str(path), str(out), cfg)
            vmaf = vmaf_neg = delta = cscore = None
            passed = False
            if outcome.ok:
                report = validate_clip(str(path), str(out), run_vmaf=True, run_perceptual=True)
                passed = report.passed
                if "vmaf" in report.metrics:
                    vmaf = float(report.metrics["vmaf"])
                    vmaf_neg = float(report.metrics["vmaf_neg"])
                    delta = float(report.metrics["vmaf_delta"])
                    cscore = float(report.metrics.get("clip_score", 0.0)) if passed else 0.0
                if not passed:
                    cscore = 0.0
            rows.append(
                {
                    "clip_id": path.stem,
                    "hard_flag": flagged,
                    "arm": arm_name,
                    "hard_mode": hard_mode,
                    "ok": outcome.ok,
                    "passed": passed,
                    "vmaf": vmaf,
                    "vmaf_neg": vmaf_neg,
                    "vmaf_delta": delta,
                    "clip_score": cscore if cscore is not None else 0.0,
                    "avg_kbps": outcome.avg_kbps,
                    "peak_kbps": outcome.peak_kbps,
                    "aim": outcome.aim_fraction,
                    "encode_seconds": outcome.encode_seconds,
                    "attempts": outcome.attempts,
                    "motion": feats.motion_complexity,
                    "spatial": feats.spatial_complexity,
                }
            )
            log.info(
                "    ok=%s pass=%s vmaf=%s score=%s util_aim=%s t=%.2fs",
                outcome.ok,
                passed,
                f"{vmaf:.2f}" if vmaf is not None else "n/a",
                f"{cscore:.3f}" if cscore is not None else "n/a",
                outcome.aim_fraction,
                outcome.encode_seconds,
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "hardpath_ab.csv", index=False)

    # Rank arms by mean clip_score on hard_flag clips (fallback: all).
    hard_df = df[df["hard_flag"] == True] if df["hard_flag"].any() else df  # noqa: E712
    summary_rows = []
    for arm, g in hard_df.groupby("arm"):
        summary_rows.append(
            {
                "arm": arm,
                "n": len(g),
                "mean_clip_score": float(g["clip_score"].mean()),
                "mean_vmaf": float(g["vmaf"].dropna().mean()) if g["vmaf"].notna().any() else None,
                "pass_rate": float(g["passed"].mean()),
                "mean_encode_s": float(g["encode_seconds"].mean()),
                "mean_aim": float(g["aim"].dropna().mean()) if g["aim"].notna().any() else None,
            }
        )
    ranked = pd.DataFrame(summary_rows).sort_values(
        ["mean_clip_score", "mean_vmaf"], ascending=False
    )
    ranked.to_csv(results_dir / "hardpath_leaderboard.csv", index=False)

    baseline = ranked[ranked["arm"] == "baseline_noaq"]
    base_score = float(baseline.iloc[0]["mean_clip_score"]) if not baseline.empty else 0.0
    winner = ranked.iloc[0].to_dict() if not ranked.empty else {}
    beats = bool(winner) and float(winner["mean_clip_score"]) > base_score + 0.005

    # Also estimate impact on the 4-clip final mix if hard clip score improves.
    # final mix used ai_hard_06; look up winner score on that clip if present.
    impact = {}
    if "ai_hard_06" in set(df["clip_id"]):
        easy = [0.920053375, 0.89536875, 0.932076475]
        b = df[(df.clip_id == "ai_hard_06") & (df.arm == "baseline_noaq")]
        w = df[(df.clip_id == "ai_hard_06") & (df.arm == winner.get("arm"))]
        b_s = float(b.iloc[0]["clip_score"]) if not b.empty else 0.066
        w_s = float(w.iloc[0]["clip_score"]) if not w.empty else b_s
        q_b = (sum(easy) + b_s) / 4
        q_w = (sum(easy) + w_s) / 4
        impact = {
            "ai_hard_06_baseline_score": b_s,
            "ai_hard_06_winner_score": w_s,
            "quality_mean_baseline_proxy": q_b,
            "quality_mean_winner_proxy": q_w,
            "delta_quality_mean": q_w - q_b,
        }

    summary = {
        "n_clips": len(clips),
        "n_arms": len(arms),
        "baseline_noaq_mean_score": base_score,
        "winner": winner,
        "beats_baseline_noaq": beats,
        "recommended_hard_config_id": winner.get("arm") if beats and str(winner.get("arm", "")).startswith("hard_") else (
            winner.get("arm") if beats else DEFAULT_KEEP
        ),
        "impact_on_final_mix_proxy": impact,
        "leaderboard": ranked.to_dict(orient="records"),
    }
    # Prefer a hard_* id for wiring; if winner is baseline_svt_p7 map to hard_svt_p5_vq etc.
    rec = summary["recommended_hard_config_id"]
    if rec in {"baseline_svt_p7", "baseline_noaq", "baseline_p6"}:
        # Map nearest hard twin
        mapping = {
            "baseline_svt_p7": "hard_svt_p5_vq",
            "baseline_noaq": "hard_nvenc_p6_noaq_fill",
            "baseline_p6": "hard_nvenc_p7_fill",
        }
        if beats:
            summary["recommended_hard_config_id"] = mapping.get(rec, "hard_svt_p5_vq")
            summary["recommended_note"] = f"winner was {rec}; mapped to hard twin for fill-up ladder"
        else:
            summary["recommended_hard_config_id"] = "hard_nvenc_p6_noaq_fill"
            summary["recommended_note"] = "no clear win; keep fill-up noaq hard twin"

    (results_dir / "hardpath_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Hard-path summary:\n%s", json.dumps(summary, indent=2, default=str))
    return 0


DEFAULT_KEEP = "hard_nvenc_p6_noaq_fill"


if __name__ == "__main__":
    raise SystemExit(main())
