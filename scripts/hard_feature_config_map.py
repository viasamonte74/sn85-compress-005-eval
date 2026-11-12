#!/usr/bin/env python3
"""Map generated-hard features → encoder/settings that maximize score.

Pipeline:
  1. Extract Phase-5b features for every generated hard clip.
  2. Probe encode (baseline NVENC) → achieved_util feature (bitrate feasibility).
  3. Stratified config sweep under a short aim ladder; rank by clip_score then VMAF.
  4. Aggregate winners into regime/recipe → config policy for hard-path selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.decision.catalog import get_config
from streamenc.decision.hardpath import is_hard_content
from streamenc.encoders.generic import EncoderConfig, build_encode_argv
from streamenc.features.analyze import analyze_video
from streamenc.features.difficulty import difficulty_score
from streamenc.features.extract import FEATURE_NAMES
from streamenc.probe import probe_video
from streamenc.util.logging import get_logger
from streamenc.util.subprocess_run import run_command
from streamenc.validate.official_streaming import (
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)
from streamenc.validate.validate_clip import validate_clip

log = get_logger("hard_feat_map")

ARM_IDS = [
    "av1_nvenc_p6_noaq",
    "av1_nvenc_p6_vbr",
    "hevc_nvenc_p6_vbr",
    "svt_av1_p7_psnr",
    "hard_nvenc_p7_fill",
    "hard_nvenc_p6_noaq_fill",
    "hard_svt_p5_psnr",
]

# Short ladder: stop early once valid; include aggressive aims for incompressible content.
MAP_AIMS = (0.92, 0.85, 0.75, 0.62, 0.50)


def _recipe_from_stem(stem: str) -> str:
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p == "sharp" and i + 1 < len(parts) and parts[i + 1] == "grain":
            return "sharp_grain"
        if p == "edge" and i + 1 < len(parts) and parts[i + 1] == "boost":
            return "edge_boost"
        if p == "shake" and i + 1 < len(parts) and parts[i + 1] == "grain":
            return "shake_grain"
        if p == "grain":
            return "grain"
    return "unknown"


def _source_from_stem(stem: str) -> str:
    for tag in ("ai_hard_", "ai_medium_", "rnd_hard_", "scr_hard_"):
        if tag in stem:
            return stem[stem.index(tag) :]
    return "unknown"


def feature_regime(feats: dict) -> str:
    noise = float(feats.get("noise_estimate") or 0.0)
    hf = float(feats.get("highfreq_energy") or 0.0)
    motion = float(feats.get("motion_complexity") or 0.0)
    spatial = float(feats.get("spatial_complexity") or 0.0)
    spatial_p95 = float(feats.get("spatial_p95") or 0.0)
    edge = float(feats.get("edge_density") or 0.0)
    util = float(feats.get("achieved_util") or 0.0)
    # Probe util is ground truth when present; else content proxy.
    if util >= 1.15 or (hf >= 19.0 and spatial_p95 >= 95.0):
        return "bitrate_incompressible"
    if noise >= 3.5 or hf >= 12.0:
        return "noisy_hf"
    if edge >= 0.06 and spatial >= 10.0:
        return "edge_heavy"
    if motion >= 0.06:
        return "high_motion"
    if spatial >= 12.0:
        return "spatial_heavy"
    return "generic_hard"


def collect_clips(
    generated: Path,
    labeled: list[Path],
    *,
    limit: int | None,
    stratify: bool,
) -> list[Path]:
    gen = sorted(generated.glob("gen_hard_*.mkv"))
    if stratify and gen:
        by_recipe: dict[str, list[Path]] = defaultdict(list)
        for p in gen:
            by_recipe[_recipe_from_stem(p.stem)].append(p)
        paths: list[Path] = []
        recipes = sorted(by_recipe)
        target = limit if limit is not None else max(12, 3 * len(recipes))
        idx = 0
        while len(paths) < target:
            progressed = False
            for r in recipes:
                bucket = by_recipe[r]
                if idx < len(bucket):
                    paths.append(bucket[idx])
                    progressed = True
                    if len(paths) >= target:
                        break
            if not progressed:
                break
            idx += 1
    else:
        paths = list(gen)
        if limit is not None and len(paths) > limit:
            step = max(1, len(paths) // limit)
            paths = paths[::step][:limit]
    for p in labeled:
        if p.exists() and p not in paths:
            paths.append(p)
    return paths


def resolve_arms() -> list[tuple[str, EncoderConfig]]:
    arms = []
    for cid in ARM_IDS:
        try:
            arms.append((cid, get_config(cid)))
        except KeyError:
            log.warning("skip missing config %s", cid)
    return arms


def _ensure_nvenc() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc


def trial_encode(
    input_path: str,
    output_path: str,
    cfg: EncoderConfig,
    *,
    aims: tuple[float, ...] = MAP_AIMS,
) -> dict:
    """Fast aim ladder: return first streaming-valid encode, else last attempt metrics."""
    meta = probe_video(input_path, count_frames=True)
    target = streaming_target_kbps(
        width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05
    )
    gop = max(1, int(2.0 * meta.fps + 1e-9))
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=".hfm-", suffix=".mp4", dir=out_dir)
    os.close(fd)

    encode_seconds = 0.0
    last = {
        "ok": False,
        "passed_streaming": False,
        "avg_kbps": None,
        "peak_kbps": None,
        "util": None,
        "aim": None,
        "attempts": 0,
        "codes": [],
    }
    try:
        for aim in aims:
            last["attempts"] += 1
            last["aim"] = aim
            argv = build_encode_argv(
                cfg=cfg,
                input_path=input_path,
                output_path=partial,
                kbps=target * aim,
                gop=gop,
                pix_fmt="yuv420p",
                color_space=meta.color_space or "bt709",
                color_primaries=meta.color_primaries or "bt709",
                color_transfer=meta.color_transfer or "bt709",
                color_range=meta.color_range or "tv",
            )
            t0 = time.perf_counter()
            proc = run_command(argv, timeout=300)
            encode_seconds += time.perf_counter() - t0
            if not proc.ok or not os.path.exists(partial) or os.path.getsize(partial) == 0:
                continue
            packets = read_packets_ffprobe(partial)
            meas = measure_streaming(
                packets, fps=meta.fps, frames=meta.nb_frames, peak_window_seconds=2.0
            )
            vs = streaming_violations(
                meas,
                target_kbps=target,
                peak_ratio=1.5,
                peak_window_seconds=2.0,
                max_keyframe_interval_seconds=2.0,
                fps=meta.fps,
            )
            codes = sorted({v.code for v in vs})
            util = float(meas.avg_kbps) / max(target, 1e-6)
            last.update(
                {
                    "avg_kbps": meas.avg_kbps,
                    "peak_kbps": meas.peak_kbps,
                    "util": util,
                    "codes": codes,
                }
            )
            if not codes:
                os.chmod(partial, 0o644)
                os.replace(partial, output_path)
                partial = ""
                last["ok"] = True
                last["passed_streaming"] = True
                break
        else:
            # Keep last attempt for optional VMAF proxy ranking.
            if os.path.exists(partial) and os.path.getsize(partial) > 0:
                os.chmod(partial, 0o644)
                os.replace(partial, output_path)
                partial = ""
                last["ok"] = True  # file exists; streaming may have failed
    finally:
        if partial and os.path.exists(partial):
            try:
                os.unlink(partial)
            except OSError:
                pass
    last["encode_seconds"] = encode_seconds
    last["target_kbps"] = target
    return last


def probe_achieved_util(path: Path) -> float:
    """Quick baseline encode → util; accurate bitrate-feasibility feature."""
    out = Path("/tmp") / f"hfm_probe_{path.stem}.mp4"
    try:
        res = trial_encode(str(path), str(out), get_config("av1_nvenc_p6_noaq"), aims=(0.90, 0.70, 0.55))
        return float(res.get("util") or 0.0)
    finally:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass


def rank_key(row: dict) -> tuple:
    score = float(row.get("clip_score") or 0.0)
    streaming = 1 if row.get("passed_streaming") else 0
    passed = 1 if row.get("passed") else 0
    vmaf = float(row["vmaf"]) if row.get("vmaf") is not None else -1.0
    # Prefer closer-to-budget util when still over; prefer higher util when under.
    util = float(row.get("util") or 0.0)
    util_score = -abs(util - 0.95) if util else -9.0
    t = -float(row.get("encode_seconds") or 1e9)
    return (streaming, passed, score, vmaf, util_score, t)


def build_policy(winners: pd.DataFrame, trials: pd.DataFrame) -> dict:
    regime_winners: dict[str, Counter] = defaultdict(Counter)
    recipe_winners: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    for _, w in winners.iterrows():
        overall[w["winner_arm"]] += 1
        regime_winners[w["regime"]][w["winner_arm"]] += 1
        recipe_winners[w["recipe"]][w["winner_arm"]] += 1

    def top(counter: Counter, default: str) -> str:
        return counter.most_common(1)[0][0] if counter else default

    default = top(overall, "hard_svt_p5_psnr")
    by_regime = {r: top(c, default) for r, c in regime_winners.items()}
    by_recipe = {r: top(c, default) for r, c in recipe_winners.items()}

    arm_stats = (
        trials.groupby("arm")
        .agg(
            mean_vmaf=("vmaf", "mean"),
            mean_score=("clip_score", "mean"),
            stream_pass_rate=("passed_streaming", "mean"),
            gate_pass_rate=("passed", "mean"),
            mean_util=("util", "mean"),
            n=("arm", "count"),
            mean_encode_s=("encode_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["mean_score", "mean_vmaf"], ascending=False)
    )

    win_feat_means = {}
    for arm, g in winners.groupby("winner_arm"):
        cols = [c for c in list(FEATURE_NAMES) + ["achieved_util", "difficulty_score"] if c in g.columns]
        win_feat_means[arm] = {c: float(g[c].mean()) for c in cols}

    rules = []
    for regime, counter in regime_winners.items():
        rules.append(
            {
                "when": f"regime=={regime}",
                "config_id": top(counter, default),
                "support": int(sum(counter.values())),
            }
        )
    rules.append({"when": "default_hard", "config_id": default, "support": int(len(winners))})

    return {
        "default_hard_config_id": default,
        "by_regime": by_regime,
        "by_recipe": by_recipe,
        "rules": rules,
        "overall_win_counts": dict(overall),
        "arm_leaderboard": arm_stats.to_dict(orient="records"),
        "winner_feature_means": win_feat_means,
        "n_clips": int(len(winners)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", default="/workspace/dataset/generated_hard")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "hard_feature_map"))
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "hard_feature_map"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--no-labeled", action="store_true")
    parser.add_argument("--skip-probe-util", action="store_true")
    parser.add_argument("--features-csv", default="", help="Reuse existing features.csv")
    args = parser.parse_args()
    _ensure_nvenc()

    labeled = [] if args.no_labeled else [
        Path("/workspace/dataset/ai/ai_hard_06.mkv"),
        Path("/workspace/dataset/ai/ai_hard_03.mkv"),
    ]
    limit = args.limit if args.limit and args.limit > 0 else None
    clips = collect_clips(
        Path(args.generated),
        labeled,
        limit=limit,
        stratify=not args.no_stratify,
    )
    arms = resolve_arms()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gen = sorted(Path(args.generated).glob("gen_hard_*.mkv"))
    sweep_stems = {p.stem for p in clips}
    feature_rows: list[dict] = []
    feat_csv = Path(args.features_csv) if args.features_csv else results_dir / "features.csv"

    if feat_csv.is_file() and args.features_csv:
        log.info("Reusing features from %s", feat_csv)
        feats_df = pd.read_csv(feat_csv)
        feature_rows = feats_df.to_dict(orient="records")
        feat_by_id = {r["clip_id"]: r for r in feature_rows}
        # Ensure labeled sweep clips present.
        for path in clips:
            if path.stem in feat_by_id:
                continue
            log.info("[feat labeled] %s", path.name)
            res = analyze_video(path, mode="sampled", max_samples=24, max_width=640)
            feats = res.features
            diff = difficulty_score(feats)
            feat_dict = {k: getattr(feats, k) for k in FEATURE_NAMES}
            achieved_util = 0.0 if args.skip_probe_util else probe_achieved_util(path)
            feat_dict["achieved_util"] = achieved_util
            row = {
                "clip_id": path.stem,
                "path": str(path),
                "recipe": "labeled",
                "source": path.stem,
                "regime": feature_regime(feat_dict),
                "generated": False,
                "difficulty_score": diff.score,
                "is_hard_pred": diff.is_hard,
                "hard_flag": is_hard_content(feats),
                "hard_reasons": ",".join(diff.reasons),
                "feature_seconds": res.total_seconds,
                "achieved_util": achieved_util,
                "swept": True,
                **{k: feat_dict[k] for k in FEATURE_NAMES},
            }
            feature_rows.append(row)
            feat_by_id[path.stem] = row
        feats_df = pd.DataFrame(feature_rows)
        feats_df.to_csv(results_dir / "features.csv", index=False)
    else:
        for i, path in enumerate(all_gen, 1):
            log.info("[feat %d/%d] %s", i, len(all_gen), path.name)
            res = analyze_video(path, mode="sampled", max_samples=24, max_width=640)
            feats = res.features
            diff = difficulty_score(feats)
            feat_dict = {k: getattr(feats, k) for k in FEATURE_NAMES}
            achieved_util = 0.0
            if not args.skip_probe_util:
                log.info("  probe util…")
                achieved_util = probe_achieved_util(path)
                feat_dict["achieved_util"] = achieved_util
            else:
                feat_dict["achieved_util"] = 0.0
            regime = feature_regime(feat_dict)
            feature_rows.append(
                {
                    "clip_id": path.stem,
                    "path": str(path),
                    "recipe": _recipe_from_stem(path.stem),
                    "source": _source_from_stem(path.stem),
                    "regime": regime,
                    "generated": True,
                    "difficulty_score": diff.score,
                    "is_hard_pred": diff.is_hard,
                    "hard_flag": is_hard_content(feats),
                    "hard_reasons": ",".join(diff.reasons),
                    "feature_seconds": res.total_seconds,
                    "achieved_util": achieved_util,
                    "swept": path.stem in sweep_stems,
                    **{k: feat_dict[k] for k in FEATURE_NAMES},
                }
            )

        feat_by_id = {r["clip_id"]: r for r in feature_rows}
        for path in clips:
            if path.stem in feat_by_id:
                continue
            log.info("[feat labeled] %s", path.name)
            res = analyze_video(path, mode="sampled", max_samples=24, max_width=640)
            feats = res.features
            diff = difficulty_score(feats)
            feat_dict = {k: getattr(feats, k) for k in FEATURE_NAMES}
            achieved_util = 0.0 if args.skip_probe_util else probe_achieved_util(path)
            feat_dict["achieved_util"] = achieved_util
            row = {
                "clip_id": path.stem,
                "path": str(path),
                "recipe": "labeled",
                "source": path.stem,
                "regime": feature_regime(feat_dict),
                "generated": False,
                "difficulty_score": diff.score,
                "is_hard_pred": diff.is_hard,
                "hard_flag": is_hard_content(feats),
                "hard_reasons": ",".join(diff.reasons),
                "feature_seconds": res.total_seconds,
                "achieved_util": achieved_util,
                "swept": True,
                **{k: feat_dict[k] for k in FEATURE_NAMES},
            }
            feature_rows.append(row)
            feat_by_id[path.stem] = row

        feats_df = pd.DataFrame(feature_rows)
        feats_df.to_csv(results_dir / "features.csv", index=False)
    log.info("clips=%d arms=%d → ~%d encodes", len(clips), len(arms), len(clips) * len(arms))

    trial_rows = []
    for i, path in enumerate(clips, 1):
        base = feat_by_id[path.stem]
        feat_dict = {k: base[k] for k in FEATURE_NAMES}
        feat_dict["achieved_util"] = base.get("achieved_util", 0.0)
        log.info("[%d/%d] sweep %s regime=%s util=%.3f", i, len(clips), path.name, base["regime"], feat_dict["achieved_util"])

        # Incompressible clips: only try the strongest few arms (bitrate rarely recoverable).
        clip_arms = arms
        if float(base.get("achieved_util") or 0.0) >= 1.15 or base.get("regime") == "bitrate_incompressible":
            keep = {"hevc_nvenc_p6_vbr", "hard_nvenc_p6_noaq_fill", "hard_svt_p5_psnr"}
            clip_arms = [(n, c) for n, c in arms if n in keep]
            log.info("  incompressible → restricted arms %s", [n for n, _ in clip_arms])

        for arm_name, cfg in clip_arms:
            out = out_dir / arm_name / f"{path.stem}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            log.info("  arm %s", arm_name)
            aims = MAP_AIMS
            if float(base.get("achieved_util") or 0.0) >= 1.15:
                aims = (0.70, 0.55, 0.40)
            enc = trial_encode(str(path), str(out), cfg, aims=aims)
            vmaf = vmaf_neg = delta = cscore = None
            passed = False
            if enc["ok"] and out.exists():
                # Only spend VMAF when streaming passed (score-relevant); else proxy skip.
                if enc["passed_streaming"]:
                    report = validate_clip(str(path), str(out), run_vmaf=True, run_perceptual=False)
                    passed = report.passed
                    if "vmaf" in report.metrics:
                        vmaf = float(report.metrics["vmaf"])
                        vmaf_neg = float(report.metrics.get("vmaf_neg") or 0.0)
                        delta = float(report.metrics.get("vmaf_delta") or 0.0)
                        cscore = float(report.metrics.get("clip_score", 0.0)) if passed else 0.0
                    if not passed:
                        cscore = 0.0
                else:
                    # Lightweight: no VMAF; score stays 0. Still records util for mapping.
                    passed = False
                    cscore = 0.0
            row = {
                "clip_id": path.stem,
                "recipe": base["recipe"],
                "source": base["source"],
                "regime": base["regime"],
                "generated": base["generated"],
                "arm": arm_name,
                "ok": enc["ok"],
                "passed_streaming": enc["passed_streaming"],
                "passed": passed,
                "vmaf": vmaf,
                "vmaf_neg": vmaf_neg,
                "vmaf_delta": delta,
                "clip_score": cscore if cscore is not None else 0.0,
                "avg_kbps": enc.get("avg_kbps"),
                "peak_kbps": enc.get("peak_kbps"),
                "util": enc.get("util"),
                "aim": enc.get("aim"),
                "encode_seconds": enc.get("encode_seconds"),
                "attempts": enc.get("attempts"),
                "codes": ",".join(enc.get("codes") or []),
                "difficulty_score": base["difficulty_score"],
                "achieved_util": feat_dict["achieved_util"],
                **{k: feat_dict[k] for k in FEATURE_NAMES},
            }
            trial_rows.append(row)
            log.info(
                "    stream=%s pass=%s vmaf=%s score=%s util=%s t=%.1fs",
                enc["passed_streaming"],
                passed,
                f"{vmaf:.2f}" if vmaf is not None else "-",
                f"{cscore:.4f}" if cscore is not None else "-",
                f"{enc.get('util'):.3f}" if enc.get("util") is not None else "-",
                enc.get("encode_seconds") or 0.0,
            )

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(results_dir / "trials.csv", index=False)

    winners = []
    for clip_id, g in trials_df.groupby("clip_id"):
        best = max(g.to_dict(orient="records"), key=rank_key)
        base = feat_by_id[clip_id]
        winners.append(
            {
                **{k: base[k] for k in base if k != "path"},
                "winner_arm": best["arm"],
                "winner_vmaf": best.get("vmaf"),
                "winner_clip_score": best.get("clip_score") or 0.0,
                "winner_passed": best.get("passed"),
                "winner_stream": best.get("passed_streaming"),
                "winner_util": best.get("util"),
                "winner_encode_s": best.get("encode_seconds"),
                "n_stream_pass": int(g["passed_streaming"].sum()),
                "n_gate_pass": int(g["passed"].sum()),
                "max_vmaf": float(g["vmaf"].max()) if g["vmaf"].notna().any() else None,
                "max_clip_score": float(g["clip_score"].max()),
            }
        )
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(results_dir / "winners.csv", index=False)

    policy = build_policy(winners_df, trials_df)
    policy_path = results_dir / "hard_config_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2))
    runtime_policy = ROOT / "streamenc" / "decision" / "hard_config_policy.json"
    runtime_policy.write_text(json.dumps(policy, indent=2))

    summary = {
        "n_feature_clips": len(feats_df),
        "n_sweep_clips": len(winners_df),
        "n_generated_features": int(feats_df["generated"].sum()),
        "n_arms": len(arms),
        "n_trials": len(trials_df),
        "overall_win_counts": policy["overall_win_counts"],
        "default_hard_config_id": policy["default_hard_config_id"],
        "by_regime": policy["by_regime"],
        "by_recipe": policy["by_recipe"],
        "rules": policy["rules"],
        "arm_leaderboard_top5": policy["arm_leaderboard"][:5],
        "mean_winner_vmaf": float(winners_df["winner_vmaf"].dropna().mean())
        if winners_df["winner_vmaf"].notna().any()
        else None,
        "mean_winner_score": float(winners_df["winner_clip_score"].mean()),
        "fraction_stream_pass": float(winners_df["winner_stream"].mean()),
        "fraction_score_gt0": float((winners_df["winner_clip_score"] > 0).mean()),
        "incompressible_rate": float((feats_df["regime"] == "bitrate_incompressible").mean()),
        "policy_path": str(policy_path),
        "runtime_policy_path": str(runtime_policy),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
