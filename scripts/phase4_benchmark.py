#!/usr/bin/env python3
"""Phase 4: staged encoder benchmark → CSV/Parquet + competition-score leaderboard."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from streamenc.benchmark.discover import ClipInfo, discover_clips, select_benchmark_clips
from streamenc.benchmark.leaderboard import config_leaderboard, stage1_rank
from streamenc.benchmark.runner import ExperimentResult, attach_validation, encode_with_retries
from streamenc.encoders.generic import EncoderConfig, staged_configs
from streamenc.util.logging import get_logger

log = get_logger("phase4")


def _ensure_nvenc() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if enc not in parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([enc, *parts])


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _save_tables(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet write failed (%s); CSV only", exc)


def _run_stage1(
    clips: list[ClipInfo],
    configs: list[EncoderConfig],
    work_dir: Path,
    jsonl: Path,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    total = len(clips) * len(configs)
    done = 0
    for cfg in configs:
        for clip in clips:
            done += 1
            out = work_dir / "stage1" / cfg.config_id / f"{clip.clip_id}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            log.info("[%d/%d] stage1 %s × %s", done, total, cfg.config_id, clip.clip_id)
            res = encode_with_retries(clip, cfg, str(out))
            # Stage-1 is encode+bitrate only (no VMAF).
            res.validation_passed = None
            row = res.to_dict()
            row["stage"] = 1
            row["content_kind"] = clip.kind
            row["content_tier"] = clip.tier
            _append_jsonl(jsonl, row)
            results.append(res)
            status = "OK" if res.encode_ok else f"FAIL:{res.encode_error[:80]}"
            log.info(
                "  -> %s aim=%.3f avg=%s peak=%s t=%.2fs attempts=%d",
                status,
                res.aim_fraction,
                f"{res.avg_kbps:.1f}" if res.avg_kbps is not None else "n/a",
                f"{res.peak_kbps:.1f}" if res.peak_kbps is not None else "n/a",
                res.encode_seconds,
                res.attempts,
            )
    return results


def _pick_promising(stage1: list[ExperimentResult], *, top_k: int) -> list[str]:
    """Promote a diverse shortlist: fastest reliable AV1/HEVC/SVT + p6 baseline."""
    df = pd.DataFrame([r.to_dict() for r in stage1])
    ranked = stage1_rank(df)
    if ranked.empty:
        return []
    ok = ranked[ranked["encode_pass_rate"] >= 0.5]
    if ok.empty:
        ok = ranked

    picked: list[str] = []

    def take_first(pred) -> None:
        for _, row in ok.iterrows():
            cid = str(row["config_id"])
            if cid in picked:
                continue
            if pred(row):
                picked.append(cid)
                return

    # Always keep Phase-2 baseline if it passed the screen.
    take_first(lambda r: r["config_id"] == "av1_nvenc_p6_vbr")
    # Fastest remaining NVENC AV1 (often multipass / preset variants).
    take_first(lambda r: r["codec"] == "av1_nvenc")
    # HEVC challenger.
    take_first(lambda r: r["codec"] == "hevc_nvenc")
    # Fastest SVT-AV1 (quality challenger; usually slower).
    take_first(lambda r: r["codec"] == "libsvtav1")
    # Another NVENC AV1 for RC/AQ coverage if budget remains.
    take_first(
        lambda r: r["codec"] == "av1_nvenc"
        and r["config_id"] not in {"av1_nvenc_p6_vbr"}
    )
    # Fill remaining slots by stage1 rank order.
    for _, row in ok.iterrows():
        if len(picked) >= top_k:
            break
        cid = str(row["config_id"])
        if cid not in picked:
            picked.append(cid)
    return picked[:top_k]


def _run_stage2(
    clips: list[ClipInfo],
    configs: list[EncoderConfig],
    work_dir: Path,
    jsonl: Path,
    *,
    reuse_stage1: dict[tuple[str, str], ExperimentResult] | None = None,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    total = len(clips) * len(configs)
    done = 0
    for cfg in configs:
        for clip in clips:
            done += 1
            key = (cfg.config_id, clip.clip_id)
            out = work_dir / "stage2" / cfg.config_id / f"{clip.clip_id}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            log.info("[%d/%d] stage2 VMAF %s × %s", done, total, cfg.config_id, clip.clip_id)
            if reuse_stage1 and key in reuse_stage1 and reuse_stage1[key].encode_ok:
                src = Path(reuse_stage1[key].output_path)
                if src.exists():
                    # Copy encode artifact into stage2 tree for provenance.
                    if not out.exists() or out.stat().st_size == 0:
                        out.write_bytes(src.read_bytes())
                        os.chmod(out, 0o644)
                    res = replace(reuse_stage1[key], output_path=str(out))
                else:
                    res = encode_with_retries(clip, cfg, str(out))
            else:
                res = encode_with_retries(clip, cfg, str(out))
            res = attach_validation(res, clip.path, run_vmaf=True)
            row = res.to_dict()
            row["stage"] = 2
            row["content_kind"] = clip.kind
            row["content_tier"] = clip.tier
            _append_jsonl(jsonl, row)
            results.append(res)
            log.info(
                "  -> pass=%s vmaf=%s clip_score=%s t=%.2fs codes=%s",
                res.validation_passed,
                f"{res.vmaf:.2f}" if res.vmaf is not None else "n/a",
                f"{res.clip_score:.3f}" if res.clip_score is not None else "n/a",
                res.encode_seconds,
                ",".join(res.violation_codes) or "-",
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "phase4"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "phase4"))
    parser.add_argument("--train-per-kind", type=int, default=2)
    parser.add_argument("--test-per-kind", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5, help="configs promoted to stage-2")
    parser.add_argument(
        "--configs",
        default="",
        help="comma-separated config_id filter (default: all staged_configs)",
    )
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    _ensure_nvenc()

    dataset = Path(args.dataset)
    work_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_clips = discover_clips(dataset)
    train_clips, test_clips = select_benchmark_clips(
        all_clips,
        train_per_kind=args.train_per_kind,
        test_per_kind=args.test_per_kind,
    )
    # Persist split for reproducibility / leakage audit.
    split_rows = [c.to_dict() for c in train_clips + test_clips]
    (results_dir / "clip_split.json").write_text(json.dumps(split_rows, indent=2))
    fam_train = {c.family for c in train_clips}
    fam_test = {c.family for c in test_clips}
    assert not (fam_train & fam_test), f"family leakage: {fam_train & fam_test}"

    configs = staged_configs()
    if args.configs.strip():
        want = {x.strip() for x in args.configs.split(",") if x.strip()}
        configs = [c for c in configs if c.config_id in want]
    (results_dir / "configs.json").write_text(
        json.dumps(
            [
                {
                    "config_id": c.config_id,
                    "codec": c.codec,
                    "stage": c.stage,
                    "params": c.params,
                    "notes": c.notes,
                }
                for c in configs
            ],
            indent=2,
        )
    )

    log.info(
        "Phase4 corpus: train=%d test=%d configs=%d (families train=%d test=%d)",
        len(train_clips),
        len(test_clips),
        len(configs),
        len(fam_train),
        len(fam_test),
    )
    for c in train_clips:
        log.info("  train %s kind=%s fam=%s res=%s", c.clip_id, c.kind, c.family, c.resolution_tag)
    for c in test_clips:
        log.info("  test  %s kind=%s fam=%s res=%s", c.clip_id, c.kind, c.family, c.resolution_tag)

    jsonl = results_dir / "experiments.jsonl"
    stage1_results: list[ExperimentResult] = []
    if not args.skip_stage1:
        if jsonl.exists():
            jsonl.unlink()
        stage1_results = _run_stage1(train_clips, configs, work_dir, jsonl)
        s1_df = pd.DataFrame([r.to_dict() for r in stage1_results])
        _save_tables(s1_df, results_dir / "stage1_experiments")
        s1_lb = stage1_rank(s1_df)
        _save_tables(s1_lb, results_dir / "stage1_leaderboard")
        log.info("Stage1 leaderboard:\n%s", s1_lb.to_string(index=False))
    else:
        # Reload from prior CSV; rewrite jsonl from stage1 then append stage2.
        p = results_dir / "stage1_experiments.csv"
        if not p.exists():
            raise SystemExit(f"--skip-stage1 requires {p}")
        s1_df = pd.read_csv(p)
        s1_lb = stage1_rank(s1_df)
        _save_tables(s1_lb, results_dir / "stage1_leaderboard")
        if jsonl.exists():
            jsonl.unlink()
        for _, row in s1_df.iterrows():
            cfg_params = row["config_params"]
            if isinstance(cfg_params, str):
                try:
                    cfg_params = json.loads(cfg_params)
                except json.JSONDecodeError:
                    cfg_params = {}
            res = ExperimentResult(
                clip_id=row["clip_id"],
                family=row["family"],
                kind=row["kind"],
                split=row["split"],
                resolution_tag=row["resolution_tag"],
                width=int(row["width"]),
                height=int(row["height"]),
                fps=float(row["fps"]),
                frames=int(row["frames"]),
                config_id=row["config_id"],
                codec=row["codec"],
                config_params=cfg_params if isinstance(cfg_params, dict) else {},
                target_kbps=float(row["target_kbps"]),
                aim_fraction=float(row["aim_fraction"]),
                avg_kbps=float(row["avg_kbps"]) if pd.notna(row["avg_kbps"]) else None,
                peak_kbps=float(row["peak_kbps"]) if pd.notna(row["peak_kbps"]) else None,
                avg_ok=bool(row["avg_ok"]) if pd.notna(row["avg_ok"]) else None,
                peak_ok=bool(row["peak_ok"]) if pd.notna(row["peak_ok"]) else None,
                keyframe_ok=bool(row["keyframe_ok"]) if pd.notna(row["keyframe_ok"]) else None,
                encode_seconds=float(row["encode_seconds"]),
                attempts=int(row["attempts"]),
                encode_ok=bool(row["encode_ok"]),
                encode_error=str(row.get("encode_error") or ""),
                output_path=str(row.get("output_path") or ""),
            )
            stage1_results.append(res)
            jl = res.to_dict()
            jl["stage"] = 1
            jl["content_kind"] = row.get("kind")
            jl["content_tier"] = row.get("tier") if "tier" in row else None
            _append_jsonl(jsonl, jl)
        log.info("Reloaded %d stage1 rows; leaderboard:\n%s", len(stage1_results), s1_lb.to_string(index=False))

    promising_ids = _pick_promising(stage1_results, top_k=args.top_k) if stage1_results else []
    cfg_by_id = {c.config_id: c for c in configs}
    promising = [cfg_by_id[i] for i in promising_ids if i in cfg_by_id]
    (results_dir / "promising_configs.json").write_text(json.dumps(promising_ids, indent=2))
    log.info("Promoted to stage2: %s", promising_ids)

    stage2_results: list[ExperimentResult] = []
    if not args.skip_stage2 and promising:
        reuse = {(r.config_id, r.clip_id): r for r in stage1_results}
        stage2_train = _run_stage2(train_clips, promising, work_dir, jsonl, reuse_stage1=reuse)
        stage2_results.extend(stage2_train)
        if not args.skip_test and test_clips:
            stage2_test = _run_stage2(test_clips, promising, work_dir, jsonl, reuse_stage1=None)
            stage2_results.extend(stage2_test)

        s2_df = pd.DataFrame([r.to_dict() for r in stage2_results])
        _save_tables(s2_df, results_dir / "stage2_experiments")
        for split in ("train", "test"):
            lb = config_leaderboard(s2_df, split=split)
            _save_tables(lb, results_dir / f"leaderboard_{split}")
            if not lb.empty:
                log.info("Leaderboard (%s):\n%s", split, lb.to_string(index=False))

    # Combined experiment table
    all_rows = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    all_df = pd.DataFrame(all_rows)
    if not all_df.empty:
        _save_tables(all_df, results_dir / "experiments")

    summary = {
        "train_clips": [c.clip_id for c in train_clips],
        "test_clips": [c.clip_id for c in test_clips],
        "train_families": sorted(fam_train),
        "test_families": sorted(fam_test),
        "n_configs_stage1": len(configs),
        "promising_configs": promising_ids,
        "n_stage1_rows": len(stage1_results),
        "n_stage2_rows": len(stage2_results),
        "results_dir": str(results_dir),
    }
    if stage2_results:
        s2_df = pd.DataFrame([r.to_dict() for r in stage2_results])
        for split in ("train", "test"):
            lb = config_leaderboard(s2_df, split=split)
            if not lb.empty:
                summary[f"top_{split}"] = lb.iloc[0].to_dict()
    (results_dir / "phase4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Phase4 complete → %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
