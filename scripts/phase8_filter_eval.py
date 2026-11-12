#!/usr/bin/env python3
"""Phase 8: compare mild prefilters vs none on a small held-out set."""

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
from streamenc.benchmark.runner import attach_validation, encode_with_retries
from streamenc.decision.catalog import get_config
from streamenc.encoders.generic import EncoderConfig
from streamenc.filters.policy import FILTER_CATALOG
from streamenc.util.logging import get_logger

log = get_logger("phase8")


def _cfg_with_vf(base_id: str, filter_name: str) -> EncoderConfig:
    base = get_config(base_id)
    vf = FILTER_CATALOG[filter_name]
    params = dict(base.params)
    if vf:
        params["vf"] = vf
    return EncoderConfig(
        config_id=f"{base_id}__{filter_name}",
        codec=base.codec,
        stage=base.stage,
        params=params,
        notes=f"filter={filter_name}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "phase8"))
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "phase8"))
    parser.add_argument("--base-config", default="av1_nvenc_p6_vbr")
    args = parser.parse_args()

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    # Compact held-out-ish set: one per kind, prefer medium/hard diversity.
    paths = [
        Path("/workspace/dataset/ai/ai_easy_08.mkv"),
        Path("/workspace/dataset/render/rnd_easy_06.mkv"),
        Path("/workspace/dataset/screen/scr_easy_02.mkv"),
        Path("/workspace/dataset/ai/ai_hard_06.mkv"),
    ]
    filters = ["none", "hqdn3d_light", "hqdn3d_spatial"]
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in paths:
        if not path.exists():
            continue
        clip = ClipInfo(
            clip_id=path.stem,
            path=str(path),
            kind=path.parent.name if path.parent.name != "ai1080" else "ai",
            family=path.stem,
            resolution_tag="720",
            split="test",
        )
        for fname in filters:
            cfg = _cfg_with_vf(args.base_config, fname)
            out = out_dir / fname / f"{path.stem}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            log.info("%s × %s", path.stem, fname)
            enc_res = encode_with_retries(clip, cfg, str(out))
            enc_res = attach_validation(enc_res, str(path), run_vmaf=True)
            row = enc_res.to_dict()
            row["filter"] = fname
            rows.append(row)
            log.info(
                "  pass=%s vmaf=%s score=%s",
                enc_res.validation_passed,
                enc_res.vmaf,
                enc_res.clip_score,
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "filter_experiments.csv", index=False)
    # Compare each filter vs none per clip
    cmp_rows = []
    for clip_id, g in df.groupby("clip_id"):
        none = g[g["filter"] == "none"]
        if none.empty:
            continue
        base_score = float(none.iloc[0]["clip_score"] or 0) if none.iloc[0]["validation_passed"] else 0.0
        base_pass = bool(none.iloc[0]["validation_passed"])
        for _, r in g.iterrows():
            score = float(r["clip_score"] or 0) if r["validation_passed"] else 0.0
            cmp_rows.append(
                {
                    "clip_id": clip_id,
                    "filter": r["filter"],
                    "vmaf": r["vmaf"],
                    "clip_score": score,
                    "delta_vs_none": score - base_score,
                    "validation_passed": bool(r["validation_passed"]),
                    "none_passed": base_pass,
                }
            )
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(results_dir / "filter_vs_none.csv", index=False)
    summary = {
        "filters": filters,
        "n_rows": len(df),
        "mean_delta_by_filter": cmp.groupby("filter")["delta_vs_none"].mean().to_dict() if not cmp.empty else {},
        "any_gate_fail_with_filter": bool(
            ((cmp["filter"] != "none") & (~cmp["validation_passed"])).any()
        )
        if not cmp.empty
        else False,
        "recommendation": "none",
    }
    # Keep a filter only if mean delta > 0.005 and no new gate failures.
    best = None
    best_delta = 0.0
    if not cmp.empty:
        for fname, delta in summary["mean_delta_by_filter"].items():
            if fname == "none":
                continue
            fails = bool(((cmp["filter"] == fname) & (~cmp["validation_passed"])).any())
            if not fails and delta > best_delta:
                best, best_delta = fname, delta
    if best is not None and best_delta > 0.005:
        summary["recommendation"] = best
    else:
        summary["recommendation"] = "none"
        summary["reason"] = "insufficient competition-score benefit or gate risk"
    (results_dir / "phase8_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase8 summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
