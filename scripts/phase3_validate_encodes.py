#!/usr/bin/env python3
"""Validate Phase 2 encodes against their FFV1 references."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.validate import validate_clip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encodes",
        default=str(ROOT / "work" / "phase2_encodes"),
        help="directory of extensionless encodes (stem matches source basename)",
    )
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-vmaf", action="store_true")
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "phase3"))
    args = parser.parse_args()

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    encodes_dir = Path(args.encodes)
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map stem -> reference path
    refs = {}
    for sub in ("ai", "ai1080", "render", "screen"):
        for mkv in (dataset / sub).glob("*.mkv"):
            refs[mkv.stem] = mkv

    candidates = sorted(
        p for p in encodes_dir.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix == ""
    )
    if args.limit > 0:
        candidates = candidates[: args.limit]

    rows = []
    ok = fail = 0
    for cand in candidates:
        ref = refs.get(cand.name)
        if ref is None:
            print(f"SKIP {cand.name}: no reference", flush=True)
            continue
        report = validate_clip(
            str(ref),
            str(cand),
            run_vmaf=not args.no_vmaf,
            run_perceptual=not args.no_perceptual,
        )
        row = report.to_dict()
        row["clip"] = cand.name
        rows.append(row)
        if report.passed:
            ok += 1
            vmaf = report.metrics.get("vmaf")
            print(
                f"PASS {cand.name}"
                + (f" vmaf={vmaf:.3f} score={report.metrics.get('clip_score'):.3f}" if vmaf is not None else ""),
                flush=True,
            )
        else:
            fail += 1
            codes = ",".join(v["code"] for v in row["violations"])
            print(f"FAIL {cand.name}: {codes}", flush=True)

    jsonl = out_dir / "phase3_reports.jsonl"
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "total": len(rows),
        "ok": ok,
        "failed": fail,
        "jsonl": str(jsonl),
        "mean_vmaf": None,
        "mean_clip_score": None,
    }
    vmafs = [r["metrics"]["vmaf"] for r in rows if r.get("passed") and "vmaf" in r.get("metrics", {})]
    scores = [
        r["metrics"]["clip_score"] for r in rows if r.get("passed") and "clip_score" in r.get("metrics", {})
    ]
    if vmafs:
        summary["mean_vmaf"] = sum(vmafs) / len(vmafs)
    if scores:
        summary["mean_clip_score"] = sum(scores) / len(scores)
    (out_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
