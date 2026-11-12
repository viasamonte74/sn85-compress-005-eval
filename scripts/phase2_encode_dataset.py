#!/usr/bin/env python3
"""Encode a representative dataset subset with the Phase 2 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.encode.baseline import encode_baseline
from streamenc.encoders import NvencAv1Config


def discover_clips(dataset: Path) -> list[Path]:
    clips: list[Path] = []
    for sub in ("ai", "ai1080", "render", "screen"):
        d = dataset / sub
        if d.is_dir():
            clips.extend(sorted(d.glob("*.mkv")))
    return clips


def pick_representative(clips: list[Path], per_family: int) -> list[Path]:
    families = {"ai": [], "ai1080": [], "render": [], "screen": []}
    for c in clips:
        families.setdefault(c.parent.name, []).append(c)
    picked: list[Path] = []
    for name, items in families.items():
        if not items:
            continue
        # Spread across list: easy/medium/hard naming tends to sort together.
        if len(items) <= per_family:
            picked.extend(items)
            continue
        step = max(1, len(items) // per_family)
        chosen = [items[i] for i in range(0, len(items), step)][:per_family]
        # Always include first and last for coverage.
        if items[0] not in chosen:
            chosen[0] = items[0]
        if items[-1] not in chosen:
            chosen[-1] = items[-1]
        picked.extend(chosen)
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "phase2_encodes"))
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="encode every mkv under dataset")
    parser.add_argument("--preset", default="p6")
    parser.add_argument("--limit", type=int, default=0, help="optional max clips")
    args = parser.parse_args()

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(dataset)
    if not args.all:
        clips = pick_representative(clips, args.per_family)
    if args.limit > 0:
        clips = clips[: args.limit]

    cfg = NvencAv1Config(preset=args.preset)
    rows = []
    ok_n = fail_n = 0
    for clip in clips:
        dest = out_dir / clip.stem  # no extension, competition style
        result = encode_baseline(str(clip), str(dest), cfg=cfg)
        row = asdict(result)
        rows.append(row)
        if result.ok:
            ok_n += 1
            print(
                f"OK {clip.name}: avg={result.avg_kbps:.1f}/{result.avg_limit_kbps:.1f} "
                f"peak={result.peak_kbps:.1f}/{result.peak_limit_kbps:.1f} "
                f"aim={result.aim_fraction} t={result.encode_seconds:.2f}s",
                flush=True,
            )
        else:
            fail_n += 1
            print(f"FAIL {clip.name}: {result.reason}", flush=True)

    json_path = out_dir / "phase2_results.jsonl"
    with json_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    csv_path = out_dir / "phase2_results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "total": len(rows),
        "ok": ok_n,
        "failed": fail_n,
        "preset": args.preset,
        "jsonl": str(json_path),
        "csv": str(csv_path),
    }
    print(json.dumps(summary, indent=2))
    (out_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
