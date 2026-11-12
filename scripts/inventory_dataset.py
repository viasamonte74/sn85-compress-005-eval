#!/usr/bin/env python3
"""Inventory /workspace/dataset for Phase 1 reporting."""

from __future__ import annotations

import json
import collections
from pathlib import Path

from streamenc.probe import probe_video


def main() -> int:
    root = Path("/workspace/dataset")
    out = {
        "root": str(root),
        "exists": root.exists(),
        "mkv_by_subdir": {},
        "manifest": None,
        "samples": [],
    }
    if not root.exists():
        print(json.dumps(out, indent=2))
        return 1

    for sub in sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        mkvs = sorted((root / sub).glob("*.mkv"))
        out["mkv_by_subdir"][sub] = len(mkvs)

    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        out["manifest"] = {
            "entries": len(rows),
            "by_kind": dict(collections.Counter(r.get("kind") for r in rows)),
            "by_split": dict(collections.Counter(r.get("split") for r in rows)),
            "by_resolution": dict(
                collections.Counter(f"{r.get('width')}x{r.get('height')}" for r in rows)
            ),
            "fps_values": sorted({float(r.get("fps")) for r in rows if r.get("fps") is not None}),
            "note": "manifest.jsonl may lag newly added clips; use mkv_by_subdir for on-disk count",
        }

    # Probe a few representative clips (one per family if present).
    candidates = [
        root / "ai" / "ai_easy_01.mkv",
        root / "ai1080" / "ai_easy_01_1080.mkv",
        root / "render" / "rnd_easy_01.mkv",
        root / "screen" / "scr_easy_01.mkv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        meta = probe_video(path, count_frames=True)
        budget = meta.budget()
        out["samples"].append(
            {
                **meta.to_dict(),
                "target_kbps": budget.target_kbps,
                "average_limit_kbps": budget.average_limit_kbps,
                "peak_limit_kbps": budget.peak_limit_kbps,
                "gop_frames_2s": budget.gop_frames(2.0),
            }
        )

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
