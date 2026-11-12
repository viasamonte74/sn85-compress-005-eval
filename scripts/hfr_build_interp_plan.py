#!/usr/bin/env python3
"""Pick Wan prompts that exist at BOTH 720p and 1080p and write the RIFE interpolation plan (jsonl).

Pairs: dataset/ai/<id>.mkv + dataset/ai1080/<id>_1080.mkv, and dataset/ai_gen2/anim*/<id>.mkv +
<id>_1080.mkv. Each chosen prompt gets one target fps (same for both resolutions): 60/50/48 at 3:1:1.
Deterministic (seeded), so re-running after more 1080p clips exist only adds pairs.
"""
import json
import random
import sys
from pathlib import Path

D = Path("/workspace/dataset")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hfr_interp_plan.jsonl")
rng = random.Random(85)


def pairs(dir720: Path, dir1080: Path, prefix: str) -> dict[str, list[tuple[Path, Path]]]:
    by_tier: dict[str, list[tuple[Path, Path]]] = {}
    for p in sorted(dir720.glob(f"{prefix}_*.mkv")):
        if p.stem.endswith("_1080") or ".tmp" in p.name:
            continue
        q = dir1080 / f"{p.stem}_1080.mkv"
        if q.exists():
            tier = p.stem.split("_")[1]
            by_tier.setdefault(f"{prefix}:{tier}", []).append((p, q))
    return by_tier


pool = {}
pool.update(pairs(D / "ai", D / "ai1080", "ai"))
pool.update(pairs(D / "ai_gen2", D / "ai_gen2", "anim"))
pool.update(pairs(D / "ai_gen2", D / "ai_gen2", "animh"))
quota = {"ai:hard": 7, "ai:medium": 5, "ai:easy": 3, "anim:hard": 9, "anim:medium": 2, "animh:hard": 4}
fps_cycle = [60, 60, 50, 60, 48]
chosen = []
for key, k in quota.items():
    cands = pool.get(key, [])
    rng.shuffle(cands)
    chosen += cands[:k]
    print(f"{key}: {min(k, len(cands))}/{k} (available {len(cands)})")
lines = []
for i, (p720, p1080) in enumerate(chosen):
    fps = fps_cycle[i % len(fps_cycle)]
    lines += [json.dumps({"src": str(p720), "fps": fps}), json.dumps({"src": str(p1080), "fps": fps})]
out.write_text("\n".join(lines) + "\n")
print(f"{len(chosen)} prompt pairs -> {len(lines)} interpolation jobs -> {out}")
