#!/usr/bin/env python3
"""Package the val16 evaluation set the way the validator hands clips to a contender (2026-11-13, user: "prepare 16
clips for eval just like validators do"). For every generated clip in /workspace/dataset/val16:
  * check: FFV1, yuv420p, 1280x720 or 1920x1080, 24-60 fps, 5-10 s; the 16 clips total 3428 frames;
  * re-encode losslessly (FFV1) with the practice pack's tags (BT.709 matrix, limited range) and prove the decoded
    frames are bit-identical to the generated clip;
  * name it by its sha256, no extension, and place it in one of 4 batch folders of 4 (seeded shuffle);
  * write manifest.json (outside the batch folders) with the practice pack's meta fields + budget at 0.05 bpp,
    and data/e2e_val16.txt (clip paths in batch order, for scripts/e2e_batched.py).
Output: /workspace/dataset/val16_validator/{batch1..4}/<sha256>, /workspace/dataset/val16_validator/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FF = str(ROOT / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg")
FP = str(ROOT / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffprobe")
import argparse as _ap
_a = _ap.ArgumentParser(); _a.add_argument("--name", default="val16"); _a.add_argument("--prefix", default="v16"); _a.add_argument("--seed", type=int, default=20261113)
_A, _ = _a.parse_known_args()
SRC = Path(f"/workspace/dataset/{_A.name}")
TAGGED = Path(f"/workspace/dataset/{_A.name}_tagged")
OUT = Path(f"/workspace/dataset/{_A.name}_validator")


def probe(p: Path) -> dict:
    j = json.loads(subprocess.run([FP, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                   "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames,color_range,color_space",
                                   "-show_entries", "format=format_name", "-of", "json", str(p)],
                                  capture_output=True, text=True, check=True).stdout)
    s = j["streams"][0]
    return {**s, "fps": Fraction(s["r_frame_rate"]), "frames": int(s["nb_read_frames"]), "format": j["format"]["format_name"]}


def frame_md5(p: Path) -> list[str]:
    r = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-i", str(p), "-map", "0:v:0", "-f", "framemd5", "-"],
                       capture_output=True, text=True, check=True).stdout
    return [l.rsplit(",", 1)[1].strip() for l in r.splitlines() if l and not l.startswith("#")]


def main() -> int:
    clips = sorted(p for p in SRC.glob(f"{_A.prefix}*.mkv") if ".tmp" not in p.name)
    assert len(clips) == 16, f"{len(clips)} clips in {SRC}"
    TAGGED.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for p in clips:
        m = probe(p)
        sec = float(m["frames"] / m["fps"])
        if m["codec_name"] != "ffv1" or m["pix_fmt"] != "yuv420p":
            problems.append(f"{p.name}: {m['codec_name']} {m['pix_fmt']}")
        if (m["width"], m["height"]) not in ((1280, 720), (1920, 1080)) or not (24 <= m["fps"] <= 60) or not (4.99 <= sec <= 10.01):
            problems.append(f"{p.name}: {m['width']}x{m['height']} {float(m['fps'])} fps {sec:.3f} s")
        t = TAGGED / p.name
        # setparams only relabels the frames: with -colorspace alone ffmpeg 8 auto-inserts a scale filter that CONVERTS
        # unknown -> BT.709 through RGB (every pixel changes; caught by the frame-md5 check below on 2026-11-13)
        subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", str(p), "-map", "0:v:0",
                        "-vf", "setparams=colorspace=bt709:range=tv", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
                        "-colorspace", "bt709", "-color_range", "tv", "-f", "matroska", str(t)], check=True)
        mt = probe(t)
        same = frame_md5(p) == frame_md5(t)
        if not same or mt["color_space"] != "bt709" or mt["color_range"] != "tv" or mt["frames"] != m["frames"]:
            problems.append(f"{p.name}: tagged copy differs (same pixels {same}, {mt['color_range']}/{mt['color_space']})")
        side = p.with_suffix(".json")
        info = json.loads(side.read_text()) if side.exists() else {}
        sha = hashlib.sha256(t.read_bytes()).hexdigest()
        kind = "AI-generated" if "ai_" in p.name else "rendered" if "rnd_" in p.name else "screen"
        rows.append({"name": p.stem, "sha256": sha, "bytes": t.stat().st_size, "kind": kind,
                     "content": info.get("prompt") or info.get("scene") or "", "tier": info.get("intended_tier") or info.get("tier"),
                     "width": m["width"], "height": m["height"], "fps": m["r_frame_rate"], "frames": m["frames"],
                     "seconds": round(sec, 3), "budget_kbps": round(0.05 * m["width"] * m["height"] * float(m["fps"]) / 1000, 1),
                     "path": str(t)})
    total = sum(r["frames"] for r in rows)
    if total != 3428:
        problems.append(f"total frames {total} != 3428")
    random.Random(_A.seed).shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for b in range(4):
        d = OUT / f"batch{b + 1}"
        d.mkdir(exist_ok=True)
        for r in rows[4 * b:4 * b + 4]:
            r["batch"] = b + 1
            dst = d / r["sha256"]
            if not dst.exists():
                dst.hardlink_to(r["path"])
    (OUT / "manifest.json").write_text(json.dumps({"clips": rows, "total_frames": total, "batches": 4, "batch_size": 4,
                                                    "bits_per_pixel": 0.05, "problems": problems}, indent=1))
    (ROOT / f"data/e2e_{_A.name}.txt").write_text(f"# {_A.name}: 16 new evaluation-only clips, validator-style batches in order (scripts/package_val16.py)\n"
                                              + "\n".join(r["path"] for r in rows) + "\n")
    for r in rows:
        print(f"batch{r['batch']} {r['sha256'][:12]} {r['name']:20s} {r['kind']:12s} {r['width']}x{r['height']} {r['fps']:>5s} "
              f"{r['frames']:3d} fr {r['seconds']:6.3f} s budget {r['budget_kbps']} kbps")
    print(f"total frames {total}; problems: {problems or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
