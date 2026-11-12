#!/usr/bin/env python3
"""Cut 5-second 3D-render training clips from CC BY Blender open movies (2026-11-10).

Sources (CC BY only, same licence policy as download_open_clips.py; licence + attribution in each sidecar):
  * Big Buck Bunny, 3840x2160 60 fps master (CC BY 3.0, Blender Foundation): 60 fps segments, and 30 fps
    segments made by dropping every other frame (different segments, never the same shot twice).
  * Caminandes 1/2/3, 1920x1080 24 fps (CC BY 3.0, Pablo Vazquez / Blender Foundation).
  * Cosmos Laundromat: First Cycle, 1920x804 24 fps letterboxed (CC BY 4.0, Blender Foundation): centre-cropped
    to 16:9 and stored at 720p only (a 1080p version would be an upscale).

Every segment is stored at 1280x720 and (except letterboxed sources) 1920x1080, FFV1 yuv420p, same clip id with
"_1080" for the large one, so both sizes share a family (and a split). Segments avoid the opening titles and end
credits; black / flat / frozen segments are rejected with download_open_clips.is_usable.

Output: /workspace/dataset/render_film/<id>[_1080].mkv + JSON sidecar + manifest line. Resumable.
  /venv/main/bin/python scripts/render_film_clips.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_open_clips import is_usable  # noqa: E402

FF = str(ROOT / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg")
FP = str(ROOT / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffprobe")
SRC_DIR = Path("/workspace/dataset-src/films")
UA = "sn85-dataset-builder/0.1 (video compression research)"

# name, url, licence, licence_url, artist, title, segments: list of output fps (None = native), 1080p allowed,
# crop to 16:9 (letterboxed source), window (fraction of duration) that excludes titles and credits
FILMS = [
    {"name": "film_bbb", "title": "Big Buck Bunny (2160p 60 fps master)",
     "url": "https://download.blender.org/demo/movies/BBB/bbb_sunflower_2160p_60fps_normal.mp4.zip",
     "licence": "CC BY 3.0", "licence_url": "https://creativecommons.org/licenses/by/3.0/",
     "artist": "Blender Foundation", "fps_plan": [60, 30] * 5, "hd": True, "crop": False, "window": (0.04, 0.86)},
    {"name": "film_caminandes1", "title": "Caminandes 1: Llama Drama",
     "url": "https://archive.org/download/Caminandes1LlamaDrama/01_llama_drama_1080p.mp4",
     "licence": "CC BY 3.0", "licence_url": "https://creativecommons.org/licenses/by/3.0/",
     "artist": "Pablo Vazquez", "fps_plan": [None] * 2, "hd": True, "crop": False, "window": (0.08, 0.80)},
    {"name": "film_caminandes2", "title": "Caminandes 2: Gran Dillama",
     "url": "https://archive.org/download/Caminandes2GranDillama/02_gran_dillama_1080p.mp4",
     "licence": "CC BY 3.0", "licence_url": "https://creativecommons.org/licenses/by/3.0/",
     "artist": "Pablo Vazquez", "fps_plan": [None] * 3, "hd": True, "crop": False, "window": (0.06, 0.82)},
    {"name": "film_caminandes3", "title": "Caminandes 3: Llamigos",
     "url": "https://archive.org/download/CaminandesLlamigos/Caminandes_%20Llamigos-1080p.mp4",
     "licence": "CC BY 3.0", "licence_url": "https://creativecommons.org/licenses/by/3.0/",
     "artist": "Blender Foundation", "fps_plan": [None] * 3, "hd": True, "crop": False, "window": (0.06, 0.82)},
    {"name": "film_cosmos", "title": "Cosmos Laundromat: First Cycle",
     "url": "https://archive.org/download/CosmosLaundromatFirstCycle/Cosmos%20Laundromat%20-%20First%20Cycle%20(1080p).mp4",
     "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
     "artist": "Blender Foundation", "fps_plan": [None] * 4, "hd": False, "crop": True, "window": (0.05, 0.82)},
]


def fetch(f: dict) -> Path:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    fname = urllib.request.unquote(f["url"].rsplit("/", 1)[1])
    dst = SRC_DIR / fname
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        print(f"download {f['url']}", flush=True)
        subprocess.run(["curl", "-sSL", "--retry", "5", "-A", UA, "-o", str(tmp), f["url"]], check=True)
        tmp.rename(dst)
    if dst.suffix == ".zip":
        with zipfile.ZipFile(dst) as z:
            member = next(n for n in z.namelist() if n.lower().endswith((".mp4", ".mkv", ".mov")))
            out = SRC_DIR / Path(member).name
            if not out.exists():
                z.extract(member, SRC_DIR / "_unz")
                (SRC_DIR / "_unz" / member).rename(out)
        return out
    return dst


def probe(p: Path) -> dict:
    o = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(p)],
                       capture_output=True, text=True, check=True).stdout
    d = json.loads(o)
    s = d["streams"][0]
    n, m = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": int(n) / int(m), "duration": float(d["format"]["duration"])}


def cut(src: Path, start: float, seg: float, out_fps: float, src_fps: float, w: int, h: int, crop: bool, out: Path) -> bool:
    vf = []
    if abs(out_fps - src_fps) > 0.01:
        vf.append(f"fps={out_fps:g}")
    if crop:  # letterboxed: crop the picture to 16:9 around the centre
        vf.append("crop=trunc(ih*16/9/2)*2:ih")
    vf += [f"scale={w}:{h}:flags=lanczos", "format=yuv420p"]
    tmp = out.with_suffix(".tmp.mkv")
    r = subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{seg}",
                        "-an", "-sn", "-vf", ",".join(vf), "-r", f"{out_fps:g}", "-c:v", "ffv1", "-level", "3",
                        "-threads", "8", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 10000:
        tmp.unlink(missing_ok=True)
        print(f"  ffmpeg failed {out.name}: {r.stderr[-300:]}", flush=True)
        return False
    tmp.rename(out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/dataset/render_film")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    seg = args.seconds
    made = 0
    for f in FILMS:
        if only and f["name"] not in only:
            continue
        src = fetch(f)
        info = probe(src)
        lo, hi = info["duration"] * f["window"][0], info["duration"] * f["window"][1] - seg
        n = len(f["fps_plan"])
        starts = [round(lo + i * (hi - lo) / max(1, n - 1), 2) for i in range(n)]
        for i, (st, fp) in enumerate(zip(starts, f["fps_plan"])):
            out_fps = float(fp or round(info["fps"], 3))
            cid = f"{f['name']}_{int(round(out_fps))}_{i:02d}"
            sizes = [(1280, 720, "")] + ([(1920, 1080, "_1080")] if f["hd"] else [])
            for w, h, suf in sizes:
                out = out_dir / f"{cid}{suf}.mkv"
                if out.exists():
                    continue
                if not cut(src, st, seg, out_fps, info["fps"], w, h, f["crop"], out):
                    continue
                ok, why = is_usable(out)
                if not ok:
                    print(f"  reject {out.name}: {why}", flush=True)
                    out.unlink(missing_ok=True)
                    continue
                meta = {
                    "id": cid + suf, "kind": "render", "source": "blender_open_movie", "title": f["title"],
                    "url": f["url"], "licence": f["licence"], "licence_url": f["licence_url"], "artist": f["artist"],
                    "start": st, "seconds": seg, "src_width": info["width"], "src_height": info["height"],
                    "src_fps": round(info["fps"], 3), "width": w, "height": h, "fps": out_fps,
                    "cropped_16_9": f["crop"], "frame_drop_from": round(info["fps"], 3) if fp and abs(out_fps - info["fps"]) > 0.01 else None,
                    "quality_check": why, "path": str(out),
                }
                out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
                with open(out_dir / "manifest.jsonl", "a") as mf:
                    mf.write(json.dumps(meta) + "\n")
                made += 1
                print(f"  ok {out.name} {w}x{h} {out_fps:g} fps start {st:.1f}s [{f['licence']}] {why}", flush=True)
    print(f"done: {made} new clips, {len(list(out_dir.glob('*.mkv')))} in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
