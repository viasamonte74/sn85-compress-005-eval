#!/usr/bin/env python3
"""Real high-frame-rate footage (>= 47 fps) cut at its NATIVE frame rate, at 1920x1080 and 1280x720.

Sources (commercially usable, licence recorded per clip):
  * Netflix Open Content (CC BY 4.0): Chimera 59.94p HDR, Meridian 59.94p HDR (tone-mapped to SDR
    BT.709 like scripts/download_open_clips.py), El Fuente Boat / FoodMarket 60p SDR (10-bit y4m).
  * Wikimedia Commons CC0 / PD / CC BY / CC BY-SA uploads whose stream is >= 47 fps.

Each 5 s segment is cut once from the remote source to 1080p FFV1 (lanczos, centre crop), the 720p
version is a lanczos downscale of that 1080p cut (one network read per segment). Same quality filter
as the 30 fps open clips (rejects black / flat / frozen). Ids: hfr_<name>_<ii>[_1080]. Resumable.

  /venv/main/bin/python scripts/hfr_open_clips.py [--sources netflix,wikimedia] [--max-wiki 12]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_open_clips as oc  # noqa: E402

FF = str(Path(__file__).resolve().parents[1] / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg")
FP = str(Path(__file__).resolve().parents[1] / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffprobe")
MIN_FPS = 47.0

NETFLIX = oc.NETFLIX
SOURCES = [
    {"name": "nfx_chimera60", "url": NETFLIX + "Chimera/Chimera_DCI4k5994p_HDR_P3PQ.mp4", "hdr": True, "segs": 12},
    {"name": "nfx_meridian60", "url": NETFLIX + "Meridian/Meridian_UHD4k5994_HDR_P3PQ.mp4", "hdr": True, "segs": 4},
    {"name": "nfx_boat60", "url": NETFLIX + "ElFuente/Netflix_Boat_4096x2160_60fps_10bit_420.y4m", "y4m": True,
     "duration": 5.0, "hdr": False},
    {"name": "nfx_foodmarket60", "url": NETFLIX + "ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m",
     "y4m": True, "duration": 10.0, "hdr": False},
]
WIKI_QUERIES_HFR = [
    "60fps", "60 fps", "50fps", "1080p60", "4K 60fps", "drone 60fps", "gameplay 60fps", "60fps city",
    "60fps nature", "60fps sport", "60fps train", "60fps waterfall", "50 fps", "2160p60",
]


def vf(w: int, h: int, hdr: bool) -> str:
    sdr = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},format=yuv420p"
    if not hdr:
        return sdr
    return ("zscale=tin=smpte2084:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100:p=bt709,format=gbrpf32le,"
            "tonemap=hable:desat=0,zscale=tin=linear:t=bt709:m=bt709:r=tv:pin=bt709:p=bt709,format=yuv420p," + sdr)


def probe(url: str) -> dict:
    r = subprocess.run([FP, "-v", "error", *oc.ua_args(url), "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,avg_frame_rate:format=duration", "-of", "json", url],
                       capture_output=True, text=True, timeout=120)
    d = json.loads(r.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    def fr(x):
        n, _, m = (x or "0/1").partition("/")
        return float(n) / float(m or 1) if float(m or 1) else 0.0
    return {"width": s.get("width") or 0, "height": s.get("height") or 0, "rate": s.get("r_frame_rate"),
            "fps": fr(s.get("r_frame_rate")), "avg_fps": fr(s.get("avg_frame_rate")),
            "duration": float((d.get("format") or {}).get("duration") or 0)}


def cut_pair(src: str, start: float, seg: float, rate: str, hdr: bool, y4m: bool, out1080: Path, out720: Path) -> bool:
    tmp = out1080.with_suffix(".tmp.mkv")
    cmd = [FF, "-y", "-v", "error", *oc.ua_args(src)]
    cmd += (["-i", src, "-ss", f"{start}"] if y4m else ["-ss", f"{start}", "-i", src])
    cmd += ["-t", f"{seg}", "-an", "-sn", "-vf", vf(1920, 1080, hdr), "-r", rate, "-c:v", "ffv1", "-level", "3",
            "-threads", "8", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 100000:
        tmp.unlink(missing_ok=True)
        print(f"  cut failed {out1080.name}: {r.stderr[-300:]}", flush=True)
        return False
    tmp.rename(out1080)
    tmp7 = out720.with_suffix(".tmp.mkv")
    subprocess.run([FF, "-y", "-v", "error", "-i", str(out1080), "-vf", "scale=1280:720:flags=lanczos,format=yuv420p",
                    "-c:v", "ffv1", "-level", "3", "-threads", "8", str(tmp7)], check=True)
    tmp7.rename(out720)
    return True


def write_meta(out_dir: Path, path: Path, job: dict, info: dict, start: float, seg: float, why: str, w: int, h: int) -> None:
    pr = probe(str(path))
    meta = {"id": path.stem, "kind": "open", "source": job["source"], "title": job.get("title", job["name"]),
            "url": job["url"], "fetched_from": job.get("fetched_from", job["url"]), "page": job.get("page"),
            "licence": job["licence"], "licence_url": job.get("licence_url"), "artist": job.get("artist"),
            "start": start, "seconds": seg, "src_width": info["width"], "src_height": info["height"],
            "src_fps": round(info["fps"], 3), "width": w, "height": h, "fps": round(pr["fps"], 3),
            "rate": pr["rate"], "hdr_tonemapped": job.get("hdr", False), "native_hfr": True,
            "quality_check": why, "path": str(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    with open(out_dir / "manifest.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")


def process(job: dict, out_dir: Path, seg: float) -> int:
    src = job.get("url_local") or job["url"]
    info = probe(src)
    if info["fps"] < MIN_FPS or info["height"] < 1080:
        print(f"[skip] {job['name']}: {info['width']}x{info['height']} @ {info['fps']:.2f}", flush=True)
        return 0
    duration = job.get("duration") or info["duration"]
    if job.get("y4m"):
        starts = [i * seg for i in range(int(duration // seg))]
    else:
        n = job.get("segs", 2)
        lo, hi = duration * 0.08, duration * 0.92 - seg
        starts = [round(lo + i * (hi - lo) / max(1, n - 1), 2) for i in range(n)] if hi > lo else [0.0]
    made = 0
    for i, st in enumerate(starts):
        cid = f"hfr_{job['name']}_{i:02d}"
        o1080, o720 = out_dir / f"{cid}_1080.mkv", out_dir / f"{cid}.mkv"
        if o1080.exists() and o720.exists():
            continue
        if not cut_pair(src, st, seg, info["rate"], job.get("hdr", False), job.get("y4m", False), o1080, o720):
            continue
        ok, why = oc.is_usable(o720)
        if not ok:
            print(f"  reject {cid}: {why}", flush=True)
            o1080.unlink(missing_ok=True)
            o720.unlink(missing_ok=True)
            continue
        write_meta(out_dir, o1080, job, info, st, seg, why, 1920, 1080)
        write_meta(out_dir, o720, job, info, st, seg, why, 1280, 720)
        made += 1
        print(f"  ok {cid} (+_1080) @ {info['rate']} [{job['licence']}] {why}", flush=True)
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/dataset/hfr_open")
    ap.add_argument("--sources", default="netflix,wikimedia")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--max-wiki", type=int, default=12, help="max Wikimedia videos (2 segments each)")
    ap.add_argument("--dry-run", action="store_true", help="list candidate sources with probed fps only")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = set(args.sources.split(","))
    total = 0
    if "netflix" in srcs:
        for s in SOURCES:
            job = {**s, "source": "netflix_open_content", "licence": "CC BY 4.0",
                   "licence_url": "https://creativecommons.org/licenses/by/4.0/", "artist": "Netflix, Inc."}
            if args.dry_run:
                print(job["name"], probe(job["url"]), flush=True)
                continue
            total += process(job, out_dir, args.seconds)
    if "wikimedia" in srcs:
        import time as _t
        cands, seen = [], set()
        for q in WIKI_QUERIES_HFR:  # one query at a time, slowly: Commons rate-limits (HTTP 429)
            oc.WIKI_QUERIES[:] = [q]
            for c in oc.search_wikimedia(per_query=10, min_height=1080):
                if c["title"] not in seen:
                    seen.add(c["title"])
                    cands.append(c)
            _t.sleep(8)
        picked = 0
        for c in cands:
            if picked >= args.max_wiki:
                break
            _t.sleep(2)
            info = probe(c["url"])
            if info["fps"] < MIN_FPS:
                continue
            name = "wm60_" + "".join(ch for ch in c["title"].split(":", 1)[-1].rsplit(".", 1)[0].lower() if ch.isalnum())[:24]
            if args.dry_run:
                print(f"{name}: {info['width']}x{info['height']} @ {info['fps']:.2f} {c['licence']} {c['duration']:.0f}s", flush=True)
                picked += 1
                continue
            local = out_dir / "_tmp" / f"{name}.webm"
            local.parent.mkdir(exist_ok=True)
            fetched = ""
            for h in (1080,):
                url = oc.wiki_transcode_url(c["url"], h)
                if oc.fetch(url, local) and probe(str(local))["fps"] >= MIN_FPS:
                    fetched = url
                    break
            if not fetched:  # transcode missing or frame-rate reduced: use the original upload
                if oc.fetch(c["url"], local):
                    fetched = c["url"]
            if not fetched:
                continue
            job = {**c, "name": name, "url_local": str(local), "fetched_from": fetched, "segs": 2}
            try:
                total += process(job, out_dir, args.seconds)
                picked += 1
            finally:
                local.unlink(missing_ok=True)
    print(f"HFR open clips: {total} new segment pairs", flush=True)


if __name__ == "__main__":
    main()
