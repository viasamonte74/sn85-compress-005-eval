#!/usr/bin/env python3
"""Cut 5-second training clips from commercially-licensed open video.

Sources (licence recorded per clip in the sidecar JSON / manifest):
  * Wikimedia Commons: only CC0, Public domain, CC BY, CC BY-SA files.
  * Netflix Open Content (CC BY 4.0): El Fuente (SDR y4m), Chimera / Meridian (HDR, tone-mapped).
  * Blender open movies (CC BY): Sintel.

Segments are pulled with ffmpeg HTTP seeking, so whole files are not downloaded
(except El Fuente y4m, which is streamed sequentially). Output matches the
dataset clip format: 1280x720 FFV1 yuv420p, native fps capped at 30, 5 s.
Resumable via the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = "sn85-dataset-builder/0.1 (video compression research; contact: none)"
API = "https://commons.wikimedia.org/w/api.php"

ALLOWED_LICENCE = re.compile(r"^(CC0( 1\.0)?|Public domain|PD.*|CC BY(-SA)? \d(\.\d)?( [A-Za-z]+)?)$", re.I)
BAD_TITLE = re.compile(
    r"SVS\d|visuali[sz]ation|simulation|animation|slideshow|lecture|interview|webinar|"
    r"presentation|trailer|\(19[0-6]\d\)|screencast|tutorial|podcast|speech|conference",
    re.I,
)

WIKI_QUERIES = [
    # hard-leaning content
    "waterfall 4K", "rain street night", "crowd festival", "fireworks 4K", "ocean waves rocks",
    "snowfall forest", "city traffic 4K", "drone forest 4K", "river rapids", "birds flock",
    "market street walking", "concert crowd", "wind grass field", "fire flames", "carnival parade",
    "train passing", "highway traffic", "storm sea", "leaves wind trees", "fountain water",
    # medium / easy content
    "drone landscape 4K", "city walk 4K", "lake mountains", "beach waves", "park people",
    "timelapse clouds", "boat harbour", "museum interior", "village street", "garden flowers",
]

NETFLIX = "http://download.opencontent.netflix.com.s3.amazonaws.com/"
NETFLIX_SRC = [
    {"name": "netflix_elfuente_boat", "url": NETFLIX + "ElFuente/Netflix_Boat_4096x2160_60fps_10bit_420.y4m",
     "duration": 5.0, "hdr": False, "y4m": True},
    {"name": "netflix_elfuente_foodmarket", "url": NETFLIX + "ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m",
     "duration": 10.0, "hdr": False, "y4m": True},
    {"name": "netflix_chimera", "url": NETFLIX + "Chimera/Chimera_DCI4k2398p_HDR_P3PQ.mp4", "hdr": True},
    {"name": "netflix_meridian", "url": NETFLIX + "Meridian/Meridian_UHD4k5994_HDR_P3PQ.mp4", "hdr": True},
]
BLENDER_SRC = [
    {"name": "blender_sintel", "url": "https://download.blender.org/demo/movies/Sintel.2010.1080p.mkv",
     "licence": "CC BY 3.0", "artist": "Blender Foundation"},
]

SDR_VF = "scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1280:720,format=yuv420p"
HDR_VF = (
    "zscale=tin=smpte2084:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100:p=bt709,format=gbrpf32le,"
    "tonemap=hable:desat=0,zscale=tin=linear:t=bt709:m=bt709:r=tv:pin=bt709:p=bt709,format=yuv420p," + SDR_VF
)

_lock = threading.Lock()
_fetch_lock = threading.Lock()


def wiki_transcode_url(url: str, height: int = 1080) -> str:
    """upload.wikimedia.org original URL -> its server-side VP9 transcode."""
    url = url.split("?")[0]
    head, _, rest = url.partition("/wikipedia/commons/")
    name = rest.rsplit("/", 1)[-1]
    return f"{head}/wikipedia/commons/transcoded/{rest}/{name}.{height}p.vp9.webm"


def fetch(url: str, dst: Path, retries: int = 3) -> bool:
    """Download one file at a time with backoff; Wikimedia rate-limits aggressively."""
    with _fetch_lock:
        for attempt in range(retries):
            r = subprocess.run(
                ["curl", "-sfL", "-A", UA, "--max-time", "1200", "-o", str(dst), "-w", "%{http_code}", url],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 100_000:
                time.sleep(2)
                return True
            code = r.stdout.strip()
            dst.unlink(missing_ok=True)
            if code == "404":
                return False
            time.sleep(30 * (attempt + 1))
    return False


def http_json(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def ua_args(url: str) -> list[str]:
    return ["-user_agent", UA] if url.startswith("http") else []


def ffprobe(url: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", *ua_args(url), "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", url],
        capture_output=True, text=True, timeout=120,
    )
    d = json.loads(out.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    num, den = (s.get("r_frame_rate") or "0/1").split("/")
    return {
        "width": s.get("width"), "height": s.get("height"),
        "fps": float(num) / float(den or 1), "duration": float(d.get("format", {}).get("duration") or 0),
    }


def search_wikimedia(per_query: int, min_height: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for q in WIKI_QUERIES:
        try:
            d = http_json({
                "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                "gsrsearch": f"filetype:video {q}", "gsrlimit": 50, "prop": "imageinfo",
                "iiprop": "url|size|mediatype|extmetadata",
                "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl",
            })
        except Exception as e:  # noqa: BLE001
            print(f"[wiki] query failed {q!r}: {e}", flush=True)
            continue
        n = 0
        for p in (d.get("query", {}).get("pages", {}) or {}).values():
            title = p.get("title", "")
            if title in seen or BAD_TITLE.search(title):
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            lic = (meta.get("LicenseShortName", {}) or {}).get("value", "").strip()
            if not ALLOWED_LICENCE.match(lic) or re.search(r"\bN[CD]\b", lic):
                continue
            dur = float(ii.get("duration") or 0)
            if (ii.get("height") or 0) < min_height or not (10 <= dur <= 3600):
                continue
            seen.add(title)
            artist = re.sub(r"<[^>]+>", "", (meta.get("Artist", {}) or {}).get("value", "")).strip()
            out.append({
                "source": "wikimedia", "title": title, "url": ii["url"], "page": ii.get("descriptionurl"),
                "licence": lic, "licence_url": (meta.get("LicenseUrl", {}) or {}).get("value"),
                "artist": artist[:200], "duration": dur, "width": ii.get("width"), "height": ii.get("height"),
                "query": q,
            })
            n += 1
            if n >= per_query:
                break
        time.sleep(0.5)
    return out


def segment_starts(duration: float, max_segs: int, seg: float) -> list[float]:
    if duration < seg + 2:
        return []
    n = int(max(1, min(max_segs, duration // 45)))
    lo, hi = duration * 0.1, duration * 0.9 - seg
    if hi <= lo:
        return [max(0.0, (duration - seg) / 2)]
    if n == 1:
        return [round((lo + hi) / 2, 2)]
    return [round(lo + i * (hi - lo) / (n - 1), 2) for i in range(n)]


def is_usable(path: Path) -> tuple[bool, str]:
    """Reject black, flat, frozen clips."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return True, "no-cv2"
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for idx in np.linspace(0, total - 1, 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype("float32"))
    cap.release()
    if len(frames) < 4:
        return False, "unreadable"
    mean = float(np.mean([f.mean() for f in frames]))
    std = float(np.mean([f.std() for f in frames]))
    motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(frames, frames[1:])]))
    if mean < 16:
        return False, f"dark({mean:.1f})"
    if std < 8:
        return False, f"flat({std:.1f})"
    if motion < 0.3:
        return False, f"frozen({motion:.2f})"
    return True, f"mean={mean:.0f} std={std:.0f} motion={motion:.1f}"


def cut(url: str, start: float, seg: float, fps: float, hdr: bool, out: Path, y4m: bool = False) -> bool:
    vf = HDR_VF if hdr else SDR_VF
    if fps > 30.5:
        vf = "fps=30," + vf
    tmp = out.with_suffix(".tmp.mkv")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *ua_args(url)]
    if y4m:
        cmd += ["-i", url, "-ss", f"{start}"]
    else:
        cmd += ["-ss", f"{start}", "-i", url]
    cmd += ["-t", f"{seg}", "-an", "-sn", "-vf", vf, "-c:v", "ffv1", "-level", "3", "-threads", "4", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 10000:
        tmp.unlink(missing_ok=True)
        print(f"  ffmpeg failed {out.name}: {r.stderr[-300:]}", flush=True)
        return False
    tmp.rename(out)
    return True


def process(job: dict, out_dir: Path, done: set[str], seg: float, max_segs: int) -> int:
    local: Path | None = None
    if job["source"] == "wikimedia":
        n_want = len(segment_starts(job.get("duration") or 0, max_segs, seg))
        if n_want == 0 or all(f"{job['name']}_{i:02d}" in done for i in range(n_want)):
            return 0
        local = out_dir / "_tmp" / f"{job['name']}.webm"
        local.parent.mkdir(exist_ok=True)
        src_url = ""
        for h in (1080, 720):
            src_url = wiki_transcode_url(job["url"], h)
            if fetch(src_url, local):
                break
        else:
            print(f"[skip] fetch failed {job['title']}", flush=True)
            return 0
        job = {**job, "fetched_from": src_url, "url_local": str(local)}
    try:
        return _process(job, out_dir, done, seg, max_segs)
    finally:
        if local is not None:
            local.unlink(missing_ok=True)


def _process(job: dict, out_dir: Path, done: set[str], seg: float, max_segs: int) -> int:
    src = job.get("url_local") or job["url"]
    try:
        info = ffprobe(src)
    except Exception as e:  # noqa: BLE001
        print(f"[skip] probe failed {job.get('title', job['name'])}: {e}", flush=True)
        return 0
    duration = job.get("duration") or info["duration"]
    fps = info["fps"] or 30.0
    if job.get("y4m"):
        starts = [i * seg for i in range(int(duration // seg))]
    else:
        starts = segment_starts(duration, max_segs, seg)
    made = 0
    for i, st in enumerate(starts):
        cid = f"{job['name']}_{i:02d}"
        if cid in done:
            continue
        out = out_dir / f"{cid}.mkv"
        if not cut(src, st, seg, fps, job.get("hdr", False), out, job.get("y4m", False)):
            continue
        ok, why = is_usable(out)
        if not ok:
            print(f"  reject {cid}: {why}", flush=True)
            out.unlink(missing_ok=True)
            with _lock:
                done.add(cid)
            continue
        meta = {
            "id": cid, "kind": "open", "source": job["source"], "title": job.get("title", job["name"]),
            "url": job["url"], "fetched_from": job.get("fetched_from", job["url"]), "page": job.get("page"), "licence": job["licence"],
            "licence_url": job.get("licence_url"), "artist": job.get("artist"), "start": st, "seconds": seg,
            "src_width": info["width"], "src_height": info["height"], "src_fps": round(fps, 3),
            "width": 1280, "height": 720, "fps": min(round(fps, 3), 30.0), "hdr_tonemapped": job.get("hdr", False),
            "quality_check": why, "path": str(out),
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        with _lock:
            with open(out_dir / "manifest.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            done.add(cid)
        made += 1
        print(f"  ok {cid} [{job['licence']}] {why}", flush=True)
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/dataset/open_clips")
    ap.add_argument("--per-query", type=int, default=8)
    ap.add_argument("--max-segs", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--min-height", type=int, default=1080)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sources", default="netflix,blender,wikimedia")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    man = out_dir / "manifest.jsonl"
    if man.exists():
        done |= {json.loads(l)["id"] for l in man.read_text().splitlines() if l.strip()}

    srcs = set(args.sources.split(","))
    jobs: list[dict] = []
    if "netflix" in srcs:
        for s in NETFLIX_SRC:
            jobs.append({**s, "source": "netflix_open_content", "licence": "CC BY 4.0",
                         "licence_url": "https://creativecommons.org/licenses/by/4.0/", "artist": "Netflix, Inc."})
    if "blender" in srcs:
        for s in BLENDER_SRC:
            jobs.append({**s, "source": "blender_open_movie",
                         "licence_url": "https://creativecommons.org/licenses/by/3.0/"})
    if "wikimedia" in srcs:
        wiki = search_wikimedia(args.per_query, args.min_height)
        print(f"[wiki] {len(wiki)} candidate files", flush=True)
        for w in wiki:
            jobs.append({**w, "name": "wm_" + hashlib.md5(w["title"].encode()).hexdigest()[:8]})
        (out_dir / "wikimedia_candidates.json").write_text(json.dumps(wiki, indent=1))

    print(f"{len(jobs)} source videos, {len(done)} clips already done", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        total = sum(ex.map(lambda j: process(j, out_dir, done, args.seconds, args.max_segs), jobs))
    print(f"done: {total} new clips -> {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
