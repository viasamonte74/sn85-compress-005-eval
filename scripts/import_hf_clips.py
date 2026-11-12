#!/usr/bin/env python3
"""Turn two Hugging Face datasets into 5-second training clips (2026-11-11, user-requested).

* nickossaf/Screen-Content-Video (CC BY 2.0; Safonov, Rakhmanov, Vatolin, "Screen Content Video Dataset and
  Benchmark", ACM MM 2025): the 100 reference recordings (`<n>/orig.mp4`, 1535x1030 30 fps 10 s, H.264 High 4:4:4,
  QP 3-19, not lossless). Centre-cropped to 16:9 (1534x862) and Lanczos-downscaled to 1280x720 4:2:0 (a 1080p
  version would be an upscale). One 5 s segment per recording: the half (0-5 s or 5-10 s) with more motion.
  -> /workspace/dataset/screen_real/scrr_<NNN>.mkv
* 0xmsu/sn85-blender-removal-pairs (Blender/Cycles, Poly Haven CC0 assets; the upload grants no licence for the
  rendered outputs, so these stay local and are never redistributed): the 30 `.input.mkv` renders only (the
  `.reference.mkv` twin is the same shot minus one object). Already 1280x720 24 fps FFV1 4-5 s: re-wrapped as is.
  -> /workspace/dataset/render_hf/bpr_<name>.mkv

FFV1 yuv420p + JSON sidecar + manifest line per clip; black / flat / frozen clips rejected (download_open_clips.is_usable).
  /venv/main/bin/python scripts/import_hf_clips.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_open_clips import is_usable  # noqa: E402

FF = str(ROOT / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg")
SRC = Path("/workspace/dataset-src")
SCREEN_URL = "https://huggingface.co/datasets/nickossaf/Screen-Content-Video"
BLENDER_URL = "https://huggingface.co/datasets/0xmsu/sn85-blender-removal-pairs"


def half_motion(p: Path) -> tuple[float, float]:
    """Mean absolute luma frame difference (0-255) of the first and second 5 s."""
    o = subprocess.run([FF, "-v", "error", "-i", str(p), "-vf",
                        "scale=320:-2,format=gray,tblend=all_mode=difference,signalstats,"
                        "metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
                       capture_output=True, text=True).stdout
    v = [float(x) for x in re.findall(r"YAVG=([\d.]+)", o)]
    h = len(v) // 2
    return (sum(v[:h]) / max(h, 1), sum(v[h:]) / max(len(v) - h, 1))


def write(meta: dict, out: Path) -> None:
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    with open(out.parent / "manifest.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")


def screen() -> int:
    # 2026-11-11: the recordings are tagged BT.601 (smpte170m); the hidden set is BT.709 and the validator never
    # overrides a stated tag, so 601-tagged references distorted the labels (scrr_003: VMAF +0.42, NEG gap -1.40).
    # Convert the matrix 601 -> 709 and tag BT.709; written to screen_real709 (the 601 originals stay in screen_real).
    out_dir = Path("/workspace/dataset/screen_real709")
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for src in sorted((SRC / "hf_screen_content").glob("*/orig.mp4"), key=lambda p: int(p.parent.name)):
        n = int(src.parent.name)
        out = out_dir / f"scrr_{n:03d}.mkv"
        if out.exists():
            continue
        m1, m2 = half_motion(src)
        start = 0.0 if m1 >= m2 else 5.0
        tmp = out.with_suffix(".tmp.mkv")
        r = subprocess.run([FF, "-y", "-v", "error", "-ss", f"{start}", "-i", str(src), "-t", "5", "-an", "-sn",
                            "-vf", "crop=1534:862,scale=1280:720:flags=lanczos:in_color_matrix=bt601:out_color_matrix=bt709"
                            ":in_range=tv:out_range=tv,format=yuv420p", "-r", "30", "-color_range", "tv",
                            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-c:v", "ffv1", "-level", "3", "-threads", "8", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ffmpeg failed {src}: {r.stderr[-200:]}", flush=True)
            tmp.unlink(missing_ok=True)
            continue
        tmp.rename(out)
        ok, why = is_usable(out)
        # static and dark-mode UIs are normal screen content (the frozen / dark rules are for natural video):
        # keep them, reject only blank (flat) or unreadable recordings
        if not ok and why.startswith(("frozen", "dark")):
            ok, why = True, f"kept {why}"
        if not ok:
            print(f"  reject scrr_{n:03d}: {why}", flush=True)
            out.unlink()
            continue
        write({"id": out.stem, "kind": "screen", "source": "hf:nickossaf/Screen-Content-Video", "url": SCREEN_URL,
               "src_file": f"{n}/orig.mp4", "licence": "CC BY 2.0", "licence_url": "https://creativecommons.org/licenses/by/2.0/",
               "attribution": "Safonov, Rakhmanov, Vatolin, Screen Content Video Dataset and Benchmark, ACM MM 2025",
               "src_width": 1535, "src_height": 1030, "src_codec": "h264 High 4:4:4 (QP 3-19)", "width": 1280,
               "height": 720, "fps": 30.0, "start": start, "seconds": 5.0, "motion_halves": [round(m1, 3), round(m2, 3)],
               "processing": "crop 1534x862 centre, lanczos 1280x720, BT.601 -> BT.709 matrix, yuv420p, tagged BT.709", "quality_check": why,
               "path": str(out)}, out)
        made += 1
        print(f"  ok {out.name} start {start:.0f}s motion {max(m1, m2):.2f} {why}", flush=True)
    return made


def blender() -> int:
    out_dir = Path("/workspace/dataset/render_hf")
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for src in sorted((SRC / "hf_blender_pairs").glob("*.input.mkv")):
        name = src.name.removesuffix(".input.mkv")
        out = out_dir / f"bpr_{name}.mkv"
        if out.exists():
            continue
        shutil.copyfile(src, out)
        ok, why = is_usable(out)
        if not ok:
            print(f"  reject {out.stem}: {why}", flush=True)
            out.unlink()
            continue
        sm = json.loads((src.parent / f"{name}.meta.json").read_text())
        write({"id": out.stem, "kind": "render", "source": "hf:0xmsu/sn85-blender-removal-pairs", "url": BLENDER_URL,
               "src_file": src.name, "licence": "assets CC0 (Poly Haven); no licence granted for the rendered outputs: "
               "local training use only, never redistribute", "width": 1280, "height": 720, "fps": 24.0,
               "render": sm.get("render"), "scene": sm.get("scene"), "quality_check": why, "path": str(out)}, out)
        made += 1
        print(f"  ok {out.name} {why}", flush=True)
    return made


if __name__ == "__main__":
    print(f"screen: {screen()} new", flush=True)  # blender pairs: not used (no licence for the renders)
