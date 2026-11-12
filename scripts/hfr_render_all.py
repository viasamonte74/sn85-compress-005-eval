#!/usr/bin/env python3
"""Render the native high-frame-rate 3D clips (Blender Cycles, scripts/hfr_blender_scene.py) at
1280x720 and 1920x1080 (same scene/seed at both sizes) and store them as FFV1 yuv420p.

Output: /workspace/dataset/hfr_render/<id>[_1080].mkv + JSON sidecar + manifest line. Resumable.
  /venv/main/bin/python scripts/hfr_render_all.py [--only id,...] [--res 720p,1080p] [--device GPU]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = "/workspace/dataset-src/blender-4.5.15-linux-x64/blender"
BLENDER_LIBS = "/workspace/dataset-src/blender-libs/root/usr/lib/x86_64-linux-gnu"
FF = str(ROOT / "third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg")
LICENCE = "own procedural render (Blender 4.5 Cycles), no outside assets"

# id, scene, tier, fps, extra args
SCENES = [
    ("hfr_rnd_easy_01", "product", "easy", 60, ["--speed", "0.6"]),
    ("hfr_rnd_easy_02", "rings", "easy", 50, ["--speed", "0.4"]),
    ("hfr_rnd_easy_03", "objects", "easy", 60, ["--speed", "0.4"]),
    ("hfr_rnd_medium_01", "objects", "medium", 60, ["--speed", "1.2", "--motion-blur"]),
    ("hfr_rnd_medium_02", "terrain", "medium", 50, ["--speed", "0.8"]),
    ("hfr_rnd_medium_03", "city", "medium", 60, ["--speed", "1.0"]),
    ("hfr_rnd_medium_04", "swarm", "medium", 60, ["--speed", "0.8"]),
    ("hfr_rnd_hard_01", "city", "hard", 60, ["--speed", "1.8", "--night", "--motion-blur"]),
    ("hfr_rnd_hard_02", "ocean", "hard", 60, ["--speed", "1.5"]),
    ("hfr_rnd_hard_03", "swarm", "hard", 50, ["--speed", "1.8", "--density", "2.0", "--motion-blur"]),
    ("hfr_rnd_hard_04", "terrain", "hard", 60, ["--speed", "2.0"]),
    ("hfr_rnd_hard_05", "forest", "hard", 60, ["--speed", "1.5", "--density", "1.5"]),
]
# render set 2 (2026-11-10, user-approved "more 3D render clips"): 5 new scene types (interior, space, toon, snow,
# robots) x 2 + the 8 HFR scene types with new seeds/params, at standard rates (24/25/30 fps) -> /workspace/dataset/render2
SCENES_STD2 = [
    ("rnd2_easy_01", "interior", "easy", 24, ["--speed", "0.5"]),
    ("rnd2_medium_01", "interior", "medium", 30, ["--speed", "1.2", "--motion-blur"]),
    ("rnd2_medium_02", "space", "medium", 24, ["--speed", "0.8"]),
    ("rnd2_hard_01", "space", "hard", 30, ["--speed", "1.6", "--density", "1.8", "--motion-blur"]),
    ("rnd2_easy_02", "toon", "easy", 24, ["--speed", "0.6"]),
    ("rnd2_medium_03", "toon", "medium", 30, ["--speed", "1.4"]),
    ("rnd2_hard_02", "snow", "hard", 24, ["--speed", "0.8"]),
    ("rnd2_hard_03", "snow", "hard", 30, ["--speed", "1.3", "--density", "1.6", "--motion-blur"]),
    ("rnd2_medium_04", "robots", "medium", 24, ["--speed", "0.8"]),
    ("rnd2_hard_04", "robots", "hard", 30, ["--speed", "1.5", "--density", "1.6", "--motion-blur"]),
    ("rnd2_easy_03", "product", "easy", 30, ["--speed", "0.8"]),
    ("rnd2_medium_05", "city", "medium", 24, ["--speed", "0.7", "--night"]),
    ("rnd2_hard_05", "forest", "hard", 30, ["--speed", "1.2", "--density", "1.3", "--motion-blur"]),
    ("rnd2_medium_06", "ocean", "medium", 25, ["--speed", "0.9"]),
    ("rnd2_hard_06", "swarm", "hard", 24, ["--speed", "1.4", "--density", "1.5"]),
    ("rnd2_medium_07", "terrain", "medium", 30, ["--speed", "1.2"]),
]
SETS = {"hfr": (SCENES, "/workspace/dataset/hfr_render"), "std2": (SCENES_STD2, "/workspace/dataset/render2")}
RES = {"720p": (1280, 720, ""), "1080p": (1920, 1080, "_1080")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hfr", choices=sorted(SETS))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--res", default="720p,1080p")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--device", default="GPU")
    # 2026-11-10: rendering 720p separately cost 25-31 % of each scene's GPU time (render2: 441-459 s of
    # 1475-1784 s per pair) for the same scene/seed; the 720p twin is now a Lanczos downscale of the 1080p
    # render's RGB frames (no extra render, exact same content, slightly cleaner: 2.25 rendered px per px).
    ap.add_argument("--derive-720p", choices=["auto", "yes", "no"], default="auto",
                    help="render 1080p only and derive the 720p clip by downscaling (auto: on for the std2 set)")
    args = ap.parse_args()
    scenes, default_dir = SETS[args.set]
    out_dir = Path(args.out_dir or default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    env = dict(os.environ, LD_LIBRARY_PATH=f"{BLENDER_LIBS}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    res_list = args.res.split(",")
    derive = (args.derive_720p == "yes" or (args.derive_720p == "auto" and args.set == "std2")) \
        and "1080p" in res_list and "720p" in res_list

    def encode(src_args: list[str], vf: str, fps: int, out: Path) -> None:
        tmp = out.with_suffix(".tmp.mkv")
        subprocess.run([FF, "-y", "-v", "error", *src_args, "-vf", vf, "-c:v", "ffv1", "-level", "3",
                        "-threads", "8", "-r", str(fps), str(tmp)], check=True)
        tmp.rename(out)

    for sid, scene, tier, fps, extra in scenes:
        if only and sid not in only:
            continue
        seed = int.from_bytes(sid.encode(), "little") % (1 << 31)
        want = int(round(args.seconds * fps))
        for res in (["1080p"] if derive else res_list):
            w, h, suffix = RES[res]
            cid = sid + suffix
            out = out_dir / f"{cid}.mkv"
            # outputs of this render: (clip id, path, width, height, derived by downscale)
            outs = [(cid, out, w, h, False)]
            if derive:
                outs.append((sid, out_dir / f"{sid}.mkv", 1280, 720, True))
            todo = [o for o in outs if not o[1].exists()]
            if not todo:
                continue
            t0 = time.time()
            frames_dir = out_dir / f".frames_{cid}"
            if out.exists():
                # 1080p already there, only the 720p twin is missing: downscale the stored clip
                src_args = ["-i", str(out)]
            else:
                shutil.rmtree(frames_dir, ignore_errors=True)
                cmd = [BLENDER, "-b", "--factory-startup", "-P", str(ROOT / "scripts/hfr_blender_scene.py"), "--",
                       "--scene", scene, "--fps", str(fps), "--width", str(w), "--height", str(h),
                       "--seconds", str(args.seconds), "--samples", str(args.samples), "--seed", str(seed),
                       "--device", args.device, "--out", str(frames_dir), *extra]
                r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                pngs = sorted(frames_dir.glob("f_*.png"))
                if r.returncode != 0 or len(pngs) != want:
                    print(f"{cid}: blender failed rc={r.returncode} frames={len(pngs)}/{want}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
                    continue
                src_args = ["-framerate", str(fps), "-start_number", "1", "-i", str(frames_dir / "f_%04d.png")]
            render_s = round(time.time() - t0, 1)
            for ocid, opath, ow, oh, derived in todo:
                vf = (f"scale={ow}:{oh}:flags=lanczos," if (ow, oh) != (w, h) else "") + "format=yuv420p"
                encode(src_args, vf, fps, opath)
                meta = {"id": ocid, "kind": "render", "intended_tier": tier, "scene": scene, "width": ow, "height": oh,
                        "fps": float(fps), "frames": want, "samples": args.samples, "denoise": True, "args": extra,
                        "seed": seed, "native_hfr": fps > 30, "blender": "4.5.15 LTS",
                        "source": "scripts/hfr_blender_scene.py", "licence": LICENCE, "path": str(opath),
                        "rendered_at": f"{w}x{h}", "downscaled": "lanczos from the RGB render frames" if derived else None,
                        "seconds": render_s}
                (out_dir / f"{ocid}.json").write_text(json.dumps(meta, indent=1))
                with open(out_dir / "manifest.jsonl", "a") as f:
                    f.write(json.dumps(meta) + "\n")
                print(f"{ocid}: {scene} {ow}x{oh}@{fps} {want} frames (rendered {w}x{h}{', downscaled' if derived else ''}), "
                      f"render {render_s}s", flush=True)
            shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
