#!/usr/bin/env python3
"""High-frame-rate AI clips: RIFE v4.26 frame interpolation of 24 fps Wan clips to 48/50/60 fps.

Each job takes one source FFV1 clip (720p or 1080p, 24 fps, 121 frames = 5.0 s) and writes a clip of
the same duration at ``fps`` (60 -> 301 frames, 50 -> 251, 48 -> 241). Output frame k sits at source
position p = k * src_fps / fps; integer positions copy the source frame, the rest are RIFE-synthesised
from the two neighbouring source frames at timestep frac(p) (RIFE v4 takes arbitrary timesteps).
Same pixel path as the Wan writer: decode to rgb24, encode FFV1 yuv420p (ffmpeg default matrix).

Ids: <source id>_i<fps>, with the _1080 suffix kept last (ai_hard_05_i60_1080) so ``clip_family``
groups the clip with its source. Resumable (existing outputs are skipped).

Run with the Wan env (torch + CUDA):
  /workspace/dataset-src/wan-env/bin/python scripts/hfr_interpolate.py --plan data/hfr_interp_plan.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

RIFE_ROOT = Path("/workspace/dataset-src/Practical-RIFE")
FF = "/workspace/sn85-compression-005/third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg"
FP = "/workspace/sn85-compression-005/third_party/ffmpeg-n8.1-linux64-gpl/bin/ffprobe"
LICENCE_RIFE = "RIFE v4.26 (Practical-RIFE, MIT) interpolation"


def out_id(src_id: str, fps: int) -> str:
    if src_id.endswith("_1080"):
        return f"{src_id[:-5]}_i{fps}_1080"
    return f"{src_id}_i{fps}"


def probe(path: Path) -> tuple[int, int, float, int]:
    r = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), float(Fraction(s["r_frame_rate"])), int(s["nb_read_packets"])


def read_frames(path: Path, w: int, h: int):
    import numpy as np

    raw = subprocess.run([FF, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def load_rife():
    import torch

    sys.path.insert(0, str(RIFE_ROOT))
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    from train_log.RIFE_HDv3 import Model  # noqa: E402

    model = Model()
    model.load_model(str(RIFE_ROOT / "train_log"), -1)
    model.eval()
    model.device()
    return model


def interpolate(model, frames, src_fps: float, fps: int, *, scale: float = 1.0):
    """Yield output frames (H, W, 3 uint8) for the same duration at ``fps``."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = dev.type == "cuda"
    n_src, h, w, _ = frames.shape
    duration = (n_src - 1) / src_fps
    n_out = int(math.floor(duration * fps + 1e-6)) + 1
    tmp = max(128, int(128 / scale))
    ph, pw = ((h - 1) // tmp + 1) * tmp, ((w - 1) // tmp + 1) * tmp

    def tens(i: int):
        t = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev).permute(2, 0, 1)[None].float() / 255.0
        t = F.pad(t, (0, pw - w, 0, ph - h))
        return t

    cache: dict[int, object] = {}

    def get(i: int):
        if i not in cache:
            if len(cache) > 3:
                cache.pop(min(cache))
            cache[i] = tens(i)
        return cache[i]

    for k in range(n_out):
        p = k * src_fps / fps
        i = int(math.floor(p + 1e-9))
        frac = p - i
        if frac < 1e-6 or i >= n_src - 1:
            yield frames[min(i, n_src - 1)]
            continue
        # fp16 on CUDA via autocast (fp32 weights/inputs; grid_sample etc. promote as needed)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=half):
            mid = model.inference(get(i), get(i + 1), timestep=frac, scale=scale)
        out = (mid[0, :, :h, :w].float().clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        yield out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="jsonl: {src, fps} per line")
    ap.add_argument("--out-dir", default="/workspace/dataset/hfr_ai")
    ap.add_argument("--max-frames", type=int, default=None, help="test: stop after N output frames")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(args.plan) if l.strip()]
    todo = [j for j in jobs if not (out_dir / f"{out_id(Path(j['src']).stem, int(j['fps']))}.mkv").exists()]
    print(f"{len(todo)} of {len(jobs)} clips to interpolate -> {out_dir}", flush=True)
    if not todo:
        return
    model = load_rife()

    for j in todo:
        t0 = time.time()
        src = Path(j["src"])
        fps = int(j["fps"])
        oid = out_id(src.stem, fps)
        w, h, src_fps, _ = probe(src)
        frames = read_frames(src, w, h)
        out = out_dir / f"{oid}.mkv"
        tmp = out.with_suffix(".tmp.mkv")
        enc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-r", str(fps), "-i", "-", "-vf", "format=yuv420p", "-c:v", "ffv1", "-level", "3",
                                "-threads", "4", str(tmp)], stdin=subprocess.PIPE)
        n = 0
        for fr in interpolate(model, frames, src_fps, fps):
            enc.stdin.write(fr.tobytes())
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {oid}")
        tmp.rename(out)
        src_meta = {}
        sj = src.with_suffix(".json")
        if sj.exists():
            src_meta = json.loads(sj.read_text())
        meta = {
            "id": oid, "kind": "ai", "style": src_meta.get("style", "live"),
            "intended_tier": src_meta.get("intended_tier") or src_meta.get("tier"),
            "prompt": src_meta.get("prompt"), "source_id": src.stem, "source_path": str(src),
            "method": "rife_v4.26", "src_fps": src_fps, "fps": float(fps), "frames": n,
            "width": w, "height": h, "seconds": round(time.time() - t0, 1),
            "source": f"{src_meta.get('source', 'Wan2.2-TI2V-5B')} + RIFE v4.26",
            "licence": f"{src_meta.get('licence', 'Apache-2.0 model output')}; {LICENCE_RIFE}",
            "path": str(out),
        }
        (out_dir / f"{oid}.json").write_text(json.dumps(meta, indent=1))
        with open(out_dir / "manifest.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")
        print(f"{oid}: {n} frames @ {fps} fps, {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
