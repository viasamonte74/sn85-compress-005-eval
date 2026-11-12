#!/usr/bin/env python3
"""Generate additional hard FFV1 clips by augmenting existing sources.

Augmentations raise high-frequency energy / temporal chaos while preserving
resolution, fps, and frame count (lossless FFV1 Matroska). Used to stress
content analysis and hard-path routing — not as official competition inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.probe import probe_video
from streamenc.util.logging import get_logger

log = get_logger("gen_hard")

# Filter recipes that increase encode difficulty without changing geometry.
RECIPES: dict[str, str] = {
    # Fine grain + temporal noise (classic bitrate killer).
    "grain": "noise=alls=18:allf=t+u,format=yuv420p",
    # Sharpen then grain → high-frequency texture.
    "sharp_grain": "unsharp=5:5:1.2:5:5:0.0,noise=alls=12:allf=t+u,format=yuv420p",
    # Temporal chaos via frame blending + noise.
    "tblend_noise": "tblend=all_mode=addition:all_opacity=0.35,noise=alls=10:allf=t,format=yuv420p",
    # Micro-shake via tiny rotate (preserves exact iw×ih) + grain.
    "shake_grain": (
        "rotate=0.012*sin(n/4):c=black:ow=iw:oh=ih,"
        "noise=alls=14:allf=t+u,format=yuv420p"
    ),
    # High-contrast edge emphasis + luma noise.
    "edge_boost": (
        "eq=contrast=1.25:saturation=1.15,unsharp=7:7:1.5:7:7:0.5,"
        "noise=alls=10:allf=t,format=yuv420p"
    ),
}

def _official_hard_sources(dataset: Path, limit: int) -> list[Path]:
    scores_path = dataset / "official_scores.jsonl"
    ranked: list[tuple[float, Path]] = []
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            clip = row["clip"]
            # Prefer low-scoring hard/medium as bases.
            if "_hard_" not in clip and "_medium_" not in clip:
                continue
            for sub in ("ai", "render", "screen", "ai1080"):
                p = dataset / sub / f"{clip}.mkv"
                if p.exists() and "_1080" not in p.stem:
                    ranked.append((float(row.get("score", 1.0)), p))
                    break
    ranked.sort(key=lambda x: x[0])
    out: list[Path] = []
    seen = set()
    for _, p in ranked:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= limit:
            break
    if len(out) < limit:
        # Fallback: any labeled hard mkv.
        for sub in ("ai", "render", "screen"):
            for p in sorted((dataset / sub).glob("*hard*.mkv")):
                if "_1080" in p.stem or p in seen:
                    continue
                out.append(p)
                seen.add(p)
                if len(out) >= limit:
                    return out
    return out


def generate_one(
    src: Path,
    dst: Path,
    *,
    recipe: str,
    vf: str,
) -> dict:
    meta = probe_video(str(src), count_frames=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Force exact frame count via -frames:v; FFV1 lossless.
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "matroska",
        "-i",
        str(src),
        "-vf",
        vf,
        "-frames:v",
        str(meta.nb_frames),
        "-r",
        str(meta.fps),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-f",
        "matroska",
        str(dst),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed for {src.name}/{recipe}: {proc.stderr[:400]}")
    out_meta = probe_video(str(dst), count_frames=True)
    info = {
        "id": dst.stem,
        "kind": "generated_hard",
        "source_clip": src.stem,
        "source_path": str(src),
        "recipe": recipe,
        "vf": vf,
        "path": str(dst),
        "width": out_meta.width,
        "height": out_meta.height,
        "fps": out_meta.fps,
        "frames": out_meta.nb_frames,
        "intended_tier": "hard",
        "tier": "hard",
        "split": "train",
        "licence": "derived-from-dataset-augmentation",
        "notes": "Synthetic hard augmentation for analysis/hard-path training",
    }
    if out_meta.width != meta.width or out_meta.height != meta.height:
        raise RuntimeError(f"geometry changed for {dst}")
    if out_meta.nb_frames != meta.nb_frames:
        log.warning(
            "frame count drift %s: in=%d out=%d",
            dst.name,
            meta.nb_frames,
            out_meta.nb_frames,
        )
    (dst.with_suffix(".json")).write_text(json.dumps(info, indent=2))
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument(
        "--out-dir",
        default="/workspace/dataset/generated_hard",
        help="destination for generated FFV1 clips",
    )
    parser.add_argument("--sources", type=int, default=8, help="number of source clips")
    parser.add_argument(
        "--recipes",
        default="grain,sharp_grain,shake_grain,edge_boost",
        help="comma-separated recipe names",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_names = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipe_names:
        if r not in RECIPES:
            raise SystemExit(f"unknown recipe {r}; choose from {sorted(RECIPES)}")

    sources = _official_hard_sources(dataset, args.sources)
    if not sources:
        raise SystemExit("no source clips found")

    manifest = []
    idx = 1
    for src in sources:
        for recipe in recipe_names:
            stem = f"gen_hard_{idx:02d}_{recipe}_{src.stem}"
            dst = out_dir / f"{stem}.mkv"
            log.info("[%d] %s ← %s (%s)", idx, dst.name, src.name, recipe)
            try:
                info = generate_one(src, dst, recipe=recipe, vf=RECIPES[recipe])
                manifest.append(info)
                idx += 1
            except Exception as exc:  # noqa: BLE001
                log.error("failed %s/%s: %s", src.name, recipe, exc)

    man_path = out_dir / "manifest.jsonl"
    with man_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    summary = {
        "n_generated": len(manifest),
        "n_sources": len(sources),
        "recipes": recipe_names,
        "out_dir": str(out_dir),
        "sources": [p.stem for p in sources],
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Generated %d hard clips → %s", len(manifest), out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
