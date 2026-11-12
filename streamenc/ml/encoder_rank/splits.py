"""Family-safe train/val/test splits for encoder-rank."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def clip_kind(path: Path) -> str:
    p = str(path)
    parent = path.parent.name
    if "generated_hard" in p:
        return "generated_hard"
    # hfr_* = high-frame-rate sets (hfr_ai: RIFE-interpolated Wan clips; hfr_wan: native Wan slow-motion
    # prompts stored at 50/60 fps; hfr_render / hfr_screen: native)
    # ai_gen3: native 1080p Wan set (720p twins downscaled), 2026-11-10
    if "/ai/" in p or parent in ("ai", "ai1080", "ai_gen2", "hfr_ai", "hfr_wan", "ai_gen3"):
        return "ai"
    # render2: procedural Blender set 2; render_film: CC BY Blender open-movie segments (2026-11-10)
    # render_hf: Hugging Face 0xmsu/sn85-blender-removal-pairs inputs (2026-11-11)
    if "/render/" in p or parent in ("render", "hfr_render", "render2", "render_film", "render_hf"):
        return "render"
    # screen2: screen set 2 at 24/25/30 fps (2026-11-10)
    # screen_real: Hugging Face nickossaf/Screen-Content-Video real recordings (2026-11-11)
    if "/screen/" in p or parent in ("screen", "hfr_screen", "screen2", "screen_real", "screen_real709"):
        return "screen"
    return "other"


def clip_tier(stem: str) -> str:
    s = stem.lower()
    if "gen_hard" in s or "_hard_" in s:
        return "hard"
    if "_medium_" in s:
        return "medium"
    if "_easy_" in s:
        return "easy"
    return "other"


def clip_family(stem: str, path: Path | None = None) -> str:
    """Group near-duplicates / augmentations so they share a split.

    Examples:
      ai_hard_13 → ai_hard_13
      gen_hard_01_grain_ai_hard_13 → ai_hard_13
      rnd_hard_01_1080 → rnd_hard_01
    """
    s = stem.replace("_1080", "")
    # frame-interpolated derivative (e.g. ai_hard_05_i60[_1080]) shares its source clip's family
    s = re.sub(r"_i\d{2,3}$", "", s)
    if s.startswith("gen_hard_"):
        for tag in ("ai_hard_", "ai_medium_", "ai_easy_", "rnd_hard_", "rnd_medium_", "rnd_easy_", "scr_hard_", "scr_medium_", "scr_easy_"):
            if tag in s:
                return s[s.index(tag) :]
        return s
    # Drop trailing recipe-like suffixes if any
    return s


def assign_split(family: str, *, val_ratio: float = 0.15, test_ratio: float = 0.15) -> str:
    """Deterministic hash split by family id."""
    h = hashlib.md5(family.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < test_ratio:
        return "test"
    if x < test_ratio + val_ratio:
        return "val"
    return "train"
