"""Dataset discovery and leakage-safe train/test splits by scene family."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClipInfo:
    clip_id: str
    path: str
    kind: str  # ai | render | screen
    family: str  # scene family without resolution suffix
    resolution_tag: str  # 720 | 1080 | other
    split: str  # train | test
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frames: int | None = None
    tier: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_FAMILY_RE = re.compile(r"^(?P<family>.+?)(?:_1080)?$")


def family_id(stem: str) -> str:
    m = _FAMILY_RE.match(stem)
    return m.group("family") if m else stem


def kind_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent in {"ai", "ai1080"}:
        return "ai"
    if parent == "render":
        return "render"
    if parent == "screen":
        return "screen"
    return parent


def resolution_tag(stem: str, width: int | None = None) -> str:
    if stem.endswith("_1080") or width == 1920:
        return "1080"
    if width == 1280:
        return "720"
    return "other"


def discover_clips(dataset: Path) -> list[ClipInfo]:
    dataset = Path(dataset)
    manifest: dict[str, dict] = {}
    man_path = dataset / "manifest.jsonl"
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            manifest[row["id"]] = row

    # Merge generated_hard manifest entries if present.
    gen_man = dataset / "generated_hard" / "manifest.jsonl"
    if gen_man.exists():
        for line in gen_man.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                manifest[row["id"]] = {**manifest.get(row["id"], {}), **row}
            except Exception:  # noqa: BLE001
                pass

    clips: list[ClipInfo] = []
    subdirs = ("ai", "ai1080", "render", "screen", "generated_hard")
    for sub in subdirs:
        for mkv in sorted((dataset / sub).glob("*.mkv")):
            stem = mkv.stem
            fam = family_id(stem)
            meta = manifest.get(stem) or manifest.get(fam) or {}
            # generated_hard sidecars / local manifest
            side = Path(str(mkv.with_suffix(".json")))
            if side.exists():
                try:
                    meta = {**meta, **json.loads(side.read_text())}
                except Exception:  # noqa: BLE001
                    pass
            width = meta.get("width")
            height = meta.get("height")
            # Prefer explicit manifest split for the family; otherwise hash-stable assign.
            split = meta.get("split")
            if split not in {"train", "test"}:
                digest = hashlib.sha256(fam.encode()).hexdigest()
                split = "test" if int(digest[:8], 16) % 5 == 0 else "train"
            kind = meta.get("kind") or kind_from_path(mkv)
            if kind == "generated_hard":
                # Map to source kind hint embedded in name when present.
                if "_ai_" in stem:
                    kind = "ai"
                elif "_rnd_" in stem:
                    kind = "render"
                elif "_scr_" in stem:
                    kind = "screen"
            clips.append(
                ClipInfo(
                    clip_id=stem,
                    path=str(mkv),
                    kind=kind,
                    family=fam,
                    resolution_tag=resolution_tag(stem, width),
                    split=split,
                    width=width,
                    height=height,
                    fps=meta.get("fps"),
                    frames=meta.get("frames"),
                    tier=meta.get("tier") or meta.get("intended_tier") or (
                        "hard" if "hard" in stem else None
                    ),
                )
            )
    return clips


def enforce_family_split(clips: list[ClipInfo]) -> list[ClipInfo]:
    """If any family appears in both splits, force entire family to majority split."""
    by_family: dict[str, list[ClipInfo]] = {}
    for c in clips:
        by_family.setdefault(c.family, []).append(c)
    fixed: list[ClipInfo] = []
    for fam, members in by_family.items():
        splits = {m.split for m in members}
        if len(splits) == 1:
            fixed.extend(members)
            continue
        # Prefer test if any member was test in manifest (held-out safety), else train.
        chosen = "test" if "test" in splits and sum(m.split == "test" for m in members) >= 1 else "train"
        # Actually for leakage: if mixed, put ALL in train unless family hash says test.
        digest = hashlib.sha256(fam.encode()).hexdigest()
        chosen = "test" if int(digest[:8], 16) % 5 == 0 else "train"
        for m in members:
            fixed.append(
                ClipInfo(**{**m.to_dict(), "split": chosen})
            )
    return fixed


def select_benchmark_clips(
    clips: list[ClipInfo],
    *,
    train_per_kind: int = 2,
    test_per_kind: int = 1,
    prefer_720p: bool = True,
) -> tuple[list[ClipInfo], list[ClipInfo]]:
    """Pick a compact staged-search corpus with no shared families across splits."""
    clips = enforce_family_split(clips)
    train_fams = {c.family for c in clips if c.split == "train"}
    test_fams = {c.family for c in clips if c.split == "test"}
    # Ensure disjoint.
    overlap = train_fams & test_fams
    if overlap:
        clips = [
            ClipInfo(**{**c.to_dict(), "split": "train"}) if c.family in overlap else c for c in clips
        ]

    def pick(split: str, n_per_kind: int) -> list[ClipInfo]:
        out: list[ClipInfo] = []
        for kind in ("ai", "render", "screen"):
            pool = [c for c in clips if c.split == split and c.kind == kind]
            if prefer_720p:
                pool = sorted(pool, key=lambda c: (0 if c.resolution_tag == "720" else 1, c.clip_id))
            else:
                pool = sorted(pool, key=lambda c: c.clip_id)
            # unique families
            seen = set()
            chosen = []
            for c in pool:
                if c.family in seen:
                    continue
                seen.add(c.family)
                chosen.append(c)
                if len(chosen) >= n_per_kind:
                    break
            out.extend(chosen)
        return out

    return pick("train", train_per_kind), pick("test", test_per_kind)
