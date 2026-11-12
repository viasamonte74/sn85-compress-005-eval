"""Video-level feature analysis: sampled vs full-frame modes."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np

from streamenc.features.extract import FEATURE_NAMES, FeatureVector, accumulate_frame_stats
from streamenc.probe import probe_video

Mode = Literal["sampled", "full"]


@dataclass(frozen=True)
class AnalysisResult:
    path: str
    clip_id: str
    mode: str
    width: int
    height: int
    fps: float
    frames_total: int
    sample_stride: int
    max_samples: int
    features: FeatureVector
    decode_seconds: float
    analyze_seconds: float
    total_seconds: float

    def to_row(self) -> dict:
        row = {
            "path": self.path,
            "clip_id": self.clip_id,
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames_total": self.frames_total,
            "sample_stride": self.sample_stride,
            "max_samples": self.max_samples,
            "decode_seconds": self.decode_seconds,
            "analyze_seconds": self.analyze_seconds,
            "total_seconds": self.total_seconds,
        }
        row.update(self.features.to_dict())
        return row


def _iter_indices(n_frames: int, *, mode: Mode, max_samples: int, stride: int) -> list[int]:
    if n_frames <= 0:
        return []
    if mode == "full":
        return list(range(n_frames))
    if max_samples <= 1:
        return [0]
    if n_frames <= max_samples:
        if stride <= 1:
            return list(range(n_frames))
        return list(range(0, n_frames, stride))
    # Sequential stride sampling (avoids expensive random seeks on FFV1/MKV).
    auto = max(1, int(np.ceil(n_frames / float(max_samples))))
    step = max(stride, auto)
    idx = list(range(0, n_frames, step))
    if len(idx) > max_samples:
        idx = idx[:max_samples]
    # Ensure last frame included for end-of-clip coverage when room allows.
    if idx[-1] != n_frames - 1 and len(idx) < max_samples:
        idx.append(n_frames - 1)
    return idx


def iter_selected_frames(
    path: str,
    indices: list[int],
) -> Iterator[np.ndarray]:
    """Decode frames by sequential grab/read (fast for FFV1; no random seek)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    want = set(indices)
    try:
        cur = 0
        while want:
            if cur in want:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame
                want.discard(cur)
                cur += 1
            else:
                if not cap.grab():
                    break
                cur += 1
    finally:
        cap.release()


def analyze_video(
    path: str | Path,
    *,
    mode: Mode = "sampled",
    max_samples: int = 24,
    sample_stride: int = 1,
    max_width: int = 640,
) -> AnalysisResult:
    path = Path(path)
    meta = probe_video(str(path), count_frames=True)
    n = int(meta.nb_frames)
    indices = _iter_indices(n, mode=mode, max_samples=max_samples, stride=sample_stride)

    t0 = time.perf_counter()
    frames: list[np.ndarray] = list(iter_selected_frames(str(path), indices))
    decode_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    feats = accumulate_frame_stats(frames, max_width=max_width)
    analyze_s = time.perf_counter() - t1

    return AnalysisResult(
        path=str(path),
        clip_id=path.stem,
        mode=mode,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        frames_total=n,
        sample_stride=sample_stride if mode == "sampled" else 1,
        max_samples=max_samples if mode == "sampled" else n,
        features=feats,
        decode_seconds=decode_s,
        analyze_seconds=analyze_s,
        total_seconds=decode_s + analyze_s,
    )


def feature_agreement(a: FeatureVector, b: FeatureVector) -> dict:
    """Compare two feature vectors (sampled vs full)."""
    va, vb = a.values(), b.values()
    abs_err = np.abs(va - vb)
    # Pearson on the 12-D vectors; guard constant vectors
    if va.std() < 1e-12 or vb.std() < 1e-12:
        corr = 1.0 if float(np.max(abs_err)) < 1e-9 else 0.0
    else:
        corr = float(np.corrcoef(va, vb)[0, 1])
    rel = abs_err / (np.abs(vb) + 1e-6)
    per = {name: float(abs_err[i]) for i, name in enumerate(FEATURE_NAMES)}
    return {
        "pearson_corr": corr,
        "mae": float(abs_err.mean()),
        "max_abs_err": float(abs_err.max()),
        "mean_rel_err": float(rel.mean()),
        "per_feature_abs_err": per,
    }


def agreement_row(sampled: AnalysisResult, full: AnalysisResult) -> dict:
    agr = feature_agreement(sampled.features, full.features)
    return {
        "clip_id": sampled.clip_id,
        "frames_total": sampled.frames_total,
        "sampled_frames": sampled.features.n_frames_analyzed,
        "full_frames": full.features.n_frames_analyzed,
        "sampled_seconds": sampled.total_seconds,
        "full_seconds": full.total_seconds,
        "speedup": (full.total_seconds / sampled.total_seconds) if sampled.total_seconds > 0 else None,
        "pearson_corr": agr["pearson_corr"],
        "mae": agr["mae"],
        "max_abs_err": agr["max_abs_err"],
        "mean_rel_err": agr["mean_rel_err"],
        **{f"err_{k}": v for k, v in agr["per_feature_abs_err"].items()},
    }
