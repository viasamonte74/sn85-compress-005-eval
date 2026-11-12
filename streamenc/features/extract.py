"""Lightweight per-clip content features (CPU, sampled-frame friendly)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np

FEATURE_NAMES = (
    "spatial_complexity",
    "temporal_complexity",
    "edge_density",
    "frame_diff_mean",
    "frame_diff_std",
    "frame_diff_p95",
    "repeated_frame_ratio",
    "motion_complexity",
    "texture_complexity",
    "smooth_gradient_ratio",
    "chroma_complexity",
    "temporal_instability",
    # Phase 5b additions for hard-content analysis
    "highfreq_energy",
    "noise_estimate",
    "entropy",
    "motion_p95",
    "spatial_p95",
    "detail_density",
)


@dataclass(frozen=True)
class FeatureVector:
    spatial_complexity: float
    temporal_complexity: float
    edge_density: float
    frame_diff_mean: float
    frame_diff_std: float
    frame_diff_p95: float
    repeated_frame_ratio: float
    motion_complexity: float
    texture_complexity: float
    smooth_gradient_ratio: float
    chroma_complexity: float
    temporal_instability: float
    highfreq_energy: float = 0.0
    noise_estimate: float = 0.0
    entropy: float = 0.0
    motion_p95: float = 0.0
    spatial_p95: float = 0.0
    detail_density: float = 0.0
    n_frames_analyzed: int = 0
    n_pairs: int = 0
    analysis_width: int = 0
    analysis_height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def values(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=np.float64)


def _to_gray_yuv(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    return yuv[:, :, 0], yuv


def _resize_for_analysis(bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    scale = max_width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (max_width, nh), interpolation=cv2.INTER_AREA)


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _highfreq_and_noise(gray: np.ndarray) -> tuple[float, float]:
    """High-pass energy + Laplacian MAD noise proxy."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    hp = gray.astype(np.float32) - blur.astype(np.float32)
    highfreq = float(np.mean(np.abs(hp)))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # Median absolute deviation ~ robust noise estimate
    med = float(np.median(lap))
    mad = float(np.median(np.abs(lap - med)))
    noise = 1.4826 * mad  # Gaussian consistency scale
    return highfreq, noise


def _spatial_metrics(gray: np.ndarray) -> tuple[float, float, float, float, float]:
    """spatial, edge_density, texture, smooth_ratio, detail_density."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    spatial = float(lap.var()) / 1000.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = mag / 255.0
    smooth = float(np.mean(mag_norm < 0.05))
    # Fraction of pixels with strong detail (encode-costly).
    detail = float(np.mean(mag_norm > 0.12))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    gh, gw = gray.shape
    bh, bw = 8, 8
    ys = gh - (gh % bh)
    xs = gw - (gw % bw)
    if ys < bh or xs < bw:
        texture = float(gray.std()) / 64.0
    else:
        blocks = gray[:ys, :xs].reshape(ys // bh, bh, xs // bw, bw)
        block_stds = blocks.std(axis=(1, 3))
        texture = float(block_stds.std()) / 32.0
    return spatial, edge_density, texture, smooth, detail


def _chroma_complexity(yuv: np.ndarray) -> float:
    u = yuv[:, :, 1].astype(np.float32)
    v = yuv[:, :, 2].astype(np.float32)
    return float((u.std() + v.std()) * 0.5 / 64.0)


def _block_motion(prev_gray: np.ndarray, gray: np.ndarray, block: int = 16) -> float:
    h, w = gray.shape
    ys = h - (h % block)
    xs = w - (w % block)
    if ys < block or xs < block:
        return float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0)
    a = prev_gray[:ys, :xs].astype(np.float32)
    b = gray[:ys, :xs].astype(np.float32)
    diff = np.abs(a - b)
    nb_y, nb_x = ys // block, xs // block
    blocks = diff.reshape(nb_y, block, nb_x, block).mean(axis=(1, 3))
    return float(blocks.mean() / 255.0)


def accumulate_frame_stats(
    frames_bgr: Iterable[np.ndarray],
    *,
    max_width: int = 640,
    repeat_thresh: float = 0.5,
) -> FeatureVector:
    spatial_vals: list[float] = []
    edge_vals: list[float] = []
    texture_vals: list[float] = []
    smooth_vals: list[float] = []
    chroma_vals: list[float] = []
    hf_vals: list[float] = []
    noise_vals: list[float] = []
    entropy_vals: list[float] = []
    detail_vals: list[float] = []
    diffs: list[float] = []
    motions: list[float] = []

    prev_gray: np.ndarray | None = None
    n = 0
    aw = ah = 0

    for bgr in frames_bgr:
        small = _resize_for_analysis(bgr, max_width)
        aw, ah = small.shape[1], small.shape[0]
        gray, yuv = _to_gray_yuv(small)
        spatial, edge, texture, smooth, detail = _spatial_metrics(gray)
        hf, noise = _highfreq_and_noise(gray)
        spatial_vals.append(spatial)
        edge_vals.append(edge)
        texture_vals.append(texture)
        smooth_vals.append(smooth)
        detail_vals.append(detail)
        chroma_vals.append(_chroma_complexity(yuv))
        hf_vals.append(hf)
        noise_vals.append(noise)
        entropy_vals.append(_entropy(gray))
        if prev_gray is not None:
            mad = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            diffs.append(mad)
            motions.append(_block_motion(prev_gray, gray))
        prev_gray = gray
        n += 1

    if n == 0:
        zeros = {k: 0.0 for k in FEATURE_NAMES}
        return FeatureVector(**zeros, n_frames_analyzed=0, n_pairs=0)

    diff_arr = np.asarray(diffs, dtype=np.float64) if diffs else np.asarray([0.0])
    motion_arr = np.asarray(motions, dtype=np.float64) if motions else np.asarray([0.0])
    spatial_arr = np.asarray(spatial_vals, dtype=np.float64)
    repeated = float(np.mean(diff_arr < repeat_thresh)) if diffs else 1.0
    instability = float(diff_arr.std() / (diff_arr.mean() + 1e-6)) if len(diff_arr) > 1 else 0.0

    return FeatureVector(
        spatial_complexity=float(spatial_arr.mean()),
        temporal_complexity=float(diff_arr.mean() / 255.0),
        edge_density=float(np.mean(edge_vals)),
        frame_diff_mean=float(diff_arr.mean()),
        frame_diff_std=float(diff_arr.std()),
        frame_diff_p95=float(np.percentile(diff_arr, 95)),
        repeated_frame_ratio=repeated,
        motion_complexity=float(motion_arr.mean()),
        texture_complexity=float(np.mean(texture_vals)),
        smooth_gradient_ratio=float(np.mean(smooth_vals)),
        chroma_complexity=float(np.mean(chroma_vals)),
        temporal_instability=instability,
        highfreq_energy=float(np.mean(hf_vals)),
        noise_estimate=float(np.mean(noise_vals)),
        entropy=float(np.mean(entropy_vals)),
        motion_p95=float(np.percentile(motion_arr, 95)),
        spatial_p95=float(np.percentile(spatial_arr, 95)),
        detail_density=float(np.mean(detail_vals)),
        n_frames_analyzed=n,
        n_pairs=len(diffs),
        analysis_width=aw,
        analysis_height=ah,
    )
