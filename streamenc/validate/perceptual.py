"""Official CPU perceptual decision surface + matching sampler.

Decision formulas and thresholds are from miner-vidaio
``vidaio/scoring/perceptual_cpu.py``. Sampling matches
``CpuPerceptualCheckBackend._analyze_opencv`` / ``_sample_positions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from streamenc.validate.official_streaming import ValidityViolation


@dataclass(frozen=True)
class CpuPerceptualConfig:
    sample_frames: int = 8
    sample_edge: int = 128
    tone_mean_delta_max: float = 0.06
    tone_std_delta_max: float = 0.08
    grayscale_reference_chroma_min: float = 0.03
    grayscale_chroma_ratio_min: float = 0.35
    chroma_mae_max: float = 0.10
    chroma_energy_ratio_min: float = 0.40
    chroma_energy_ratio_max: float = 1.80


@dataclass(frozen=True)
class PerceptualStatistics:
    sampled_pixels: int
    reference_luma_mean: float
    candidate_luma_mean: float
    reference_luma_std: float
    candidate_luma_std: float
    reference_chroma_energy: float
    candidate_chroma_energy: float
    chroma_mae: float


@dataclass(frozen=True)
class PerceptualCheckResult:
    name: str
    passed: bool
    measure: float
    limit: float | None
    detail: str


def sample_positions(length: int, count: int) -> list[int]:
    count = min(length, count)
    if count <= 0:
        return []
    if count == 1:
        return [length // 2]
    return [i * (length - 1) // (count - 1) for i in range(count)]


def _frame_count(path: str) -> int:
    from streamenc.probe import probe_video

    return probe_video(path, count_frames=True).nb_frames


def _extract_bgr_frame(path: str, frame_index: int, *, ffmpeg: str, tmp_png: str) -> "object":
    """Decode one frame with software ffmpeg (AV1-safe) into a BGR ndarray."""
    import cv2

    from streamenc.util.subprocess_run import run_command

    # select exact frame by index; force software decode where HW fails.
    vf = f"select=eq(n\\,{frame_index})"
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-hwaccel",
        "none",
        "-i",
        path,
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        "-y",
        tmp_png,
    ]
    result = run_command(argv, timeout=120)
    if not result.ok:
        raise RuntimeError(f"could not decode frame {frame_index}: {result.stderr.strip()}")
    img = cv2.imread(tmp_png, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read decoded frame {frame_index}")
    return img


def collect_statistics(
    reference: str,
    candidate: str,
    *,
    config: CpuPerceptualConfig | None = None,
    ffmpeg: str | None = None,
) -> PerceptualStatistics:
    import numpy as np

    from streamenc.validate.vmaf import default_ffmpeg_with_vmaf

    config = config or CpuPerceptualConfig()
    ffmpeg = ffmpeg or default_ffmpeg_with_vmaf()
    ref_frames = _frame_count(reference)
    total = sum_y_ref = sum_y_cand = sum_y2_ref = sum_y2_cand = 0
    sum_chroma_ref = sum_chroma_cand = sum_chroma_diff = 0
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="perc-") as tmp:
        ref_png = str(Path(tmp) / "ref.png")
        cand_png = str(Path(tmp) / "cand.png")
        for frame_index in sample_positions(ref_frames, config.sample_frames):
            ref = _extract_bgr_frame(reference, frame_index, ffmpeg=ffmpeg, tmp_png=ref_png)
            cand = _extract_bgr_frame(candidate, frame_index, ffmpeg=ffmpeg, tmp_png=cand_png)
            if ref.shape != cand.shape or len(ref.shape) != 3 or ref.shape[2] != 3:
                raise RuntimeError(f"frame shape mismatch {ref.shape!r} != {cand.shape!r}")
            rows = np.asarray(sample_positions(ref.shape[0], config.sample_edge), dtype=np.intp)
            cols = np.asarray(sample_positions(ref.shape[1], config.sample_edge), dtype=np.intp)
            ref = ref[rows[:, None], cols[None, :]].astype(np.int32)
            cand = cand[rows[:, None], cols[None, :]].astype(np.int32)
            ref_b, ref_g, ref_r = (ref[:, :, i] for i in range(3))
            cand_b, cand_g, cand_r = (cand[:, :, i] for i in range(3))
            y_ref = (29 * ref_b + 150 * ref_g + 77 * ref_r + 128) >> 8
            y_cand = (29 * cand_b + 150 * cand_g + 77 * cand_r + 128) >> 8
            rg_ref, bg_ref = ref_r - ref_g, ref_b - ref_g
            rg_cand, bg_cand = cand_r - cand_g, cand_b - cand_g
            pixels = int(y_ref.size)
            total += pixels
            sum_y_ref += int(y_ref.sum(dtype=np.int64))
            sum_y_cand += int(y_cand.sum(dtype=np.int64))
            sum_y2_ref += int((y_ref * y_ref).sum(dtype=np.int64))
            sum_y2_cand += int((y_cand * y_cand).sum(dtype=np.int64))
            sum_chroma_ref += int((np.abs(rg_ref) + np.abs(bg_ref)).sum(dtype=np.int64))
            sum_chroma_cand += int((np.abs(rg_cand) + np.abs(bg_cand)).sum(dtype=np.int64))
            sum_chroma_diff += int(
                (np.abs(rg_ref - rg_cand) + np.abs(bg_ref - bg_cand)).sum(dtype=np.int64)
            )
    if total < 1:
        raise RuntimeError("CPU perceptual check sampled no pixels")
    ref_mean_raw = sum_y_ref / total
    cand_mean_raw = sum_y_cand / total
    ref_variance = max(0.0, sum_y2_ref / total - ref_mean_raw**2)
    cand_variance = max(0.0, sum_y2_cand / total - cand_mean_raw**2)
    return PerceptualStatistics(
        sampled_pixels=total,
        reference_luma_mean=ref_mean_raw / 255.0,
        candidate_luma_mean=cand_mean_raw / 255.0,
        reference_luma_std=math.sqrt(ref_variance) / 255.0,
        candidate_luma_std=math.sqrt(cand_variance) / 255.0,
        reference_chroma_energy=sum_chroma_ref / (total * 510.0),
        candidate_chroma_energy=sum_chroma_cand / (total * 510.0),
        chroma_mae=sum_chroma_diff / (total * 1020.0),
    )


def tone_check(stats: PerceptualStatistics, config: CpuPerceptualConfig) -> PerceptualCheckResult:
    mean_delta = abs(stats.candidate_luma_mean - stats.reference_luma_mean)
    std_delta = abs(stats.candidate_luma_std - stats.reference_luma_std)
    normalized = max(mean_delta / config.tone_mean_delta_max, std_delta / config.tone_std_delta_max)
    return PerceptualCheckResult(
        name="tone_manipulation",
        passed=normalized <= 1.0,
        measure=normalized,
        limit=1.0,
        detail=(
            f"CPU tone check: normalized deviation {normalized:.6f}; "
            f"mean delta {mean_delta:.6f}/{config.tone_mean_delta_max:.6f}, "
            f"luma-std delta {std_delta:.6f}/{config.tone_std_delta_max:.6f}"
        ),
    )


def grayscale_check(stats: PerceptualStatistics, config: CpuPerceptualConfig) -> PerceptualCheckResult:
    reference = stats.reference_chroma_energy
    if reference < config.grayscale_reference_chroma_min:
        return PerceptualCheckResult(
            name="color_grayscale",
            passed=True,
            measure=1.0,
            limit=None,
            detail=(
                "CPU grayscale check: reference is naturally low-chroma "
                f"({reference:.6f} < {config.grayscale_reference_chroma_min:.6f})"
            ),
        )
    ratio = stats.candidate_chroma_energy / reference
    return PerceptualCheckResult(
        name="color_grayscale",
        passed=ratio >= config.grayscale_chroma_ratio_min,
        measure=ratio,
        limit=config.grayscale_chroma_ratio_min,
        detail=(
            "CPU grayscale check: candidate/reference chroma-energy ratio "
            f"{ratio:.6f}, minimum {config.grayscale_chroma_ratio_min:.6f}"
        ),
    )


def chroma_check(stats: PerceptualStatistics, config: CpuPerceptualConfig) -> PerceptualCheckResult:
    mae_factor = stats.chroma_mae / config.chroma_mae_max
    reference = stats.reference_chroma_energy
    ratio = (
        stats.candidate_chroma_energy / reference
        if reference >= config.grayscale_reference_chroma_min
        else 1.0
    )
    if ratio < config.chroma_energy_ratio_min:
        ratio_factor = config.chroma_energy_ratio_min / max(ratio, 1e-12)
    elif ratio > config.chroma_energy_ratio_max:
        ratio_factor = ratio / config.chroma_energy_ratio_max
    else:
        ratio_factor = 1.0
    normalized = max(mae_factor, ratio_factor)
    return PerceptualCheckResult(
        name="chroma_uv",
        passed=normalized <= 1.0,
        measure=normalized,
        limit=1.0,
        detail=(
            f"CPU chroma check: normalized deviation {normalized:.6f}; "
            f"chroma MAE {stats.chroma_mae:.6f}/{config.chroma_mae_max:.6f}, "
            f"energy ratio {ratio:.6f} in "
            f"[{config.chroma_energy_ratio_min:.6f}, {config.chroma_energy_ratio_max:.6f}]"
        ),
    )


def run_perceptual_gates(
    reference: str,
    candidate: str,
    *,
    config: CpuPerceptualConfig | None = None,
) -> tuple[list[PerceptualCheckResult], list[ValidityViolation]]:
    config = config or CpuPerceptualConfig()
    stats = collect_statistics(reference, candidate, config=config)
    results = [
        tone_check(stats, config),
        grayscale_check(stats, config),
        chroma_check(stats, config),
    ]
    violations: list[ValidityViolation] = []
    code_map = {
        "tone_manipulation": "TONE_MANIPULATION",
        "color_grayscale": "COLOR_GRAYSCALE",
        "chroma_uv": "CHROMA_UV_MANIPULATION",
    }
    for r in results:
        if not r.passed:
            violations.append(
                ValidityViolation(
                    code=code_map[r.name],
                    detail=r.detail,
                    measured=r.measure,
                    limit=r.limit,
                )
            )
    return results, violations
