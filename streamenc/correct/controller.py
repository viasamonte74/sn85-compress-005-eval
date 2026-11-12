"""Bounded selective correction for streaming gate failures.

Keeps the last known-valid encode as fallback and caps retries.
Uses competition score heuristics to decide whether an alternate
quality attempt is worth the extra runtime.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from streamenc.encoders.generic import EncoderConfig, build_encode_argv
from streamenc.probe import probe_video
from streamenc.probe.ffprobe_meta import safe_gop
from streamenc.util.logging import get_logger
from streamenc.util.subprocess_run import run_command
from streamenc.validate.official_streaming import (
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)

log = get_logger("streamenc.correct")

# Bounded aims (avg/peak pressure). Default is slightly conservative.
AIM_LADDER = (0.975, 0.94, 0.90, 0.85, 0.78, 0.70)
# Hard-path: start high but not so high that peak gates fail repeatedly.
# (0.995 caused pass-rate drops on some hard renders in A/B.)
HARD_AIM_LADDER = (0.970, 0.940, 0.900, 0.850, 0.780, 0.700)
MAX_ATTEMPTS = 8


@dataclass
class CorrectionOutcome:
    ok: bool
    output_path: str
    config_id: str
    attempts: int
    encode_seconds: float
    avg_kbps: float | None = None
    peak_kbps: float | None = None
    aim_fraction: float | None = None
    violation_codes: list[str] = field(default_factory=list)
    fallback_used: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


def _clone_cfg(cfg: EncoderConfig, **param_updates: Any) -> EncoderConfig:
    params = dict(cfg.params)
    params.update(param_updates)
    suffix = param_updates.get("_tag", "corr")
    params.pop("_tag", None)
    return EncoderConfig(
        config_id=f"{cfg.config_id}+{suffix}",
        codec=cfg.codec,
        stage=cfg.stage,
        params=params,
        notes=f"{cfg.notes}; corrected",
    )


def _apply_violation_fix(
    cfg: EncoderConfig,
    codes: set[str],
    *,
    attempt_idx: int,
) -> EncoderConfig:
    """Specialize encoder params for the observed streaming violation."""
    updates: dict[str, Any] = {"_tag": f"fix{attempt_idx}"}
    if "STREAMING_PEAK_EXCEEDED" in codes:
        # Tighter VBV / peak envelope.
        updates["maxrate_fraction"] = min(float(cfg.params.get("maxrate_fraction", 1.30)), 1.15)
        updates["bufsize_fraction"] = min(float(cfg.params.get("bufsize_fraction", 2.00)), 1.50)
        updates["rc_lookahead"] = max(int(cfg.params.get("rc_lookahead", 8)), 16)
    if "STREAMING_BITRATE_EXCEEDED" in codes:
        updates["maxrate_fraction"] = min(float(cfg.params.get("maxrate_fraction", 1.30)), 1.20)
        updates["bufsize_fraction"] = min(float(cfg.params.get("bufsize_fraction", 2.00)), 1.75)
    if "STREAMING_KEYFRAME_INTERVAL" in codes:
        updates["strict_gop"] = True
        updates["no_scenecut"] = True
        # Force shorter effective GOP via svt keyint or NVENC g (handled by caller gop).
        if cfg.codec == "libsvtav1":
            svt = dict(cfg.params.get("svtav1_params") or {})
            svt["scd"] = 0
            updates["svtav1_params"] = svt
    if "STREAMING_PEAK_EXCEEDED" in codes and cfg.codec == "av1_nvenc":
        # AQ can inflate local peaks on complex frames.
        updates["spatial_aq"] = False
        updates["temporal_aq"] = False
    return _clone_cfg(cfg, **updates)


class CorrectionController:
    """Encode with selective, bounded corrections and a valid-output fallback."""

    def __init__(
        self,
        *,
        bits_per_pixel: float = 0.05,
        max_keyframe_interval: float = 2.0,
        peak_window: float = 2.0,
        peak_ratio: float = 1.5,
        max_attempts: int = MAX_ATTEMPTS,
        quality_alt: EncoderConfig | None = None,
        time_budget_seconds: float | None = None,
        aim_ladder: tuple[float, ...] | None = None,
        fill_up: bool = False,
        fill_up_util_threshold: float = 0.90,
    ) -> None:
        self.bits_per_pixel = bits_per_pixel
        self.max_keyframe_interval = max_keyframe_interval
        self.peak_window = peak_window
        self.peak_ratio = peak_ratio
        self.max_attempts = max_attempts
        self.quality_alt = quality_alt
        self.time_budget_seconds = time_budget_seconds
        self.aim_ladder = aim_ladder or AIM_LADDER
        self.fill_up = fill_up
        self.fill_up_util_threshold = fill_up_util_threshold

    def encode(
        self,
        input_path: str,
        output_path: str,
        primary: EncoderConfig,
    ) -> CorrectionOutcome:
        meta = probe_video(input_path, count_frames=True)
        target = streaming_target_kbps(
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            bits_per_pixel=self.bits_per_pixel,
        )
        gop = safe_gop(input_path, meta.fps, self.max_keyframe_interval)
        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        os.makedirs(out_dir, exist_ok=True)

        best_valid: str | None = None
        best_meta: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        encode_seconds = 0.0
        attempts = 0
        cfg = primary
        aim_idx = 0
        tried_quality_alt = False
        tried_fill_up = False
        ladder = self.aim_ladder
        t_start = time.perf_counter()

        while attempts < self.max_attempts:
            if self.time_budget_seconds is not None:
                if time.perf_counter() - t_start >= self.time_budget_seconds:
                    break

            aim = ladder[min(aim_idx, len(ladder) - 1)]
            attempts += 1
            fd, partial = tempfile.mkstemp(prefix=".corr-partial-", suffix=".mp4", dir=out_dir)
            os.close(fd)
            kbps = target * aim
            argv = build_encode_argv(
                cfg=cfg,
                input_path=input_path,
                output_path=partial,
                kbps=kbps,
                gop=gop,
                pix_fmt=meta.pix_fmt if meta.pix_fmt in {"yuv420p", "yuv420p10le", "p010le"} else "yuv420p",
                color_space=meta.color_space or "bt709",
                color_primaries=meta.color_primaries or "bt709",
                color_transfer=meta.color_transfer or "bt709",
                color_range=meta.color_range or "tv",
            )
            t0 = time.perf_counter()
            proc = run_command(argv, timeout=3600)
            dt = time.perf_counter() - t0
            encode_seconds += dt

            rec: dict[str, Any] = {
                "attempt": attempts,
                "config_id": cfg.config_id,
                "aim": aim,
                "encode_seconds": dt,
                "ffmpeg_ok": proc.ok,
            }
            if not proc.ok or not os.path.exists(partial) or os.path.getsize(partial) == 0:
                rec["error"] = (proc.stderr or proc.stdout or "empty")[:240]
                history.append(rec)
                try:
                    os.unlink(partial)
                except OSError:
                    pass
                aim_idx = min(aim_idx + 1, len(ladder) - 1)
                continue

            try:
                packets = read_packets_ffprobe(partial)
                meas = measure_streaming(
                    packets,
                    fps=meta.fps,
                    frames=meta.nb_frames,
                    peak_window_seconds=self.peak_window,
                )
                vs = streaming_violations(
                    meas,
                    target_kbps=target,
                    peak_ratio=self.peak_ratio,
                    peak_window_seconds=self.peak_window,
                    max_keyframe_interval_seconds=self.max_keyframe_interval,
                    fps=meta.fps,
                )
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"measure: {exc}"
                history.append(rec)
                try:
                    os.unlink(partial)
                except OSError:
                    pass
                continue

            codes = {v.code for v in vs}
            util = float(meas.avg_kbps) / max(target, 1e-6)
            rec.update(
                {
                    "avg_kbps": meas.avg_kbps,
                    "peak_kbps": meas.peak_kbps,
                    "util": util,
                    "codes": sorted(codes),
                }
            )
            history.append(rec)

            if not codes:
                # Valid streaming candidate — promote to best / final.
                os.chmod(partial, 0o644)
                if best_valid and best_valid != partial and os.path.exists(best_valid):
                    try:
                        os.unlink(best_valid)
                    except OSError:
                        pass
                # Keep stable fallback path then publish.
                fallback = os.path.join(out_dir, f".corr-valid-{os.getpid()}.mp4")
                os.replace(partial, fallback)
                best_valid = fallback
                best_meta = {
                    "config_id": cfg.config_id,
                    "aim": aim,
                    "avg_kbps": meas.avg_kbps,
                    "peak_kbps": meas.peak_kbps,
                    "util": util,
                }
                # Fill-up: if we left budget on the table, try a higher aim once.
                if (
                    self.fill_up
                    and not tried_fill_up
                    and util < self.fill_up_util_threshold
                    and aim_idx > 0
                ):
                    tried_fill_up = True
                    aim_idx = max(0, aim_idx - 1)
                    continue
                if (
                    self.fill_up
                    and not tried_fill_up
                    and util < self.fill_up_util_threshold
                    and aim < ladder[0]
                ):
                    # First success was already at top aim but util still low — nothing to fill.
                    tried_fill_up = True
                # Optional quality alt: only if we still have budget and alt promised.
                if (
                    not tried_quality_alt
                    and self.quality_alt is not None
                    and self.quality_alt.config_id.split("+")[0] != primary.config_id.split("+")[0]
                    and self._worth_quality_retry(encode_seconds, meta.nb_frames)
                ):
                    tried_quality_alt = True
                    cfg = self.quality_alt
                    aim_idx = 0
                    continue
                break

            # Invalid — delete partial, adjust, retry.
            try:
                os.unlink(partial)
            except OSError:
                pass
            cfg = _apply_violation_fix(cfg, codes, attempt_idx=attempts)
            if "STREAMING_BITRATE_EXCEEDED" in codes or "STREAMING_PEAK_EXCEEDED" in codes:
                aim_idx = min(aim_idx + 1, len(ladder) - 1)
            # Keyframe issues: shrink GOP one step.
            if "STREAMING_KEYFRAME_INTERVAL" in codes:
                gop = max(1, int(gop * 0.75))

        if best_valid and os.path.exists(best_valid):
            os.chmod(best_valid, 0o644)
            os.replace(best_valid, output_path)
            return CorrectionOutcome(
                ok=True,
                output_path=output_path,
                config_id=str(best_meta.get("config_id", primary.config_id)),
                attempts=attempts,
                encode_seconds=encode_seconds,
                avg_kbps=best_meta.get("avg_kbps"),
                peak_kbps=best_meta.get("peak_kbps"),
                aim_fraction=best_meta.get("aim"),
                fallback_used=attempts > 1,
                history=history,
            )

        return CorrectionOutcome(
            ok=False,
            output_path="",
            config_id=primary.config_id,
            attempts=attempts,
            encode_seconds=encode_seconds,
            violation_codes=list(history[-1].get("codes", [])) if history else [],
            history=history,
            reason="no valid encode within attempt/time budget",
        )

    @staticmethod
    def _worth_quality_retry(spent_seconds: float, frames: int) -> bool:
        """Heuristic: only spend more if we are still well under speed saturation."""
        # Speed score saturates near ~114s / 3428 frames. Per-clip share ~ frames/3428 * 114.
        share = max(5.0, (frames / 3428.0) * 114.0)
        return spent_seconds < share * 0.5
