"""Single-item pipeline: features → decision → filter → corrected encode."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streamenc.correct.controller import HARD_AIM_LADDER, CorrectionController, CorrectionOutcome
from streamenc.decision.catalog import baseline_config, get_config
from streamenc.decision.hard_adapt import hard_adapt_plan
from streamenc.decision.rules import decide_rules
from streamenc.encoders.generic import EncoderConfig
from streamenc.encoders.nvenc_probe import CPU_FALLBACK_DEFAULT, CPU_FALLBACK_HARD, nvenc_available
from streamenc.features.analyze import analyze_video
from streamenc.filters.policy import choose_prefilter, filter_vf
from streamenc.util.logging import get_logger

log = get_logger("streamenc.pipeline")


@dataclass
class ItemResult:
    input: str
    output: str
    ok: bool
    clip_id: str
    config_id: str
    rule_id: str
    prefilter: str
    feature_seconds: float
    encode_seconds: float
    total_seconds: float
    attempts: int
    avg_kbps: float | None = None
    peak_kbps: float | None = None
    frames: int = 0
    reason: str = ""
    fallback_used: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _with_prefilter(cfg: EncoderConfig, prefilter: str) -> EncoderConfig:
    vf = filter_vf(prefilter)
    if not vf:
        return cfg
    params = dict(cfg.params)
    params["vf"] = vf
    return EncoderConfig(
        config_id=f"{cfg.config_id}+{prefilter}",
        codec=cfg.codec,
        stage=cfg.stage,
        params=params,
        notes=f"{cfg.notes}; prefilter={prefilter}",
    )


def _controller(
    *,
    hard_path: bool,
    bits_per_pixel: float,
    max_keyframe_interval: float,
    peak_window: float,
    peak_ratio: float,
    remaining: float | None,
) -> CorrectionController:
    return CorrectionController(
        bits_per_pixel=bits_per_pixel,
        max_keyframe_interval=max_keyframe_interval,
        peak_window=peak_window,
        peak_ratio=peak_ratio,
        quality_alt=None,
        time_budget_seconds=remaining,
        aim_ladder=HARD_AIM_LADDER if hard_path else None,
        fill_up=hard_path,
        fill_up_util_threshold=0.92 if hard_path else 0.90,
    )


def _rank_outcome(outcome: CorrectionOutcome) -> tuple:
    """Prefer valid encode, then higher util (budget use), then fewer attempts."""
    util = 0.0
    if outcome.history:
        for h in reversed(outcome.history):
            if h.get("util") is not None:
                util = float(h["util"])
                break
    # Prefer util just under 1.0; penalize overshoot.
    util_score = util if util <= 1.01 else (1.01 - (util - 1.01))
    return (1 if outcome.ok else 0, util_score, -outcome.attempts)


def process_item(
    input_path: str,
    output_path: str,
    *,
    prefer_quality: bool = True,
    allow_filter: bool = True,
    use_decision: bool = True,
    hard_adapt: bool = True,
    hard_dual: bool = False,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    peak_window: float = 2.0,
    peak_ratio: float = 1.5,
    time_budget_seconds: float | None = None,
    max_samples: int = 24,
) -> ItemResult:
    """Process one FFV1 input into a competition output file."""
    t0 = time.perf_counter()
    clip_id = Path(input_path).name
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    feature_seconds = 0.0
    rule_id = "BASELINE"
    prefilter = "none"
    primary = baseline_config()
    hard_path = False
    features = None
    adapt_extras: dict[str, Any] = {}

    if use_decision:
        analysis = analyze_video(
            input_path,
            mode="sampled",
            max_samples=max_samples,
            max_width=640,
        )
        feature_seconds = analysis.total_seconds
        features = analysis.features
        decision = decide_rules(features, prefer_quality=prefer_quality)
        rule_id = decision.rule_id
        primary = decision.encoder_config()
        hard_path = bool(decision.hard_path)
        if hard_path and hard_adapt:
            # Hard-adapt owns prefilter (regime denoise / none).
            prefilter = decision.prefilter if allow_filter else "none"
        else:
            prefilter = choose_prefilter(features, allow_filter=allow_filter)
            if prefilter == "none" and decision.prefilter != "none" and allow_filter:
                prefilter = decision.prefilter
    else:
        primary = baseline_config()
        rule_id = "BASELINE"

    # NVENC may be absent in the sandbox (no libnvidia-encode / no hardware encoder):
    # probe once per process and fall back to SVT-AV1 instead of failing the item.
    if primary.codec.endswith("nvenc") and not nvenc_available(primary.codec):
        cpu_id = CPU_FALLBACK_HARD if hard_path else CPU_FALLBACK_DEFAULT
        adapt_extras["nvenc_fallback"] = {"from": primary.config_id, "to": cpu_id}
        primary = get_config(cpu_id)
        rule_id = f"{rule_id}+cpu_fallback"

    def _remaining() -> float | None:
        if time_budget_seconds is None:
            return None
        return max(1.0, time_budget_seconds - (time.perf_counter() - t0))

    encode_seconds = 0.0
    attempts = 0
    history: list[dict[str, Any]] = []
    best: tuple[CorrectionOutcome, str, str] | None = None  # outcome, prefilter, tag

    def _run(cfg: EncoderConfig, pf: str, tag: str, dest: str) -> CorrectionOutcome:
        nonlocal encode_seconds, attempts
        ctrl = _controller(
            hard_path=hard_path,
            bits_per_pixel=bits_per_pixel,
            max_keyframe_interval=max_keyframe_interval,
            peak_window=peak_window,
            peak_ratio=peak_ratio,
            remaining=_remaining(),
        )
        outcome = ctrl.encode(input_path, dest, _with_prefilter(cfg, pf))
        encode_seconds += outcome.encode_seconds
        attempts += outcome.attempts
        history.append({"tag": tag, "prefilter": pf, "ok": outcome.ok, "config_id": outcome.config_id, "attempts": outcome.attempts})
        return outcome

    # Primary encode
    outcome = _run(primary, prefilter, "primary", output_path)
    # NVENC answered the probe but every real attempt failed to start (driver/session limit):
    # fall back to the CPU configuration instead of returning no output.
    if (
        not outcome.ok
        and primary.codec.endswith("nvenc")
        and outcome.history
        and all(not h.get("ffmpeg_ok", True) for h in outcome.history)
    ):
        cpu_id = CPU_FALLBACK_HARD if hard_path else CPU_FALLBACK_DEFAULT
        adapt_extras["nvenc_fallback"] = {"from": primary.config_id, "to": cpu_id, "after_failure": True}
        primary = get_config(cpu_id)
        rule_id = f"{rule_id}+cpu_fallback"
        outcome = _run(primary, prefilter, "primary", output_path)
    best = (outcome, prefilter, "primary")

    # Hard-adapt: alt candidate + salvage denoise when primary fails streaming.
    if hard_path and hard_adapt and features is not None and allow_filter:
        plan = hard_adapt_plan(features)
        adapt_extras["regime"] = plan.regime
        adapt_extras["plan"] = {
            "primary_config_id": plan.primary_config_id,
            "primary_prefilter": plan.primary_prefilter,
            "alt_config_id": plan.alt_config_id,
            "salvage_prefilter": plan.salvage_prefilter,
        }

        candidates: list[tuple[str, str, str]] = []  # config_id, prefilter, tag
        # Dual-candidate (quality): optional; costs extra encodes.
        if hard_dual and plan.alt_config_id:
            candidates.append((plan.alt_config_id, plan.alt_prefilter, "alt"))
        # Salvage: only when primary failed streaming — try strong denoise / alt arm.
        if not outcome.ok:
            if plan.salvage_prefilter != prefilter:
                candidates.append((plan.primary_config_id, plan.salvage_prefilter, "salvage"))
            if plan.alt_config_id:
                candidates.append((plan.alt_config_id, plan.salvage_prefilter, "salvage_alt"))

        for cid, pf, tag in candidates:
            # Skip duplicate of primary.
            if cid == primary.config_id and pf == prefilter:
                continue
            try:
                cfg = get_config(cid)
            except KeyError:
                continue
            fd, tmp = tempfile.mkstemp(prefix=f".hard-{tag}-", suffix=".mp4", dir=out_dir)
            os.close(fd)
            try:
                cand = _run(cfg, pf, tag, tmp)
                if _rank_outcome(cand) > _rank_outcome(best[0]):
                    # Promote winner to output_path.
                    if cand.ok and os.path.exists(tmp):
                        os.replace(tmp, output_path)
                        best = (cand, pf, tag)
                        prefilter = pf
                    elif not best[0].ok and cand.ok:
                        os.replace(tmp, output_path)
                        best = (cand, pf, tag)
                        prefilter = pf
            finally:
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

        adapt_extras["winner_tag"] = best[2]

    outcome = best[0]
    prefilter = best[1]
    total = time.perf_counter() - t0

    for p in Path(out_dir).glob(".corr-partial-*"):
        try:
            p.unlink()
        except OSError:
            pass
    for p in Path(out_dir).glob(f".corr-valid-{os.getpid()}.mp4"):
        try:
            p.unlink()
        except OSError:
            pass
    for p in Path(out_dir).glob(".hard-*"):
        try:
            p.unlink()
        except OSError:
            pass

    frames = 0
    try:
        from streamenc.probe import probe_video

        frames = probe_video(input_path, count_frames=True).nb_frames
    except Exception:  # noqa: BLE001
        pass

    return ItemResult(
        input=input_path,
        output=output_path if outcome.ok else "",
        ok=outcome.ok,
        clip_id=clip_id,
        config_id=outcome.config_id,
        rule_id=rule_id,
        prefilter=prefilter,
        feature_seconds=feature_seconds,
        encode_seconds=encode_seconds,
        total_seconds=total,
        attempts=attempts,
        avg_kbps=outcome.avg_kbps,
        peak_kbps=outcome.peak_kbps,
        frames=frames,
        reason=outcome.reason,
        fallback_used=outcome.fallback_used or (adapt_extras.get("winner_tag") not in (None, "primary")),
        extras={"history": history, "hard_adapt": adapt_extras},
    )
