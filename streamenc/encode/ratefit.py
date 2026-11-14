"""Closed-loop rate fit: encode an arm at ~99% of the streaming budget.

Encoders under/overshoot the requested bitrate by up to ~25% on hard content, so the
request is rescaled from the measured average bitrate until a streaming-valid encode
lands in [FIT_LOW, 1.01] of the budget (or attempts run out; best valid one is kept).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from streamenc.encoders.arms import FFPROBE, build_commands
from streamenc.probe import probe_video
from streamenc.probe.ffprobe_meta import safe_gop
from streamenc.util.subprocess_run import run_command
from streamenc.validate.official_streaming import (
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)

START_AIM = 0.985
FIT_GOAL = 0.99
FIT_LOW = 0.97
MAX_FIT_ATTEMPTS = 5
# Fractional-CRF search (constant-quality arms). SVT-AV1 >= 3.0 accepts quarter-step CRF via
# ``svtav1-params crf=``; one whole CRF step is ~12-15 % of bitrate below CRF 58 and steeper above.
# Accept a valid encode at >= 97 % of budget (same as VBR) and keep the highest valid use.
CRF_START = 45.0
CRF_FIT_LOW = 0.97
CRF_STEP_LOG = 0.14  # default ln(bitrate ratio) per CRF step until two points >= 0.75 apart are measured
CRF_QUANT = 0.25
CRF_MIN, CRF_MAX = 1.0, 63.0
CRF_MAX_FIT_ATTEMPTS = 7  # CRF arms: v7 labels ran out at 5 on 56 trials (easy clips at CRF < 8)
# Downward extrapolation below CRF 30 is damped: ln(bitrate) steepens sharply below CRF ~8
# (ai_easy_13 pilot: secant from CRF 23.25 jumped to CRF 1.0 = 4.4x budget; true answer 5.25).
CRF_EXTRAP_DAMP = 0.75
PEAK_RATIO = 1.5
RATE_CODES = ("STREAMING_BITRATE_EXCEEDED", "STREAMING_PEAK_EXCEEDED")
# Share of the validator's documented 1 % average-bitrate slack the runtime may use (MINING.md: "the average bitrate
# stays within the budget, with 1 % slack"; vidaio/scoring/streaming.py limit = target * 1.01). Our packet measurement
# is bit-identical to the validator's ffprobe backend (2026-11-12: 22 outputs, avg and peak diff 0.0). The 2 s peak
# rule has no slack and stays exact. 0 = old behaviour (cap at 1.000 x budget).
# Default 0.008 since 2026-11-12 (user chose option A; e2e r6_*_S: practice 0.7628 -> 0.7633, held-out pooled
# 0.6300 -> 0.6310, no NEG fallbacks, every list up).
RATE_SLACK = float(os.environ.get("STREAMENC_RATE_SLACK", "0.008"))


def budget_use(avg_kbps: float, peak_kbps: float | None, target: float, peak_ratio: float = PEAK_RATIO) -> float:
    """Binding share of the streaming budget: max(avg / target, 2 s peak / (peak_ratio x target)).

    The CRF search solves on this instead of the average alone, so peak-limited clips (average well
    under budget, 2 s peak at the 1.5x cap) step to the right CRF and stop when the peak binds
    (v7 labels: 101 trials failed on the peak alone, 58 more kept searching under a binding peak).
    """
    return max(avg_kbps / (target * (1.0 + RATE_SLACK)), float(peak_kbps or 0.0) / (peak_ratio * target))


def _quant(x: float) -> float:
    return round(round(x / CRF_QUANT) * CRF_QUANT, 2)


def default_slope(crf: float) -> float:
    """ln(bitrate) per CRF step as a function of CRF (SVT-AV1 4.3, measured on the label clips):
    ~0.2-0.3 below CRF 6 (v7 fits: scr_hard_06 0.19-0.21, ai_easy_13 0.20-0.32, rnd_easy_02_1080 0.27,
    ai_easy_12_1080 0.29), ~0.04 around CRF 15, ~0.07 around 35-45, ~0.14 around 50-55, ~0.25 at 59-63."""
    knots = ((0.0, 0.25), (5.0, 0.22), (10.0, 0.07), (15.0, 0.04), (35.0, 0.07), (50.0, 0.12), (55.0, 0.14),
             (59.0, 0.25), (63.0, 0.25))
    for (c1, s1), (c2, s2) in zip(knots, knots[1:]):
        if crf <= c2:
            return s1 + (s2 - s1) * (crf - c1) / (c2 - c1)
    return knots[-1][1]


def solve_crf(points: list[tuple[float, float]], aim: float = 0.985, slope_hint: float | None = None) -> float:
    """Continuous CRF estimate reaching ``aim`` budget use from measured (crf, util) points.

    With points on both sides of ``aim`` interpolate ln(util) linearly between the two closest
    bracketing points. Otherwise extrapolate from the nearest point with the secant slope of the two
    points nearest the target (>= 0.5 CRF apart); with a single point use ``slope_hint`` (e.g. the
    slope measured on a pilot arm of the same clip) or :func:`default_slope`.
    """
    import math

    pts = [(c, max(u, 1e-3)) for c, u in points]
    if not pts:
        return CRF_START
    over = [p for p in pts if p[1] > aim]
    under = [p for p in pts if p[1] <= aim]
    if over and under:
        (c1, u1) = max(over, key=lambda p: p[0])   # highest CRF that is still over
        (c2, u2) = min(under, key=lambda p: p[0])  # lowest CRF that is under
        if c2 > c1 and u1 != u2:
            t = (math.log(u1) - math.log(aim)) / (math.log(u1) - math.log(u2))
            return max(CRF_MIN, min(CRF_MAX, c1 + t * (c2 - c1)))
    ranked = sorted(pts, key=lambda p: abs(math.log(p[1] / aim)))
    crf, util = ranked[0]
    slope, at = (slope_hint, crf) if slope_hint else (None, crf)
    for c1, u1 in ranked[1:]:
        if abs(c1 - crf) >= 0.5:
            est = abs(math.log(u1 / util) / (crf - c1))
            if 0.02 <= est <= 0.6:
                slope, at = est, 0.5 * (crf + c1)
            break
    return extrapolate_crf(crf, util, aim, slope=slope, slope_at=at)


def extrapolate_crf(crf: float, util: float, aim: float, *, slope: float | None = None,
                    slope_at: float | None = None, step: float = 0.05) -> float:
    """CRF where use reaches ``aim`` from one point, integrating the CRF-dependent slope curve
    (:func:`default_slope`) over the jump, scaled to a measured ``slope`` at ``slope_at`` (x0.5..2).
    A constant slope taken at the start point overshoots badly when the jump crosses the steep
    low-CRF region (e2e: pilot CRF 22.25 -> 1.0 = 16.6x budget; final 7.75 -> 1.0 = 4.7x)."""
    import math

    k = 1.0
    if slope:
        k = min(2.0, max(0.5, slope / default_slope(slope_at if slope_at is not None else crf)))
    need = math.log(max(util, 1e-3) / aim)  # > 0: too many bits -> raise CRF
    c, acc = float(crf), 0.0
    direction = 1.0 if need > 0 else -1.0
    need = abs(need)
    while acc < need:
        nxt = c + direction * step
        if nxt < CRF_MIN or nxt > CRF_MAX:
            return max(CRF_MIN, min(CRF_MAX, nxt))
        acc += k * default_slope(0.5 * (c + nxt)) * step
        c = nxt
    return max(CRF_MIN, min(CRF_MAX, c))


def local_slope(points: list[tuple[float, float]], aim: float = 0.985) -> float | None:
    """Secant ln(util)/CRF slope of the two points nearest ``aim`` (None if unavailable)."""
    import math

    pts = sorted(((c, max(u, 1e-3)) for c, u in points), key=lambda p: abs(math.log(p[1] / aim)))
    for c1, u1 in pts[1:]:
        if abs(c1 - pts[0][0]) >= 0.5:
            est = abs(math.log(u1 / pts[0][1]) / (pts[0][0] - c1))
            return est if 0.02 <= est <= 0.6 else None
    return None


def _next_crf(points: list[tuple[float, float]], codes: list[str],
              over_flags: list[bool] | None = None) -> float | None:
    """Next CRF to try from the measured (crf, budget use) points; None when nothing new can be tried.

    ``over_flags[i]``: point i broke the rate/peak gate (default: use > 1, plus ``codes`` for the last
    point). The next CRF stays strictly inside the tightest (over, valid) bracket; once the bracket has
    no untried quarter step left the search stops instead of spending encodes outside it.
    """
    crf, util = points[-1]
    if over_flags is None:
        over_flags = [u > 1.0 for _, u in points]
        over_flags[-1] = over_flags[-1] or any(c in codes for c in RATE_CODES)
    over = over_flags[-1]
    # Over budget: aim a little lower so the (convex) curve does not leave us asymptotically just above
    # the line; a valid landing anywhere in [0.97, 1.0] ends the search.
    nxt = _quant(solve_crf(points, 0.975 if over else 0.985))
    if over and nxt <= crf:
        nxt = _quant(crf + CRF_QUANT)
    if not over and nxt >= crf:
        nxt = _quant(crf - CRF_QUANT)
    hi_over = max((c for (c, _), o in zip(points, over_flags) if o), default=None)
    lo_ok = min((c for (c, _), o in zip(points, over_flags) if not o), default=None)
    bracket = hi_over is not None and lo_ok is not None and lo_ok > hi_over
    if not bracket and not over and crf < 30.0:
        nxt = _quant(crf - max(CRF_QUANT, CRF_EXTRAP_DAMP * (crf - nxt)))
    nxt = max(CRF_MIN, min(CRF_MAX, nxt))
    tried = {c for c, _ in points}
    if bracket:
        lo, hi = _quant(hi_over + CRF_QUANT), _quant(lo_ok - CRF_QUANT)
        if lo > hi + 1e-9:
            return None  # adjacent quarter steps: over at hi_over, valid at lo_ok -> done
        if len(over_flags) >= 2 and over_flags[-1] == over_flags[-2]:
            # Interpolation stalls on the convex low-CRF curve (scr_hard_06: 3.75 -> 3.25 -> 3.0 -> 2.75,
            # one quarter step per encode): two landings on the same side -> bisect the bracket.
            nxt = _quant(0.5 * (hi_over + lo_ok))
        return min(max(nxt, lo), hi)  # no tried point lies strictly inside the bracket
    step = CRF_QUANT if over else -CRF_QUANT
    while nxt in tried:
        nxt = _quant(nxt + step)
        if nxt < CRF_MIN or nxt > CRF_MAX:
            return None
    return nxt


def encode_rate_fit(
    arm: dict[str, Any],
    input_path: str,
    output_path: str,
    *,
    vf: str | None = None,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    max_attempts: int | None = None,
    start_aim: float = START_AIM,
    crf_hint: float | None = None,
    encode_input: str | None = None,
) -> dict[str, Any]:
    """``crf_hint``: starting CRF for ``rc == "crf"`` arms (e.g. the CRF a pilot arm landed on;
    CRF transfers between SVT presets 2-6 within ~0.3). Falls back to ``arm["crf_start"]``/CRF_START.
    ``encode_input``: byte-identical decoded copy of ``input_path`` to feed the encoder (raw cache);
    geometry, GOP and validation still come from ``input_path``."""
    meta = probe_video(input_path, count_frames=True)
    from streamenc.validate.vmaf import relabel_vf  # 2026-11-13 colour fix (untagged sources are relabelled, not converted)
    relabel = relabel_vf(meta.pix_fmt, meta.color_space, meta.color_range)
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps,
                                   bits_per_pixel=bits_per_pixel)
    gop = safe_gop(input_path, meta.fps, max_keyframe_interval)
    if vf and "{" in vf:  # {fps}: exact source rate, for prefilters that hold frames (2026-11-10)
        vf = vf.format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction))
    out_dir = Path(output_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".ratefit-", dir=out_dir))
    res: dict[str, Any] = {"ok": False, "passed_streaming": False, "attempts": 0, "encode_seconds": 0.0,
                           "util": None, "aim": None, "avg_kbps": None, "peak_kbps": None, "codes": [],
                           "error": "", "target_kbps": target, "frames": meta.nb_frames, "history": [],
                           "cpu_seconds": 0.0, "enc_cpu_seconds": None, "crf_estimate": None}
    aim, best, best_util = start_aim, None, -1e9
    crf_mode = arm.get("rc") == "crf"
    if max_attempts is None:
        max_attempts = CRF_MAX_FIT_ATTEMPTS if crf_mode else MAX_FIT_ATTEMPTS
    over_flags: list[bool] = []
    crf = _quant(float(crf_hint if crf_hint is not None else arm.get("crf_start", CRF_START)))
    crf = max(CRF_MIN, min(CRF_MAX, crf))
    crf_points: list[tuple[float, float]] = []  # (crf, util) measured so far
    try:
        for i in range(1, max_attempts + 1):
            out = str(work / f"out{i}.mp4")
            t0 = time.perf_counter()
            err = ""
            if crf_mode:
                aim = float(crf)  # recorded as the "aim" of a CRF trial
            att_cpu = 0.0
            for cmd in build_commands(arm, encode_input or input_path, out, target * (1.0 if crf_mode else aim), gop,
                                      str(work / "pass"), vf, crf=crf if crf_mode else None, pre_vf=relabel):
                r = run_command(cmd, timeout=7200)
                att_cpu += r.cpu_seconds
                if not r.ok:
                    err = (r.stderr or r.stdout or "")[-300:].replace("\n", " ")
                    break
            res["encode_seconds"] += time.perf_counter() - t0
            res["cpu_seconds"] += att_cpu
            res["attempts"] = i
            if err or not os.path.exists(out):
                res["error"] = err or "no output"
                break
            m = measure_streaming(read_packets_ffprobe(out, ffprobe=FFPROBE), fps=meta.fps,
                                  frames=meta.nb_frames, peak_window_seconds=2.0)
            vs = streaming_violations(m, target_kbps=target, peak_ratio=1.5, peak_window_seconds=2.0,
                                      max_keyframe_interval_seconds=max_keyframe_interval, fps=meta.fps)
            codes = sorted({v.code for v in vs})
            util = m.avg_kbps / target
            use = budget_use(m.avg_kbps, m.peak_kbps, target)
            res["history"].append({"aim": round(aim, 4), "util": round(util, 4), "use": round(use, 4),
                                   "codes": codes, "cpu_seconds": round(att_cpu, 3)})
            # CRF arms keep the lowest valid CRF (most bits = best quality, robust to the small
            # non-monotonic util wiggles between adjacent quarter steps); VBR keeps the highest util.
            key = -float(aim) if crf_mode else util
            if not vs and key > best_util:
                best, best_util = (out, aim, m, att_cpu), key
            if "STREAMING_KEYFRAME_INTERVAL" in codes:
                res["codes"] = codes
                break
            if crf_mode:
                crf_points.append((crf, use))
                over_flags.append(use > 1.0 or any(c in codes for c in RATE_CODES))
                res["crf_estimate"] = round(solve_crf(crf_points), 3)
                if not vs and use >= CRF_FIT_LOW:
                    break
                crf = _next_crf(crf_points, codes, over_flags)
                if crf is None:
                    break
                continue
            if not vs and util >= FIT_LOW:
                break
            if "STREAMING_PEAK_EXCEEDED" in codes and "STREAMING_BITRATE_EXCEEDED" not in codes:
                aim *= 0.93
            else:
                aim *= FIT_GOAL / max(util, 1e-3)
            aim = min(aim, 2.5)
        if best is not None:
            out, aim, m, enc_cpu = best
            res["enc_cpu_seconds"] = enc_cpu
            os.replace(out, output_path)
            os.chmod(output_path, 0o644)
            res.update({"ok": True, "passed_streaming": True, "aim": round(aim, 4),
                        "avg_kbps": m.avg_kbps, "peak_kbps": m.peak_kbps,
                        "util": m.avg_kbps / target, "codes": [], "error": ""})
        elif not res["codes"] and res["history"]:
            res["codes"] = res["history"][-1]["codes"]
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)
