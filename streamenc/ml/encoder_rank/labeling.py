"""Strict offline labeling: only gate-valid, highest-score settings become winners (policy v3).

Rules
-----
1. Sweep the **full** v3 catalog (all encoder families) for every clip.
2. Encode each arm with ``encode_rate_fit`` (closed loop to ~99% of the streaming budget),
   which is the same encode path the runtime uses.
3. Score only streaming-valid outputs with full validate (VMAF + VMAF-NEG + perceptual).
4. Winner ∈ {trials | passed_streaming ∧ passed}; maximize (clip_score, VMAF, util).
5. If no gate-valid trial → ``trainable=False`` (do **not** use as CE supervision).
6. Soft ranking targets keep all gate-valid trial scores for the listwise loss.
7. ``efficient_config_id``: fastest gate-valid trial within ``EFFICIENT_VMAF_MARGIN`` of the
   winner, for speed-aware runtime policies.
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from streamenc.decision.hardpath import hard_feature_regime, is_hard_content
from streamenc.encode.ratefit import encode_rate_fit
from streamenc.features.analyze import analyze_video
from streamenc.features.difficulty import difficulty_score
from streamenc.features.extract import FEATURE_NAMES
from streamenc.filters.policy import filter_vf
from streamenc.ml.encoder_rank.catalog import LABEL_CATALOG_IDS, resolve_label, PILOT_LABEL_ID, LABEL_POLICY_VERSION
from streamenc.ml.encoder_rank.schema import empty_label, empty_trial
from streamenc.ml.encoder_rank.splits import assign_split, clip_family, clip_kind, clip_tier
from streamenc.validate.validate_clip import validate_clip

# Minimum margin vs 2nd-best clip_score to mark winner as high-confidence.
SCORE_MARGIN = 0.005
VMAF_MARGIN = 0.5
EFFICIENT_VMAF_MARGIN = 0.5
ARM_WORKERS = int(os.environ.get("LABEL_ARM_WORKERS", "6"))


def ensure_nvenc_lib() -> None:
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc


def util_score(util: float | None) -> float:
    if util is None:
        return -9.0
    u = float(util)
    # Prefer near-full budget without overshoot.
    if u <= 1.0:
        return u
    return 1.0 - min(1.0, u - 1.0)


def rank_key_valid(row: dict[str, Any]) -> tuple:
    """Ranking among gate-valid trials only."""
    score = float(row.get("clip_score") or 0.0)
    vmaf = float(row["vmaf"]) if row.get("vmaf") is not None else -1.0
    return (score, vmaf, util_score(row.get("util")), -float(row.get("encode_seconds") or 1e9))


def encode_and_score(input_path: str, output_path: str, label_id: str,
                     crf_hint: float | None = None, max_attempts: int | None = None,
                     encode_input: str | None = None) -> dict[str, Any]:
    """Rate-fit encode + full competition validation for one catalog entry."""
    arm, prefilter = resolve_label(label_id)
    vf = filter_vf(prefilter) if prefilter != "none" else None
    kw = {"max_attempts": max_attempts} if max_attempts else {}
    enc = encode_rate_fit(arm, input_path, output_path, vf=vf, crf_hint=crf_hint, encode_input=encode_input, **kw)
    result: dict[str, Any] = {
        "ok": bool(enc["ok"]),
        "passed_streaming": bool(enc["passed_streaming"]),
        "passed": False,
        "vmaf": None,
        "vmaf_neg": None,
        "vmaf_delta": None,
        "clip_score": 0.0,
        "util": enc["util"],
        "aim": enc["aim"],
        "avg_kbps": enc["avg_kbps"],
        "peak_kbps": enc["peak_kbps"],
        "encode_seconds": enc["encode_seconds"],
        "cpu_seconds": enc.get("cpu_seconds"),
        "enc_cpu_seconds": enc.get("enc_cpu_seconds"),
        "crf_estimate": enc.get("crf_estimate"),
        "fit_history": " ".join(f"{h['aim']}:{h['util']}" for h in enc.get("history", [])),
        "attempts": enc["attempts"],
        "codes": list(enc["codes"]) + ([f"ERROR:{enc['error'][:80]}"] if enc.get("error") else []),
    }
    if enc["ok"] and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        report = validate_clip(input_path, output_path, run_vmaf=True, run_perceptual=True)
        result["passed"] = bool(report.passed)
        if "vmaf" in report.metrics:
            result["vmaf"] = float(report.metrics["vmaf"])
            result["vmaf_neg"] = float(report.metrics.get("vmaf_neg") or 0.0)
            result["vmaf_delta"] = float(report.metrics.get("vmaf_delta") or 0.0)
            # Official score only if all gates pass.
            result["clip_score"] = float(report.metrics.get("clip_score", 0.0)) if report.passed else 0.0
        if not report.passed:
            result["clip_score"] = 0.0
            result["codes"] = list(result["codes"]) + [v.code for v in report.violations]
    try:
        os.unlink(output_path)
    except OSError:
        pass
    return result


PILOT_MAX_ATTEMPTS = 7
HOPELESS_VMAF = 57.0  # best VMAF below this -> clip cannot score; label the fastest arm
SPEED_LABEL_ID = "svt_crf_p8_t1"


def select_winner(trials: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Pick highest-score gate-valid trial; describe label quality."""
    valid = [t for t in trials if t.get("passed_streaming") and t.get("passed") and t.get("vmaf") is not None]
    quality: dict[str, Any] = {
        "n_valid": len(valid),
        "trainable": False,
        "winner_confidence": 0.0,
        "reject_reason": None,
        "efficient_config_id": None,
    }
    if not valid:
        quality["reject_reason"] = "no_gate_valid_trial"
        return None, quality

    ranked = sorted(valid, key=rank_key_valid, reverse=True)
    winner = ranked[0]
    quality["trainable"] = True
    quality["hopeless"] = False
    best_vmaf = max(float(t["vmaf"]) for t in valid)
    if best_vmaf < HOPELESS_VMAF:
        # No arm can reach the 60-VMAF floor (with margin): the clip scores 0 whatever we do, so the
        # right label is the fastest valid arm (speed term) — the speed arm when it is valid.
        quality["hopeless"] = True
        speed = [t for t in valid if t["config_id"] == SPEED_LABEL_ID]
        winner = speed[0] if speed else min(valid, key=lambda t: float(t.get("encode_seconds") or 1e9))
        quality["efficient_config_id"] = winner["config_id"]
        quality["winner_confidence"] = 1.0
        return winner, quality
    near = [t for t in valid if float(t["vmaf"]) >= float(winner["vmaf"]) - EFFICIENT_VMAF_MARGIN]
    quality["efficient_config_id"] = min(near, key=lambda t: float(t.get("encode_seconds") or 1e9))["config_id"]

    if len(ranked) == 1:
        quality["winner_confidence"] = 1.0
    else:
        second = ranked[1]
        ds = float(winner.get("clip_score") or 0) - float(second.get("clip_score") or 0)
        dv = float(winner["vmaf"]) - float(second["vmaf"])
        if ds > SCORE_MARGIN or (ds >= 0 and dv >= VMAF_MARGIN):
            quality["winner_confidence"] = min(1.0, 0.55 + 10.0 * max(ds, 0) + 0.02 * max(dv, 0))
        else:
            # Tie / weak margin — still trainable but low confidence.
            quality["winner_confidence"] = 0.35
            quality["tie_with"] = second.get("config_id")
    return winner, quality


def source_cache(path: Path) -> tuple[str | None, str | None]:
    """(raw copy, its temp dir) of ``path`` for the encoders, or (None, None)."""
    try:
        from streamenc.encode.crf_pipeline import _raw_source
        from streamenc.probe import probe_video
        from streamenc.probe.ffprobe_meta import packet_pts

        raw, d, _ = _raw_source(str(path), probe_video(str(path), count_frames=True), packet_pts(str(path)))
        return raw, d
    except Exception:  # noqa: BLE001
        return None, None


def run_label_trial(path: Path, lid: str, work_dir: Path, *, crf_hint: float | None = None,
                    max_attempts: int | None = None, encode_input: str | None = None) -> dict[str, Any]:
    """Rate-fit encode + full validation of one catalog entry on one clip, as a trial row."""
    out = work_dir / "full" / lid / f"{path.stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored = encode_and_score(str(path), str(out), lid, crf_hint=crf_hint, max_attempts=max_attempts,
                              encode_input=encode_input)
    _, prefilter = resolve_label(lid)
    return empty_trial(
        clip_id=path.stem,
        config_id=lid,
        config_index=LABEL_CATALOG_IDS.index(lid) if lid in LABEL_CATALOG_IDS else -1,
        prefilter=prefilter,
        stage="full",
        ok=scored["ok"],
        passed_streaming=scored["passed_streaming"],
        passed=scored["passed"],
        vmaf=scored["vmaf"],
        vmaf_neg=scored.get("vmaf_neg"),
        vmaf_delta=scored.get("vmaf_delta"),
        clip_score=scored["clip_score"],
        util=scored["util"],
        aim=scored["aim"],
        avg_kbps=scored["avg_kbps"],
        peak_kbps=scored["peak_kbps"],
        encode_seconds=scored["encode_seconds"],
        cpu_seconds=scored.get("cpu_seconds"),
        enc_cpu_seconds=scored.get("enc_cpu_seconds"),
        crf_estimate=scored.get("crf_estimate"),
        fit_history=scored.get("fit_history"),
        attempts=scored["attempts"],
        codes=",".join(str(c) for c in (scored.get("codes") or [])),
    )


def label_clip(
    path: Path,
    *,
    work_dir: Path,
    catalog_ids: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Full-catalog strict labeling for one clip (arms encoded in parallel)."""
    ensure_nvenc_lib()
    t0 = time.perf_counter()
    analysis = analyze_video(path, mode="sampled", max_samples=24, max_width=640)
    feats = analysis.features
    diff = difficulty_score(feats)
    feat_dict = {k: float(getattr(feats, k)) for k in FEATURE_NAMES}
    regime = hard_feature_regime(feats)
    hard = bool(diff.is_hard or is_hard_content(feats))

    family = clip_family(path.stem, path)
    kind = clip_kind(path)
    tier = clip_tier(path.stem)
    split = assign_split(family)
    ids = list(catalog_ids or LABEL_CATALOG_IDS)

    # Decode the source once for all ~35 encodes of this clip (byte-identical outputs, see crf_pipeline).
    raw, raw_dir = source_cache(path)

    def _one(lid: str, crf_hint: float | None = None, max_attempts: int | None = None) -> dict[str, Any]:
        return run_label_trial(path, lid, work_dir, crf_hint=crf_hint, max_attempts=max_attempts, encode_input=raw)

    # Pilot first: its CRF seeds every other CRF arm (CRF transfers between SVT presets within ~0.3),
    # cutting the per-arm rate search from ~4 attempts to ~1-2.
    trials: list[dict[str, Any]] = []
    hint: float | None = None
    rest = list(ids)
    try:
        if PILOT_LABEL_ID in ids:
            pilot = _one(PILOT_LABEL_ID, None, PILOT_MAX_ATTEMPTS)  # cheap arm: more attempts to seed the rest
            trials.append(pilot)
            if pilot.get("crf_estimate") is not None:
                hint = float(pilot["crf_estimate"])  # continuous estimate at 98.5 % budget, even if the fit failed
            elif pilot.get("ok") and pilot.get("aim") is not None:
                hint = float(pilot["aim"])
            rest = [i for i in ids if i != PILOT_LABEL_ID]
        with ThreadPoolExecutor(ARM_WORKERS) as ex:
            trials += list(ex.map(lambda lid: _one(lid, hint), rest))
    finally:
        if raw_dir:
            shutil.rmtree(raw_dir, ignore_errors=True)
    order = {lid: i for i, lid in enumerate(ids)}
    trials.sort(key=lambda t: order.get(t["config_id"], 10**6))

    winner, quality = select_winner(trials)
    valid = [t for t in trials if t.get("passed") and t.get("passed_streaming")]

    label = empty_label(
        clip_id=path.stem,
        path=str(path),
        kind=kind,
        tier=tier,
        family=family,
        split=split,
        winner_config_id=winner["config_id"] if winner else None,
        winner_config_index=winner["config_index"] if winner else None,
        winner_prefilter=winner["prefilter"] if winner else None,
        winner_aim=winner.get("aim") if winner else None,
        winner_vmaf=winner.get("vmaf") if winner else None,
        winner_clip_score=float(winner.get("clip_score") or 0.0) if winner else 0.0,
        winner_util=winner.get("util") if winner else None,
        winner_passed=bool(winner.get("passed")) if winner else False,
        winner_passed_streaming=bool(winner.get("passed_streaming")) if winner else False,
        n_trials=len(trials),
        n_streaming_pass=sum(1 for t in trials if t.get("passed_streaming")),
        max_vmaf=max((t["vmaf"] for t in valid if t.get("vmaf") is not None), default=None),
        max_clip_score=max((float(t.get("clip_score") or 0.0) for t in valid), default=0.0),
        difficulty_score=float(diff.score),
        is_hard=hard,
        regime=regime,
        label_seconds=time.perf_counter() - t0,
    )
    label["trainable"] = bool(quality["trainable"])
    label["winner_confidence"] = float(quality["winner_confidence"])
    label["n_gate_valid"] = int(quality["n_valid"])
    label["reject_reason"] = quality.get("reject_reason")
    label["tie_with"] = quality.get("tie_with")
    label["efficient_config_id"] = quality.get("efficient_config_id")
    label["label_policy_version"] = LABEL_POLICY_VERSION
    label["hopeless"] = bool(quality.get("hopeless", False))

    meta = {
        "features": feat_dict,
        "feature_seconds": analysis.total_seconds,
        "difficulty_reasons": list(diff.reasons),
        "label_quality": quality,
    }
    return label, trials, meta


def collect_clip_paths(
    dataset_root: Path,
    *,
    include_generated: bool = True,
    per_tier: int | None = None,
    exclude_1080: bool = True,
    extra_dirs: tuple[str, ...] = (),
) -> list[Path]:
    """Gather source clips stratified lightly by tier."""
    paths: list[Path] = []
    subs = ("ai", "render", "screen") if exclude_1080 else ("ai", "ai1080", "render", "screen")
    for sub in (*subs, *extra_dirs):
        d = dataset_root / sub
        if not d.is_dir():
            continue
        paths.extend(sorted(p for p in d.glob("*.mkv") if not p.name.endswith(".tmp.mkv")))
    if include_generated:
        g = dataset_root / "generated_hard"
        if g.is_dir():
            paths.extend(sorted(g.glob("gen_hard_*.mkv")))
    if exclude_1080:
        paths = [p for p in paths if "_1080" not in p.stem]

    if per_tier is None:
        return paths

    buckets: dict[str, list[Path]] = {"easy": [], "medium": [], "hard": [], "other": []}
    for p in paths:
        buckets[clip_tier(p.stem)].append(p)
    out: list[Path] = []
    for tier, bucket in buckets.items():
        if not bucket:
            continue
        if len(bucket) <= per_tier:
            out.extend(bucket)
        else:
            step = max(1, len(bucket) // per_tier)
            out.extend(bucket[::step][:per_tier])
    return out
