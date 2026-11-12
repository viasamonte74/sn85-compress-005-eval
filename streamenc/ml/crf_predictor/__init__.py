"""Feature-based predictor for the CRF contender: per-arm VMAF at budget, pilot CRF, CRF offsets.

Trained from label trials (``scripts/train_crf_predictor.py``); stored as a single pickle plus a
JSON summary. Inference builds a :class:`streamenc.encode.crf_pipeline.Plan`:

* ``final_arm``  — argmax over arms of ``predicted_vmaf - LAMBDA * predicted_cpu_seconds``
                   (speed term: ~0.016 VMAF per CPU-second = 0.13 per wall-second at 8 cores/clip)
* ``crf_start``  — predicted CRF of the pilot arm at budget (saves ~1 pilot probe)
* ``rate_ratio_log`` — median ln(bits(final_arm)/bits(pilot_arm)) at equal CRF (per arm, learned)
* ``hopeless``   — predicted best VMAF < HOPELESS_VMAF  -> speed arm
"""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from streamenc.features.extract import FEATURE_NAMES

GEOM_NAMES = ("log_pixels", "fps", "log_frames", "log_target_kbps")
HOPELESS_VMAF = 57.0
SPEED_ARM = "svt_crf_p8_t1"


def geometry_features(width: int, height: int, fps: float, frames: int, bits_per_pixel: float = 0.05) -> list[float]:
    target = bits_per_pixel * width * height * fps / 1000.0
    return [math.log(width * height), float(fps), math.log(max(frames, 1)), math.log(max(target, 1e-3))]


def pilot_log_util_at(fit_history: str, crf: float, max_extrap: float = 2.0) -> float | None:
    """ln(budget use) of the pilot arm at ``crf`` from its label fit history ("crf:util ..."):
    log-linear interpolation between measured points, short first-order extrapolation otherwise."""
    from streamenc.encode.ratefit import default_slope

    pts = []
    for tok in (fit_history or "").split():
        c, _, u = tok.partition(":")
        try:
            pts.append((float(c), max(float(u), 1e-3)))
        except ValueError:
            continue
    pts = sorted(set(pts))
    if not pts:
        return None
    for (c1, u1), (c2, u2) in zip(pts, pts[1:]):
        if c1 <= crf <= c2 and c2 > c1:
            t = (crf - c1) / (c2 - c1)
            return math.log(u1) + t * (math.log(u2) - math.log(u1))
    c0, u0 = min(pts, key=lambda q: abs(q[0] - crf))
    if abs(c0 - crf) > max_extrap:
        return None
    return math.log(u0) - default_slope(c0) * (crf - c0)


class DeltaRegressor:
    """VMAF of an arm = reference-arm VMAF model + model of (arm VMAF - reference VMAF) on the same clip.
    Training on the per-clip difference removes the clip-difficulty term that dominates absolute VMAF error
    (offline grouped CV 2026-11-11, AI/render/screen: utility regret 0.0096 -> 0.0061)."""

    def __init__(self, ref: Any, delta: Any) -> None:
        self.ref, self.delta = ref, delta

    def predict(self, X: Any) -> np.ndarray:
        return self.ref.predict(X) + self.delta.predict(X)


def feature_row(features: Any, meta: Any, bits_per_pixel: float = 0.05) -> np.ndarray:
    f = [float(getattr(features, n)) for n in FEATURE_NAMES]
    g = geometry_features(meta.width, meta.height, meta.fps, meta.nb_frames, bits_per_pixel)
    return np.asarray(f + g, dtype=np.float64)


@dataclass
class CrfPredictorModel:
    arms: list[str]
    pilot_arm: str
    vmaf_models: dict[str, Any]         # arm -> regressor (features -> vmaf at budget)
    cpu_models: dict[str, Any]          # arm -> regressor (features -> ln cpu seconds of one encode)
    crf_model: Any                      # features -> CRF of the pilot arm at budget
    offsets: dict[str, float]           # arm -> median ln(bits(arm)/bits(pilot)) at equal CRF
    lam: float                          # VMAF per CPU-second trade-off
    meta: dict[str, Any]

    def predict(self, features: Any, vmeta: Any, *, lam: float | None = None):
        from streamenc.encode.crf_pipeline import Plan

        x = feature_row(features, vmeta).reshape(1, -1)
        lam = self.lam if lam is None else lam
        vm = {a: float(m.predict(x)[0]) for a, m in self.vmaf_models.items()}
        cpu = {a: float(math.exp(m.predict(x)[0])) for a, m in self.cpu_models.items()}
        best_vmaf = max(vm.values())
        # Trees cannot extrapolate: only trust "hopeless" inside the frame-rate range seen in training.
        fr = self.meta.get("fps_range")
        in_range = not fr or (fr[0] - 1.0 <= float(vmeta.fps) <= fr[1] + 1.0)
        if best_vmaf < HOPELESS_VMAF and in_range:
            arm, hopeless = SPEED_ARM, True
        else:
            # Utility: clip score gain (VMAF above the 60 floor counts, capped at 100) minus time cost.
            util = {a: min(max(vm[a], 60.0), 100.0) - lam * cpu.get(a, 0.0) for a in self.arms if a != SPEED_ARM}
            arm, hopeless = max(util, key=util.get), False
        crf_start = float(np.clip(self.crf_model.predict(x)[0], 1.0, 63.0))
        rm = (getattr(self, "ratio_models", None) or {}).get(arm)
        if rm is not None:
            def ratio_fn(crf: float, _m=rm, _x=x) -> float:
                return float(_m.predict(np.append(_x, float(crf)).reshape(1, -1))[0])
            ratio = ratio_fn(crf_start)
        else:
            ratio_fn, ratio = None, float(self.offsets.get(arm, 0.0))
        plan = Plan(final_arm=arm, pilot_arm=self.pilot_arm, crf_start=crf_start, rate_ratio_log=ratio,
                    hopeless=hopeless, pred_vmaf=vm.get(arm), pred_best=best_vmaf, pred_cpu=cpu.get(arm),
                    lam=lam, notes=f"pred_vmaf={vm.get(arm, 0):.1f} best={best_vmaf:.1f} cpu={cpu.get(arm, 0):.0f}s "
                                   f"ratio={ratio:+.3f}")
        plan.ratio_fn = ratio_fn  # not a dataclass field: per-clip ln bit ratio at the pilot CRF
        gm = getattr(self, "gap_models", None) or {}
        gsd = getattr(self, "gap_sd", None) or {}

        def gap_fn(arm_id: str, crf: float, vmaf: float | None = None, _x=x, _vm=vm):
            """Predicted |VMAF - VMAF-NEG| of ``arm_id`` at ``crf`` and its CV residual sd (None: no model)."""
            m = gm.get(arm_id)
            if m is None:
                return None
            v = vmaf if vmaf is not None else _vm.get(arm_id, _vm.get(arm, 80.0))
            g = float(m.predict(np.append(_x, [float(v), float(crf)]).reshape(1, -1))[0])
            return g, float(gsd.get(arm_id, 0.5))

        plan.gap_fn = gap_fn if gm else None
        plan.vmaf_by_arm = vm
        rms = getattr(self, "ratio_models", None) or {}

        def ratio_fn_for(arm_id: str, _x=x):
            m = rms.get(arm_id)
            if m is None:
                c = float(self.offsets.get(arm_id, 0.0))
                return lambda crf: c
            return lambda crf, _m=m: float(_m.predict(np.append(_x, float(crf)).reshape(1, -1))[0])

        v2 = getattr(self, "vmaf2_models", None) or {}

        def stage2_fn(pilot_crf: float, _x=x):
            """After the pilot: (arm, vmaf by arm, hopeless, best) from features + the pilot's CRF at budget."""
            if not v2:
                return None
            xx = np.append(_x, float(pilot_crf)).reshape(1, -1)
            vm2 = {a: float(m.predict(xx)[0]) for a, m in v2.items()}
            best2 = max(vm2.values())
            if best2 < HOPELESS_VMAF and in_range:
                return SPEED_ARM, vm2, True, best2
            u2 = {a: min(max(vm2[a], 60.0), 100.0) - lam * cpu.get(a, 0.0) for a in vm2 if a != SPEED_ARM}
            return max(u2, key=u2.get), vm2, False, best2

        v3 = getattr(self, "vmaf3_models", None) or {}

        def stage3_fn(pilot_crf: float, pilot_vmaf: float, _x=x):
            """After the pilots + their measured VMAF (2026-11-12 prediction fix): (arm, vmaf by arm, hopeless, best).
            ``pilot_vmaf`` = the pilot arm's validator VMAF at budget use 0.985 (interpolated from the pilots)."""
            if not v3:
                return None
            xx = np.append(_x, [float(pilot_crf), float(pilot_vmaf), math.log(0.985)]).reshape(1, -1)
            vm3 = {a: float(pilot_vmaf) + float(m.predict(xx)[0]) for a, m in v3.items()}
            best3 = max(vm3.values())
            if best3 < HOPELESS_VMAF and in_range:
                return SPEED_ARM, vm3, True, best3
            u3 = {a: min(max(vm3[a], 60.0), 100.0) - lam * cpu.get(a, 0.0) for a in vm3 if a != SPEED_ARM}
            return max(u3, key=u3.get), vm3, False, best3

        fm = getattr(self, "fgain_model", None)

        def fgain_fn(pilot_crf: float, _x=x, _m=fm) -> float:
            """Predicted VMAF gain of the hard prefilter (2026-11-12, trained on offline single-segment probes)."""
            return float(_m.predict(np.append(_x, float(pilot_crf)).reshape(1, -1))[0])

        plan.ratio_fn_for = ratio_fn_for
        plan.stage2_fn = stage2_fn if v2 else None
        plan.stage3_fn = stage3_fn if v3 else None
        plan.fgain_fn = fgain_fn if fm is not None else None
        return plan


def save_model(model: CrfPredictorModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump({"arms": model.arms, "pilot_arm": model.pilot_arm, "offsets": model.offsets,
                   "lam": model.lam, "meta": model.meta}, fh, indent=2)


def load_model(path: str | Path) -> CrfPredictorModel:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def make_predictor(path: str | Path, *, lam: float | None = None):
    model = load_model(path)
    if lam is None and os.environ.get("STREAMENC_LAM"):
        lam = float(os.environ["STREAMENC_LAM"])  # experiment override of the speed trade-off

    def _predict(features: Any, vmeta: Any):
        return model.predict(features, vmeta, lam=lam)

    return _predict
