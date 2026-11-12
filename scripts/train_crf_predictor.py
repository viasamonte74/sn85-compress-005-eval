#!/usr/bin/env python3
"""Train the CRF-contender predictor from label trials (features.csv + trials.jsonl).

Per arm: gradient-boosted regressors for VMAF at budget and ln(CPU seconds of one encode);
one regressor for the pilot arm's CRF at budget; per-arm CRF offsets to the pilot.
Evaluation is grouped K-fold by clip family (split column) so no clip leaks between folds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from streamenc.features.extract import FEATURE_NAMES  # noqa: E402
from streamenc.ml.crf_predictor import (GEOM_NAMES, CrfPredictorModel, DeltaRegressor, geometry_features,  # noqa: E402
                                        pilot_log_util_at, save_model)
from streamenc.probe import probe_video  # noqa: E402

SLOPE = 0.14  # ln(bitrate) per CRF, used to normalise CRF to util 0.985


def _gbr(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
                                         min_samples_leaf=8, l2_regularization=1.0, random_state=seed)


def _gbr_abs(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
                                         min_samples_leaf=8, l2_regularization=1.0, random_state=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="merged label dir (features.csv, trials.jsonl, labels.jsonl)")
    ap.add_argument("--out", default=str(ROOT / "models" / "crf_predictor.pkl"))
    ap.add_argument("--pilot-arm", default="svt_crf_p8_t1")
    ap.add_argument("--lam", type=float, default=0.012,
                    help="VMAF per CPU-second trade-off (local e2e break-even 0.0128 at R3428 ~206 s)")
    ap.add_argument("--min-arm-rows", type=int, default=20)
    ap.add_argument("--exclude-split", default="", help="comma list of label splits to hold out (e.g. test)")
    # 2026-11-11 training experiments (scripts/train_experiments.py): predicting each arm's VMAF difference to a
    # reference arm cut utility regret 36 %; natural-video down-weighting / family balancing are small extras.
    # 2026-11-11 compliance: MINING.md allows training only on own / CC0 / CC-BY / permissive-synthetic data; the
    # natural-video set has CC BY-SA clips and the hidden set has no camera footage -> user: exclude kind "other".
    ap.add_argument("--exclude-kinds", default="", help="comma list of label kinds to leave out of training")
    ap.add_argument("--target", choices=["abs", "delta"], default="abs", help="VMAF models: absolute or difference to --ref")
    ap.add_argument("--ref", default="svt_crf_p4_qm_la", help="reference arm of --target delta")
    ap.add_argument("--w-natural", type=float, default=1.0, help="sample weight of natural-video ('other') clips")
    ap.add_argument("--family-balance", type=float, default=0.0, help="weight x family_size ** -p")
    args = ap.parse_args()

    d = Path(args.data)
    feats = pd.read_csv(d / "features.csv")
    trials = pd.DataFrame([json.loads(l) for l in open(d / "trials.jsonl")])
    labels = pd.DataFrame([json.loads(l) for l in open(d / "labels.jsonl")])
    fam = dict(zip(labels.clip_id, labels.family))
    kind = dict(zip(labels.clip_id, labels.kind))
    fam_n = labels.family.value_counts().to_dict()

    def W(clip_ids) -> np.ndarray:
        """Per-row sample weight (natural-video weight x family balance)."""
        return np.array([(args.w_natural if kind.get(c) == "other" else 1.0)
                         * fam_n.get(fam.get(c, c), 1) ** -args.family_balance for c in clip_ids], dtype=float)
    if args.exclude_split:
        held = set(labels[labels.split.isin(args.exclude_split.split(","))].clip_id)
        feats = feats[~feats.clip_id.isin(held)].reset_index(drop=True)
        print(f"held out {len(held)} clips of split(s) {args.exclude_split}")

    if args.exclude_kinds:
        drop = set(labels[labels.kind.isin(args.exclude_kinds.split(","))].clip_id)
        feats = feats[~feats.clip_id.isin(drop)].reset_index(drop=True)
        print(f"excluded {len(drop)} clips of kind(s) {args.exclude_kinds}")

    # geometry per clip (probe once)
    geo = {}
    for cid, path in zip(feats.clip_id, feats.path):
        try:
            m = probe_video(path, count_frames=True)
            geo[cid] = geometry_features(m.width, m.height, m.fps, m.nb_frames)
        except Exception as exc:  # noqa: BLE001
            print("probe failed", cid, exc)
    feats = feats[feats.clip_id.isin(geo)].reset_index(drop=True)
    X_by_clip = {r.clip_id: [float(getattr(r, n)) for n in FEATURE_NAMES] + geo[r.clip_id] for r in feats.itertuples()}

    valid = trials[(trials.passed == True) & trials.vmaf.notna() & trials.clip_id.isin(X_by_clip)]  # noqa: E712
    arms = sorted(valid.config_id.unique())
    print(f"{len(X_by_clip)} clips, {len(valid)} valid trials, arms={arms}")

    from streamenc.encode.ratefit import default_slope

    def norm_crf(row) -> float:
        # CRF the arm would need at util 0.985 (first-order correction for the fit's residual)
        return float(row.aim) + math.log(max(float(row.util), 1e-3) / 0.985) / default_slope(float(row.aim))

    def log_bits_at(row, crf: float) -> float:
        # ln(util) the arm would have at ``crf`` (first-order, local slope)
        return math.log(max(float(row.util), 1e-3)) - default_slope(float(row.aim)) * (crf - float(row.aim))

    vmaf_models, cpu_models, report = {}, {}, {}
    for arm in arms:
        sub = valid[valid.config_id == arm]
        if len(sub) < args.min_arm_rows:
            print("skip", arm, len(sub)); continue
        X = np.array([X_by_clip[c] for c in sub.clip_id]); y = sub.vmaf.to_numpy(float)
        ycpu = np.log(np.clip(sub.enc_cpu_seconds.fillna(sub.cpu_seconds / sub.attempts).to_numpy(float), 0.05, None))
        groups = np.array([fam.get(c, c) for c in sub.clip_id])
        # grouped CV for the VMAF model
        cv_pred = np.zeros_like(y)
        nsp = min(5, len(set(groups)))
        if nsp >= 2:
            for tr, te in GroupKFold(n_splits=nsp).split(X, y, groups):
                cv_pred[te] = _gbr().fit(X[tr], y[tr]).predict(X[te])
            mae = float(np.mean(np.abs(cv_pred - y)))
        else:
            mae = float("nan")
        w = W(sub.clip_id)
        vmaf_models[arm] = _gbr().fit(X[w > 0], y[w > 0], sample_weight=w[w > 0])
        cpu_models[arm] = _gbr().fit(X[w > 0], ycpu[w > 0], sample_weight=w[w > 0])
        report[arm] = {"n": int(len(sub)), "cv_mae_vmaf": round(mae, 3), "mean_vmaf": round(float(y.mean()), 2),
                       "mean_cpu_s": round(float(np.exp(ycpu).mean()), 1)}
        print(f"{arm:24s} n={len(sub):4d} cv MAE {mae:.2f} mean vmaf {y.mean():.2f} cpu {np.exp(ycpu).mean():.1f}s")

    # pilot CRF model + offsets
    pil = valid[valid.config_id == args.pilot_arm]
    pil_crf = {r.clip_id: norm_crf(r) for r in pil.itertuples()}
    pil_rows = {r.clip_id: r for r in pil.itertuples()}
    Xp = np.array([X_by_clip[c] for c in pil_crf]); yp = np.array(list(pil_crf.values()))
    gp = np.array([fam.get(c, c) for c in pil_crf])
    cvp = np.zeros_like(yp)
    for tr, te in GroupKFold(n_splits=min(5, len(set(gp)))).split(Xp, yp, gp):
        cvp[te] = _gbr().fit(Xp[tr], yp[tr]).predict(Xp[te])
    crf_mae = float(np.mean(np.abs(cvp - yp)))
    print(f"pilot CRF model: n={len(yp)} cv MAE {crf_mae:.2f} CRF (sd of target {yp.std():.2f})")
    crf_model = _gbr().fit(Xp, yp)
    offsets = {}
    for arm in vmaf_models:
        sub = valid[valid.config_id == arm]
        # ln(bits(arm)/bits(pilot)) at the arm's own CRF: evaluate both at crf_arm (first-order for the pilot)
        diffs = [math.log(max(float(r.util), 1e-3)) - log_bits_at(pil_rows[r.clip_id], float(r.aim))
                 for r in sub.itertuples() if r.clip_id in pil_rows]
        offsets[arm] = float(np.median(diffs)) if diffs else 0.0
        print(f"rate ratio {arm:24s} median ln {offsets[arm]:+.3f} sd {np.std(diffs):.3f} n={len(diffs)}")

    # Per-clip ln(bits(arm)/bits(pilot)) at the arm's CRF: pilot use interpolated from its own fit
    # history (global sd 0.14-0.22, MAD 0.06-0.10); GBR on features + CRF (absolute loss, heavy tails)
    # halves the MAD (0.04-0.055, grouped CV). Runtime evaluates it at the pilot's CRF.
    pil_hist = {r.clip_id: r.fit_history for r in trials[trials.config_id == args.pilot_arm].itertuples()}
    ratio_models, ratio_report = {}, {}
    st = trials[(trials.passed_streaming == True) & trials.clip_id.isin(X_by_clip)]  # noqa: E712
    for arm in vmaf_models:
        if arm == args.pilot_arm:
            continue
        rows = []
        for r in st[st.config_id == arm].itertuples():
            if r.clip_id not in pil_hist or r.aim is None or r.util is None:
                continue
            lp = pilot_log_util_at(pil_hist[r.clip_id], float(r.aim))
            if lp is None:
                continue
            rows.append((r.clip_id, math.log(max(float(r.util), 1e-3)) - lp, float(r.aim)))
        if len(rows) < args.min_arm_rows:
            continue
        Xr = np.array([X_by_clip[c] + [crf] for c, _, crf in rows]); yr = np.array([y for _, y, _ in rows])
        gr = np.array([fam.get(c, c) for c, _, _ in rows])
        cvr = np.zeros_like(yr)
        for tr, te in GroupKFold(n_splits=min(5, len(set(gr)))).split(Xr, yr, gr):
            cvr[te] = _gbr_abs().fit(Xr[tr], yr[tr]).predict(Xr[te])
        res = yr - cvr
        ratio_models[arm] = _gbr_abs().fit(Xr, yr)
        ratio_report[arm] = {"n": len(rows), "median": round(float(np.median(yr)), 4),
                             "global_mad": round(float(np.median(np.abs(yr - np.median(yr)))), 4),
                             "cv_mad": round(float(np.median(np.abs(res))), 4), "cv_sd": round(float(res.std()), 4)}
        print(f"ratio model {arm:24s} n={len(rows)} median {np.median(yr):+.3f} global MAD "
              f"{ratio_report[arm]['global_mad']:.3f} -> cv MAD {ratio_report[arm]['cv_mad']:.3f} (sd {res.std():.3f})")

    # Stage-2 VMAF models: features + geometry + the pilot's CRF at budget (norm_crf, measured at
    # runtime by the pilot search). Grouped CV (605 clips): VMAF MAE 7.1 -> 3.8, argmax regret
    # 0.457 -> 0.401. Runtime re-picks the final arm and re-sets pred_vmaf after the pilot.
    vmaf2_models, report2 = {}, {}
    for arm in vmaf_models:
        sub = valid[(valid.config_id == arm) & valid.clip_id.isin(pil_crf)]
        if len(sub) < args.min_arm_rows:
            continue
        X2 = np.array([X_by_clip[c] + [pil_crf[c]] for c in sub.clip_id]); y2 = sub.vmaf.to_numpy(float)
        g2 = np.array([fam.get(c, c) for c in sub.clip_id])
        cv2 = np.zeros_like(y2)
        for tr, te in GroupKFold(n_splits=min(5, len(set(g2)))).split(X2, y2, g2):
            cv2[te] = _gbr().fit(X2[tr], y2[tr]).predict(X2[te])
        w2 = W(sub.clip_id)
        vmaf2_models[arm] = _gbr().fit(X2[w2 > 0], y2[w2 > 0], sample_weight=w2[w2 > 0])
        report2[arm] = {"n": int(len(sub)), "cv_mae_vmaf": round(float(np.mean(np.abs(cv2 - y2))), 3)}
        print(f"stage2 {arm:24s} n={len(sub):4d} cv MAE {report2[arm]['cv_mae_vmaf']:.2f} (stage1 {report[arm]['cv_mae_vmaf']:.2f})")

    # Stage-3 VMAF models (2026-11-12 prediction fix): after the pilots the runtime MEASURES the pilot arm's VMAF
    # (quick check on the two pilots, interpolated to budget use 0.985). Predict each arm as that measured VMAF plus a
    # learned difference. Held-out AI-hard clips: stage-2 MAE 6.6 (over-predicts hard clips: regression to the mean)
    # -> 0.73 with the runtime measurement (scripts/predfix_exp1.py / predfix_exp2.py).
    vmaf3_models, report3 = {}, {}
    v8 = {c: float(r.vmaf) for c, r in pil_rows.items()}
    lu8 = {c: math.log(max(float(r.util), 1e-3)) for c, r in pil_rows.items()}
    for arm in vmaf_models:
        sub = valid[(valid.config_id == arm) & valid.clip_id.isin(pil_crf)]
        if len(sub) < args.min_arm_rows:
            continue
        X3 = np.array([X_by_clip[c] + [pil_crf[c], v8[c], lu8[c]] for c in sub.clip_id])
        y3 = sub.vmaf.to_numpy(float) - np.array([v8[c] for c in sub.clip_id])
        g3 = np.array([fam.get(c, c) for c in sub.clip_id])
        cv3 = np.zeros_like(y3)
        for tr, te in GroupKFold(n_splits=min(5, len(set(g3)))).split(X3, y3, g3):
            cv3[te] = _gbr().fit(X3[tr], y3[tr]).predict(X3[te])
        w3 = W(sub.clip_id)
        vmaf3_models[arm] = _gbr().fit(X3[w3 > 0], y3[w3 > 0], sample_weight=w3[w3 > 0])
        report3[arm] = {"n": int(len(sub)), "cv_mae_vmaf": round(float(np.mean(np.abs(cv3 - y3))), 3)}
        print(f"stage3 {arm:24s} n={len(sub):4d} cv MAE {report3[arm]['cv_mae_vmaf']:.2f} "
              f"(stage2 {report2.get(arm, {}).get('cv_mae_vmaf', float('nan')):.2f})")

    if args.target == "delta":
        V = {a: dict(zip(valid[valid.config_id == a].clip_id, valid[valid.config_id == a].vmaf.astype(float)))
             for a in vmaf_models}
        if args.ref not in vmaf_models:
            raise SystemExit(f"--ref {args.ref} has no VMAF model")

        def to_delta(models: dict, stage2: bool) -> dict:
            out = {}
            for arm in models:
                if arm == args.ref:
                    out[arm] = models[arm]
                    continue
                cl = [c for c in V[arm] if c in V[args.ref] and (not stage2 or c in pil_crf)]
                w = W(cl)
                cl = [c for c, x in zip(cl, w) if x > 0]
                Xd = np.array([X_by_clip[c] + ([pil_crf[c]] if stage2 else []) for c in cl])
                yd = np.array([V[arm][c] - V[args.ref][c] for c in cl])
                out[arm] = DeltaRegressor(models[args.ref], _gbr().fit(Xd, yd, sample_weight=w[w > 0]))
            print(f"delta models ({'stage 2' if stage2 else 'stage 1'}): ref {args.ref}, {len(out) - 1} arms")
            return out

        vmaf_models = to_delta(vmaf_models, False)
        vmaf2_models = to_delta(vmaf2_models, True)

    # Per-arm VMAF-NEG gap model (validator gate |VMAF - NEG| <= 3.0 -> else clip score 0).
    # Inputs: features + geometry + VMAF + CRF at budget; runtime evaluates it on the shipped encode
    # (predicted VMAF, actual CRF) and measures the gap only when P(gap > NEG_SAFE) is non-trivial.
    gap_models, gap_sd, gap_report = {}, {}, {}
    gt = trials[(trials.passed_streaming == True) & trials.vmaf_delta.notna() & trials.vmaf.notna()  # noqa: E712
                & trials.clip_id.isin(X_by_clip)]
    for arm in sorted(gt.config_id.unique()):
        sub = gt[gt.config_id == arm]
        if len(sub) < args.min_arm_rows:
            continue
        Xg = np.array([X_by_clip[r.clip_id] + [float(r.vmaf), float(r.aim)] for r in sub.itertuples()])
        yg = sub.vmaf_delta.to_numpy(float)
        gg = np.array([fam.get(c, c) for c in sub.clip_id])
        cvg = np.zeros_like(yg)
        for tr, te in GroupKFold(n_splits=min(5, len(set(gg)))).split(Xg, yg, gg):
            cvg[te] = _gbr().fit(Xg[tr], yg[tr]).predict(Xg[te])
        res = yg - cvg
        gap_models[arm] = _gbr().fit(Xg, yg)
        gap_sd[arm] = float(res.std())
        fail = yg > 3.0
        gap_report[arm] = {"n": int(len(yg)), "fail_rate": round(float(fail.mean()), 4),
                           "cv_mae": round(float(np.mean(np.abs(res))), 3), "cv_sd": round(gap_sd[arm], 3),
                           "recall_at_2.4": round(float(((cvg > 2.4) & fail).sum() / max(1, fail.sum())), 3),
                           "flag_rate_at_2.4": round(float((cvg > 2.4).mean()), 3)}
        print(f"gap model {arm:24s} n={len(yg)} fail {100 * fail.mean():.1f}% cv MAE {gap_report[arm]['cv_mae']:.3f} "
              f"sd {gap_sd[arm]:.3f} recall@2.4 {gap_report[arm]['recall_at_2.4']:.2f}")

    # Winner evaluation (grouped CV argmax vs oracle) on clips with every arm valid
    cl = [c for c in X_by_clip if all(((valid.clip_id == c) & (valid.config_id == a)).any() for a in vmaf_models)]
    if cl:
        V = {a: {r.clip_id: r.vmaf for r in valid[valid.config_id == a].itertuples()} for a in vmaf_models}
        X = np.array([X_by_clip[c] for c in cl]); g = np.array([fam.get(c, c) for c in cl])
        regret = []
        for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(X, X[:, 0], g):
            preds = {a: _gbr().fit(X[tr], np.array([V[a][cl[i]] for i in tr])).predict(X[te]) for a in vmaf_models}
            for j, i in enumerate(te):
                c = cl[i]
                pick = max(preds, key=lambda a: preds[a][j])
                regret.append(max(V[a][c] for a in vmaf_models) - V[pick][c])
        print(f"winner regret (VMAF, pure quality argmax): mean {np.mean(regret):.3f} p90 {np.percentile(regret, 90):.2f} n={len(regret)}")

    model = CrfPredictorModel(arms=list(vmaf_models), pilot_arm=args.pilot_arm, vmaf_models=vmaf_models,
                              cpu_models=cpu_models, crf_model=crf_model, offsets=offsets, lam=args.lam,
                              meta={"features": list(FEATURE_NAMES) + list(GEOM_NAMES), "report": report,
                                    "pilot_crf_cv_mae": crf_mae, "data": str(d),
                                    "exclude_split": args.exclude_split, "ratio_report": ratio_report,
                                    "exclude_kinds": args.exclude_kinds, "target": args.target, "ref": args.ref, "w_natural": args.w_natural,
                                    "family_balance": args.family_balance,
                                    "gap_report": gap_report, "stage2_report": report2,
                                    "fps_range": [float(min(g[1] for g in geo.values() if True)),
                                                  float(max(g[1] for g in geo.values()))]})
    model.ratio_models = ratio_models
    model.gap_models = gap_models
    model.vmaf2_models = vmaf2_models
    model.vmaf3_models = vmaf3_models
    model.meta["stage3_report"] = report3
    model.gap_sd = gap_sd
    save_model(model, args.out)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
