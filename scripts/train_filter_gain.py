#!/usr/bin/env python3
"""Zero-runtime-cost prefilter decision (2026-11-12, user: "Go ahead"): predict the cblur4_ds85 gain from the content
features + geometry + the pilot arm's CRF at budget (all known at runtime after the pilots), trained on the offline
single-segment probe labels (data/filter_gain/labels.jsonl). Grouped CV by family; test split vs the probe labels and vs
the 14 real end-to-end gains (results/predfix/filter_gain_truth.json). Saves copies of the CRF predictor pickles with
.fgain_model attached (stage 1/2 untouched):
  --base models/crf_predictor_eval_c_abs.pkl --out models/crf_predictor_eval_c_abs_fg.pkl --exclude-split test
  --base models/crf_predictor.pkl            --out models/crf_predictor_c0_fg.pkl
"""
from __future__ import annotations

import argparse, collections, json, math, pickle, statistics as st, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.features.extract import FEATURE_NAMES  # noqa: E402
from streamenc.ml.crf_predictor import geometry_features  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402

SLOPE = 0.14
PILOT = "svt_crf_p8_t1"


def gbr():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=8,
                                         l2_regularization=1.0, random_state=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-split", default="")
    ap.add_argument("--margin", type=float, default=0.1)
    args = ap.parse_args()
    D = ROOT / "data" / "encoder_rank_all"
    lab = {json.loads(l)["clip_id"]: json.loads(l) for l in open(ROOT / "data/filter_gain/labels.jsonl")}
    feats = pd.read_csv(D / "features.csv").drop_duplicates("clip_id", keep="last").set_index("clip_id")
    p8 = {}
    for l in open(D / "trials.jsonl"):
        t = json.loads(l)
        if t["config_id"] == PILOT and t.get("passed") and t.get("vmaf") is not None:
            p8[t["clip_id"]] = t
    rows = []
    for c, r in lab.items():
        if c not in feats.index or c not in p8:
            continue
        m = probe_video(feats.loc[c, "path"], count_frames=True)
        x = [float(feats.loc[c, n]) for n in FEATURE_NAMES] + geometry_features(m.width, m.height, m.fps, m.nb_frames, 0.05)
        t = p8[c]
        crf = float(t["aim"]) + math.log(max(float(t["util"]), 1e-3) / 0.985) / SLOPE
        rows.append((c, r["family"], r["split"], r["kind"], x + [crf], r["gain"], float(t["vmaf"])))
    excl = set(args.exclude_split.split(",")) - {""}
    tr = [r for r in rows if r[2] not in excl]
    te = [r for r in rows if r[2] in excl] if excl else []
    X = np.array([r[4] for r in tr]); y = np.array([r[5] for r in tr]); g = np.array([r[1] for r in tr])
    oof = np.zeros_like(y)
    for i, j in GroupKFold(n_splits=5).split(X, y, g):
        oof[j] = gbr().fit(X[i], y[i]).predict(X[j])

    def report(tag, pred, true, v8):
        e = pred - true
        big = np.abs(true) > 0.2
        sign = np.mean((pred[big] > 0) == (true[big] > 0)) if big.any() else float("nan")
        take = float(np.sum(true[pred > args.margin])); allf = float(np.sum(true)); orc = float(np.sum(np.maximum(true, 0)))
        hard = v8 < 85
        take_h = float(np.sum(true[(pred > args.margin) & hard])); all_h = float(np.sum(true[hard])); orc_h = float(np.sum(np.maximum(true[hard], 0)))
        print(f"{tag}: n={len(true)} MAE {np.mean(np.abs(e)):.2f} bias {np.mean(e):+.2f} sign(|gain|>0.2) {sign:.3f} | decide: {take:+.1f} "
              f"vs always {allf:+.1f} oracle {orc:+.1f} | p8 VMAF<85 (n={int(hard.sum())}): {take_h:+.1f} vs always {all_h:+.1f} oracle {orc_h:+.1f}")
    report("train OOF (grouped)", oof, y, np.array([r[6] for r in tr]))
    model = gbr().fit(X, y)
    if te:
        Xt = np.array([r[4] for r in te]); yt = np.array([r[5] for r in te])
        report("test split vs probe labels", model.predict(Xt), yt, np.array([r[6] for r in te]))
        truth = json.load(open(ROOT / "results/predfix/filter_gain_truth.json"))
        idx = {r[0]: r for r in te}
        pr = [(n, float(model.predict(np.array([idx[n][4]]))[0]), v["gain"]) for n, v in truth.items() if n in idx]
        print("vs the 14 REAL end-to-end gains (clip, predicted, real):")
        for n, p, r in sorted(pr, key=lambda z: z[2]):
            print(f"   {n:26s} {p:+6.2f} {r:+6.2f} {'ok' if (p > args.margin) == (r > 0) or abs(r) < 0.2 else 'WRONG'}")
        print(f"   decide by model: {sum(r for _, p, r in pr if p > args.margin):+.2f} (always {sum(r for *_, r in pr):+.2f}, oracle "
              f"{sum(max(r, 0) for *_, r in pr):+.2f}, probe +7.97)")
    base = pickle.load(open(args.base, "rb"))
    base.fgain_model = model
    base.meta["fgain"] = {"n": len(tr), "exclude_split": args.exclude_split, "margin": args.margin,
                          "inputs": "features + geometry + pilot CRF at budget", "label": "single-segment p8 probe"}
    with open(args.out, "wb") as f:
        pickle.dump(base, f)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
