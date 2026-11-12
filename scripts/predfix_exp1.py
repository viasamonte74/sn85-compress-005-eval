#!/usr/bin/env python3
"""Prediction fix, experiment 1 (2026-11-12): what would fix the AI-hard over-prediction? Offline, labels only.

Train set = the eval model's (non-test, non-natural); test = test split, non-natural. Per arm, VMAF at budget:
  A  stage-2 as shipped: features + geometry + pilot CRF at budget
  B  A + isotonic calibration fitted on A's grouped out-of-fold predictions (no runtime cost)
  C  stage 3: A + the pilot arm's (p8) measured VMAF at budget + ln(its util)   [upper bound of "measure the pilot"]
  D  stage 3 as a delta: predict arm VMAF - p8 VMAF from C's inputs
Grouped 5-fold CV by family on train (out-of-fold), then fit on all train -> test. Reports bias / MAE by group.
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.features.extract import FEATURE_NAMES  # noqa: E402
from streamenc.ml.crf_predictor import geometry_features  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402

AI_HARD = ("ai_hard", "aix_hard", "anim_hard", "animh_hard", "aig3_hard")
PILOT = "svt_crf_p8_t1"
SLOPE = 0.14


def gbr():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=8,
                                         l2_regularization=1.0, random_state=0)


def main() -> int:
    d = ROOT / "data" / "encoder_rank_all"
    feats = pd.read_csv(d / "features.csv").drop_duplicates("clip_id", keep="last")
    labels = {}
    for l in open(d / "labels.jsonl"):
        r = json.loads(l); labels[r["clip_id"]] = r
    trials = collections.defaultdict(dict)
    for l in open(d / "trials.jsonl"):
        t = json.loads(l)
        if t.get("passed") and t.get("vmaf") is not None:
            trials[t["clip_id"]][t["config_id"]] = t
    keep = [r for r in feats.itertuples() if r.clip_id in labels and labels[r.clip_id]["kind"] != "other"
            and PILOT in trials.get(r.clip_id, {})]
    X, meta = {}, {}
    for r in keep:
        try:
            m = probe_video(r.path, count_frames=True)
        except Exception:  # noqa: BLE001
            continue
        X[r.clip_id] = [float(getattr(r, n)) for n in FEATURE_NAMES] + geometry_features(m.width, m.height, m.fps, m.nb_frames, 0.05)
        p = trials[r.clip_id][PILOT]
        meta[r.clip_id] = {"crf": float(p["aim"]) + math.log(max(float(p["util"]), 1e-3) / 0.985) / SLOPE,
                           "v8": float(p["vmaf"]), "lu8": math.log(max(float(p["util"]), 1e-3)),
                           "split": labels[r.clip_id]["split"], "fam": labels[r.clip_id]["family"],
                           "grp": "ai_hard" if r.clip_id.startswith(AI_HARD) else labels[r.clip_id]["kind"]}
    train = [c for c in X if meta[c]["split"] != "test"]
    test = [c for c in X if meta[c]["split"] == "test"]
    print(f"train {len(train)} clips, test {len(test)} clips")
    arms = sorted({a for c in X for a in trials[c]})
    res = collections.defaultdict(list)  # (variant, set, grp, labelbin) -> errors

    def inputs(c, v):
        b = X[c] + [meta[c]["crf"]]
        return b + [meta[c]["v8"], meta[c]["lu8"]] if v in ("C", "D") else b

    for a in arms:
        tr = [c for c in train if a in trials[c]]; te = [c for c in test if a in trials[c]]
        ytr = np.array([trials[c][a]["vmaf"] for c in tr]); yte = np.array([trials[c][a]["vmaf"] for c in te])
        g = np.array([meta[c]["fam"] for c in tr])
        for v in ("A", "C", "D"):
            Xtr = np.array([inputs(c, v) for c in tr]); Xte = np.array([inputs(c, v) for c in te])
            off_tr = np.array([meta[c]["v8"] for c in tr]) if v == "D" else 0.0
            off_te = np.array([meta[c]["v8"] for c in te]) if v == "D" else 0.0
            oof = np.zeros_like(ytr)
            for i, j in GroupKFold(n_splits=5).split(Xtr, ytr, g):
                oof[j] = gbr().fit(Xtr[i], (ytr - off_tr)[i]).predict(Xtr[j]) + (off_tr[j] if v == "D" else 0.0)
            pte = gbr().fit(Xtr, ytr - off_tr).predict(Xte) + off_te
            sets = [("oof", tr, oof, ytr), ("test", te, pte, yte)]
            if v == "A":  # B: isotonic on A's OOF (per arm)
                iso = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr)
                sets += [("oofB", tr, None, ytr), ("testB", te, iso.predict(pte), yte)]
                # honest OOF for B: calibrate each fold's OOF with an isotonic fitted on the other folds' OOF
                oofB = np.zeros_like(ytr)
                for i, j in GroupKFold(n_splits=5).split(Xtr, ytr, g):
                    oofB[j] = IsotonicRegression(out_of_bounds="clip").fit(oof[i], ytr[i]).predict(oof[j])
                sets[2] = ("oofB", tr, oofB, ytr)
            for name, cs, p, y in sets:
                var = v if not name.endswith("B") else "B"
                s = name.rstrip("B")
                for c, pp, yy in zip(cs, p, y):
                    res[(var, s, meta[c]["grp"])].append(pp - yy)
                    res[(var, s, "lvl" + str(min(int(yy // 10) * 10, 90)))].append(pp - yy)
    out = {}
    for s in ("oof", "test"):
        print(f"\n===== {'train out-of-fold (grouped CV)' if s == 'oof' else 'held-out test split'}")
        grps = sorted({k[2] for k in res if k[1] == s and not k[2].startswith("lvl")}) + \
               sorted({k[2] for k in res if k[1] == s and k[2].startswith("lvl")}, key=lambda x: int(x[3:]))
        print(f"{'group':10s} " + " ".join(f"{v + ' bias/MAE':>16s}" for v in "ABCD"))
        for gname in grps:
            cells = []
            for v in "ABCD":
                e = res.get((v, s, gname), [])
                cells.append(f"{st.mean(e):+6.2f}/{st.mean(map(abs, e)):5.2f}" if e else " " * 12)
                out[f"{s}|{gname}|{v}"] = (len(e), st.mean(e) if e else None, st.mean(map(abs, e)) if e else None)
            n = len(res.get(("A", s, gname), []))
            print(f"{gname:10s} " + " ".join(f"{c:>16s}" for c in cells) + f"   n={n}")
    (ROOT / "results" / "predfix").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "predfix" / "exp1.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
