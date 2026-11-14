#!/usr/bin/env python3
"""Content classifier for the clean-content SVT tools (2026-11-13): P(render or screen) vs AI-generated, from the
content features only (no geometry: frame size / rate say nothing about the content and would learn this dataset's
mix). Trained on the compliant label set (kinds ai, generated_hard, render, screen; natural footage excluded).
Grouped CV by family is printed. Saves a copy of the CRF predictor pickle with .kind_model attached (all other
models untouched):
  --base models/crf_predictor_eval_c_abs.pkl --out models/crf_predictor_eval_c_abs_kind.pkl --exclude-split test
  --base models/crf_predictor.pkl            --out models/crf_predictor_kind.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.features.extract import FEATURE_NAMES  # noqa: E402

KINDS = ["ai", "generated_hard", "render", "screen"]
CLEAN = ["render", "screen"]


def clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/encoder_rank_all/features.csv"))
    ap.add_argument("--exclude-split", default="")
    a = ap.parse_args()
    F = pd.read_csv(a.data)
    F = F[F.kind.isin(KINDS)]
    if a.exclude_split:
        F = F[F.split != a.exclude_split]
    F = F.reset_index(drop=True)
    X = F[list(FEATURE_NAMES)].to_numpy(float)
    y = F.kind.isin(CLEAN).to_numpy(int)
    g = F.family.to_numpy()
    oof = np.zeros(len(F))
    for tr, te in GroupKFold(5).split(X, y, g):
        oof[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    p = oof >= 0.5
    print(f"{len(F)} clips ({y.sum()} render/screen); grouped CV at 0.5: accuracy {(p == y).mean():.3f}, "
          f"render/screen recall {p[y == 1].mean():.3f}, AI called clean {p[y == 0].mean():.3f}")
    for k in KINDS:
        s = (F.kind == k).to_numpy()
        print(f"  {k:15s} {s.sum():4d} clips, called clean {p[s].mean():.3f}")
    model = pickle.load(open(a.base, "rb"))
    model.kind_model = clf().fit(X, y)
    with open(a.out, "wb") as fh:
        pickle.dump(model, fh)
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
