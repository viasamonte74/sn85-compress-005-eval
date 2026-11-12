#!/usr/bin/env python3
"""Stage-3 retrain check: the retrained model's stage-1 / stage-2 / CRF / CPU / ratio parts must predict exactly like
the existing model on every labelled clip (only vmaf3_models may be new)."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.features.extract import FEATURE_NAMES
old, new = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
f = pd.read_csv(ROOT / "data/encoder_rank_all/features.csv").drop_duplicates("clip_id", keep="last")
X = np.array([[float(getattr(r, n)) for n in FEATURE_NAMES] + [1280.0, 720.0, 30.0, 150.0, 0.05, 0.0][: len(old.meta["features"]) - len(FEATURE_NAMES)]
              for r in f.itertuples()])
X = X[:, : len(old.meta["features"])]
bad = 0
for part in ("vmaf_models", "cpu_models", "vmaf2_models", "ratio_models", "gap_models"):
    for a, m in getattr(old, part).items():
        n_in = m.n_features_in_
        Xa = np.hstack([X, np.full((len(X), n_in - X.shape[1]), 55.0)]) if n_in > X.shape[1] else X[:, :n_in]
        d = float(np.max(np.abs(m.predict(Xa) - getattr(new, part)[a].predict(Xa))))
        bad += d != 0.0
        if d: print("DIFF", part, a, d)
d = float(np.max(np.abs(old.crf_model.predict(X) - new.crf_model.predict(X)))); bad += d != 0.0
print(f"{sys.argv[2]}: {'IDENTICAL' if not bad else str(bad) + ' parts differ'} stage-1/2 parts; vmaf3 arms {len(getattr(new, 'vmaf3_models', {}) or {})}")
