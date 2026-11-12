#!/usr/bin/env python3
"""Prediction-error diagnosis on the held-out (test-split) clips (2026-11-12, user: "solve this model prediction issue").

For every held-out clip and every arm, compare the labelled VMAF at budget (trials.jsonl, whole-clip encode) with
  s1      stage-1 prediction (features + geometry)
  s2true  stage-2 prediction fed the labelled pilot CRF at budget (what stage 2 was trained on)
  s2rt    stage-2 prediction fed the runtime pilot CRF (solve_crf of the runtime pilot points, as crf_pipeline does)
and, for the arm the runtime shipped, the runtime VMAF (pipeline gap = runtime - label). Uses the rc3 4-clip batched
held-out run (results/e2e_batched/rc3_heldout) for the runtime side.
  python scripts/predfix_diag.py --model models/crf_predictor_eval_c_abs.pkl --run results/e2e_batched/rc3_heldout
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import pickle
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encode.ratefit import solve_crf  # noqa: E402
from streamenc.features.extract import FEATURE_NAMES  # noqa: E402
from streamenc.ml.crf_predictor import geometry_features  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402

AI_HARD = ("ai_hard", "aix_hard", "anim_hard", "animh_hard", "aig3_hard")
SLOPE = 0.14


def group(name: str, kind: str) -> str:
    if name.startswith(AI_HARD):
        return "ai_hard"
    return kind


def parse_run(run: Path) -> dict:
    rows = list(map(json.loads, open(run / "per_clip.jsonl")))
    batches, cur, it = [], None, None
    for line in open(run / "contender.log"):
        if line.startswith("===== batch"):
            cur = []; batches.append(cur); it = None; continue
        m = re.search(r"crf_pipeline\] item \w+: (\S+) crf=\S+ util=(\S+) .*plan=(\S+) \(pred_vmaf=([\d.]+)", line)
        if m:
            it = {"arm": m.group(1), "util": m.group(2), "plan": m.group(3), "h": []}; cur.append(it); continue
        if it is not None and "{'tag'" in line:
            it["h"].append(ast.literal_eval(line[line.index("{"):].strip()))
        elif it is not None and "crf_pipeline" not in line:
            it = None
    out = {}
    for r in rows:
        name = Path(r["ref"]).stem
        if name in out:
            continue
        c = [i for i in batches[r["batch"]] if r.get("util") is not None and i["util"] == str(round(r["util"], 4))]
        i = c[0] if len(c) == 1 else None
        pts = [(h["crf"], h["use"]) for h in (i["h"] if i else []) if h["tag"] == "pilot" and "use" in h]
        s2 = next((h for h in (i["h"] if i else []) if h["tag"] == "stage2"), None)
        pf = next((h["filter"] for h in (i["h"] if i else []) if h["tag"] == "hard_prefilter"), "none")
        out[name] = {"path": r["ref"], "vmaf": r.get("vmaf"), "score": r["clip_score"], "arm": i and i["arm"],
                     "points": pts, "s2_logged": s2, "prefilter": pf}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "crf_predictor_eval_c_abs.pkl"))
    ap.add_argument("--run", default=str(ROOT / "results" / "e2e_batched" / "rc3_heldout"))
    ap.add_argument("--data", default=str(ROOT / "data" / "encoder_rank_all"))
    ap.add_argument("--out", default=str(ROOT / "results" / "predfix" / "diag.json"))
    args = ap.parse_args()
    model = pickle.load(open(args.model, "rb"))
    d = Path(args.data)
    feats = pd.read_csv(d / "features.csv").drop_duplicates("clip_id", keep="last").set_index("clip_id")
    labels = {}
    for l in open(d / "labels.jsonl"):
        r = json.loads(l); labels[r["clip_id"]] = r
    lab = collections.defaultdict(dict)
    for l in open(d / "trials.jsonl"):
        t = json.loads(l)
        if t.get("passed") and t.get("vmaf") is not None:
            lab[t["clip_id"]][t["config_id"]] = t
    rt = parse_run(Path(args.run))
    recs = []
    for name, r in rt.items():
        if name not in feats.index or name not in lab:
            print("no label/features:", name); continue
        m = probe_video(r["path"], count_frames=True)
        x = np.array([float(feats.loc[name, n]) for n in FEATURE_NAMES]
                     + geometry_features(m.width, m.height, m.fps, m.nb_frames, 0.05), dtype=float)
        p8 = lab[name].get(model.pilot_arm)
        crf_true = None if p8 is None else float(p8["aim"]) + math.log(max(float(p8["util"]), 1e-3) / 0.985) / SLOPE
        crf_rt = solve_crf(r["points"], 0.985) if r["points"] else None
        rec = {"clip": name, "group": group(name, labels[name]["kind"]), "res": f"{m.width}x{m.height}", "fps": m.fps,
               "arm_rt": r["arm"], "prefilter": r["prefilter"], "vmaf_rt": r["vmaf"], "score": r["score"],
               "crf_true": crf_true, "crf_rt": crf_rt, "arms": {}}
        for a in model.vmaf_models:
            L = lab[name].get(a)
            e = {"label": None if L is None else float(L["vmaf"]),
                 "s1": float(model.vmaf_models[a].predict(x.reshape(1, -1))[0])}
            v2 = model.vmaf2_models.get(a)
            if v2 is not None:
                if crf_true is not None:
                    e["s2true"] = float(v2.predict(np.append(x, crf_true).reshape(1, -1))[0])
                if crf_rt is not None:
                    e["s2rt"] = float(v2.predict(np.append(x, crf_rt).reshape(1, -1))[0])
            rec["arms"][a] = e
        recs.append(rec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(recs, indent=1))

    def summ(tag, sel, key, arm_pick=None):
        errs = []
        for r in sel:
            arms = [arm_pick(r)] if arm_pick else list(r["arms"])
            for a in arms:
                e = r["arms"].get(a or "", {})
                if e.get(key) is not None and e.get("label") is not None:
                    errs.append(e[key] - e["label"])
        if errs:
            print(f"  {tag:34s} n={len(errs):4d} bias(pred-label) {st.mean(errs):+6.2f}  MAE {st.mean(map(abs, errs)):5.2f}"
                  f"  |err|>10: {sum(abs(x) > 10 for x in errs)}")

    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    for g, sel in sorted(groups.items()):
        print(f"== {g} ({len(sel)} clips)")
        for key in ("s1", "s2true", "s2rt"):
            summ(f"{key}, all arms", sel, key)
        summ("s2rt, shipped arm", sel, "s2rt", lambda r: r["arm_rt"])
        pe = [r["crf_rt"] - r["crf_true"] for r in sel if r["crf_rt"] is not None and r["crf_true"] is not None]
        if pe:
            print(f"  pilot CRF runtime - label: n={len(pe)} bias {st.mean(pe):+.2f} MAE {st.mean(map(abs, pe)):.2f}")
        gap = [(r["vmaf_rt"] - r["arms"][r["arm_rt"]]["label"], r["prefilter"]) for r in sel
               if r["arm_rt"] in r["arms"] and r["arms"][r["arm_rt"]]["label"] is not None and r["vmaf_rt"] is not None]
        for pf in sorted({p for _, p in gap}):
            v = [x for x, p in gap if p == pf]
            print(f"  pipeline gap runtime - label ({pf:11s}): n={len(v)} mean {st.mean(v):+.2f}")
    print("== stage-1 residual by label VMAF level (all held-out, all arms)")
    bins = collections.defaultdict(list)
    for r in recs:
        for a, e in r["arms"].items():
            if e["label"] is not None:
                bins[min(int(e["label"] // 10) * 10, 90)].append(e["s1"] - e["label"])
    for b in sorted(bins):
        print(f"  label {b:3d}-{b + 10:<3d} n={len(bins[b]):4d} bias {st.mean(bins[b]):+6.2f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
