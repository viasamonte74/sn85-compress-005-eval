#!/usr/bin/env python3
"""Offline training experiments for the CRF predictor (2026-11-11, user-approved plan).

Simulates the runtime arm choice (stage 2: features + geometry + the pilot's CRF at budget -> VMAF per arm;
pick argmax of clip(VMAF, 60, 100) - lam * CPU seconds; predicted best < 57 -> speed arm) and looks up what the
picked arm actually scored in the label trials.  A pick that failed only the VMAF-NEG gate falls back to the
NEG-safe arm (as the runtime guard does).

Reported per experiment, on AI / render / screen clips only (the hidden-set content types):
  score   mean clip score of the picks ((VMAF - 60) / 40, 0 below 60 or invalid)
  regret  mean (oracle best clip score - pick's clip score)           <- quality-only regret
  uregret mean utility regret, utility = clip(VMAF,60,100)/40 - lam*CPU/40   <- what the runtime optimises
  cpu     mean CPU seconds of the picked arm's encode
Two evaluations: `test` = frozen family-level test split (train on everything else), and `cv` = repeated grouped
5-fold over all clips (more clips, less noise).  Output: results/train_exp/<name>.json + printed table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from streamenc.encode.ratefit import default_slope  # noqa: E402
from streamenc.features.extract import FEATURE_NAMES  # noqa: E402
from streamenc.ml.crf_predictor import geometry_features  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402

OUT = ROOT / "results/train_exp"
PILOT, SPEED, NEG_SAFE_ARM, HOPELESS = "svt_crf_p8_t1", "svt_crf_p8_t1", "svt_crf_p3_t0_qm_la", 57.0
REL = ("ai", "render", "screen")
LAM = 0.012
DEF_GBR = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=8, l2_regularization=1.0)


def cscore(v: float | None) -> float:
    return 0.0 if v is None else min(max((v - 60.0) / 40.0, 0.0), 1.0)


def load(data: Path, write_geo: bool = False) -> dict:
    feats = pd.read_csv(data / "features.csv")
    labels = {r["clip_id"]: r for r in map(json.loads, open(data / "labels.jsonl"))}
    geo_p = OUT / "geo.json"
    geo = json.loads(geo_p.read_text()) if geo_p.exists() else {}
    for cid, path in zip(feats.clip_id, feats.path):
        if cid not in geo:
            try:
                m = probe_video(path, count_frames=True)
                geo[cid] = geometry_features(m.width, m.height, m.fps, m.nb_frames)
            except Exception as exc:  # noqa: BLE001
                print("probe failed", cid, exc)
    if write_geo:  # only the main process writes the cache (parallel workers only read it), atomically
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = geo_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(geo))
        tmp.replace(geo_p)
    clips = {}
    for r in feats.itertuples():
        if r.clip_id in geo and r.clip_id in labels:
            lb = labels[r.clip_id]
            clips[r.clip_id] = {"x": [float(getattr(r, n)) for n in FEATURE_NAMES] + geo[r.clip_id], "kind": lb["kind"],
                                "tier": lb["tier"], "family": lb["family"], "split": lb["split"], "t": {},
                                "hd": geo[r.clip_id][0] > math.log(1600 * 900), "fps": geo[r.clip_id][1]}
    for t in map(json.loads, open(data / "trials.jsonl")):
        c = clips.get(t["clip_id"])
        if c is None or t.get("prefilter") not in (None, "none"):
            continue
        ok = bool(t.get("passed")) and t.get("vmaf") is not None
        cpu = t.get("enc_cpu_seconds") or (t.get("cpu_seconds") or 0.0) / max(t.get("attempts") or 1, 1)
        c["t"][t["config_id"]] = {"ok": ok, "vmaf": t.get("vmaf"), "cpu": float(cpu or 0.0), "crf": t.get("aim"),
                                  "util": t.get("util"), "stream": bool(t.get("passed_streaming")),
                                  "delta": t.get("vmaf_delta")}
    for c in clips.values():
        p = c["t"].get(PILOT)
        c["pcrf"] = None
        if p and p["stream"] and p["crf"] is not None and p["util"]:
            c["pcrf"] = float(p["crf"]) + math.log(max(float(p["util"]), 1e-3) / 0.985) / default_slope(float(p["crf"]))
    return clips


# ---------------------------------------------------------------- sample weights
def make_weights(clips: dict, ids: list[str], cfg: dict) -> np.ndarray:
    w = np.ones(len(ids))
    kinds = np.array([clips[i]["kind"] for i in ids])
    w[kinds == "other"] *= cfg.get("w_natural", 1.0)
    w[kinds == "generated_hard"] *= cfg.get("w_genhard", 1.0)
    if cfg.get("kind_balance"):  # AI / render / screen get equal total weight
        tot = {k: w[kinds == k].sum() for k in REL}
        mean = np.mean([v for v in tot.values() if v > 0])
        for k in REL:
            if tot[k] > 0:
                w[kinds == k] *= mean / tot[k]
    if cfg.get("family_balance"):  # a family of n rows counts as n ** (1 - p) clips
        fam = defaultdict(int)
        for i in ids:
            fam[clips[i]["family"]] += 1
        w *= np.array([fam[clips[i]["family"]] ** -cfg["family_balance"] for i in ids])
    k = cfg.get("spread_k", 0.0)
    if k:  # clips where the arm choice matters count more: spread of clip score over the non-speed arms
        sp = []
        for i in ids:
            s = [cscore(t["vmaf"]) if t["ok"] else 0.0 for a, t in clips[i]["t"].items() if a != SPEED]
            sp.append((max(s) - min(s)) if s else 0.0)
        w *= 1.0 + k * np.array(sp) / 0.05  # +k per 2 VMAF of spread
    if cfg.get("hd_w"):
        w[np.array([clips[i]["hd"] for i in ids])] *= cfg["hd_w"]
    if cfg.get("hard_w"):
        w[np.array([clips[i]["tier"] == "hard" for i in ids])] *= cfg["hard_w"]
    return w


# ---------------------------------------------------------------- fit + pick
def fit_pick(clips: dict, tr: list[str], te: list[str], cfg: dict) -> dict[str, dict]:
    gp = {**DEF_GBR, **cfg.get("gbr", {})}
    arms = sorted({a for i in tr for a in clips[i]["t"]})
    stage2 = cfg.get("stage2", True)
    tr = [i for i in tr if clips[i]["pcrf"] is not None] if stage2 else tr
    te = [i for i in te if clips[i]["pcrf"] is not None] if stage2 else te
    X = lambda ids: np.array([clips[i]["x"] + ([clips[i]["pcrf"]] if stage2 else []) for i in ids])  # noqa: E731
    Xc = lambda ids: np.array([clips[i]["x"] for i in ids])  # noqa: E731
    Xte, Xcte = X(te), Xc(te)
    mode, ref = cfg.get("target", "abs"), cfg.get("ref", "svt_crf_p4_qm_la")
    vm, cpu = {}, {}
    loss = cfg.get("loss", "squared_error")
    for a in arms:
        ids = [i for i in tr if clips[i]["t"].get(a, {}).get("ok")]
        if len(ids) < 20:
            continue
        w = make_weights(clips, ids, cfg)
        keep = w > 0
        ids, w = [i for i, k in zip(ids, keep) if k], w[keep]
        y = np.array([clips[i]["t"][a]["vmaf"] for i in ids])
        if mode == "delta" and a != ref:
            ids2 = [i for i in ids if clips[i]["t"].get(ref, {}).get("ok")]
            w2 = make_weights(clips, ids2, cfg)
            y2 = np.array([clips[i]["t"][a]["vmaf"] - clips[i]["t"][ref]["vmaf"] for i in ids2])
            vm[a] = ("d", HistGradientBoostingRegressor(loss=loss, random_state=0, **gp).fit(X(ids2), y2, sample_weight=w2))
        elif mode == "score":  # regress the clipped value the score uses (floor 60) instead of raw VMAF
            vm[a] = ("a", HistGradientBoostingRegressor(loss=loss, random_state=0, **gp).fit(X(ids), np.clip(y, 60, 100), sample_weight=w))
        else:
            vm[a] = ("a", HistGradientBoostingRegressor(loss=loss, random_state=0, **gp).fit(X(ids), y, sample_weight=w))
        yc = np.log(np.clip([clips[i]["t"][a]["cpu"] for i in ids], 0.05, None))
        cpu[a] = HistGradientBoostingRegressor(random_state=0, **DEF_GBR).fit(Xc(ids), yc, sample_weight=w)
    # hopeless detection always uses an absolute raw-VMAF model set (best arm < 57)
    if mode in ("delta", "score"):
        hop = {}
        for a in (arms if mode == "score" else [ref]):
            ids = [i for i in tr if clips[i]["t"].get(a, {}).get("ok")]
            w = make_weights(clips, ids, cfg)
            ids, w = [i for i, k in zip(ids, w > 0) if k], w[w > 0]
            hop[a] = HistGradientBoostingRegressor(random_state=0, **gp).fit(
                X(ids), np.array([clips[i]["t"][a]["vmaf"] for i in ids]), sample_weight=w)
        hp = {a: m.predict(Xte) for a, m in hop.items()}
    P = {a: m.predict(Xte) for a, (_, m) in vm.items()}
    if mode == "delta":
        base = hp[ref]
        P = {a: (base + P[a] if vm[a][0] == "d" else base) for a in P}
    C = {a: np.exp(m.predict(Xcte)) for a, m in cpu.items()}
    lam = cfg.get("lam", LAM)
    hop_thr = cfg.get("hopeless", HOPELESS)
    out = {}
    for j, i in enumerate(te):
        pv = {a: float(P[a][j]) for a in P}
        best_raw = max(pv.values()) if mode != "score" else max(float(hp[a][j]) for a in hp)
        if best_raw < hop_thr:
            pick = SPEED
        else:
            u = {a: min(max(pv[a], 60.0), 100.0) - lam * float(C[a][j]) for a in pv if a != SPEED}
            pick = max(u, key=u.get)
        out[i] = outcome(clips[i], pick)
    return out


def outcome(c: dict, pick: str, lam: float = LAM) -> dict:
    T = c["t"]
    t = T.get(pick)
    cpu = t["cpu"] if t else 0.0
    used = pick
    if t and not t["ok"] and t["stream"] and (t["delta"] or 0) > 3.0 and T.get(NEG_SAFE_ARM, {}).get("ok"):
        used, cpu, t = NEG_SAFE_ARM, cpu + T[NEG_SAFE_ARM]["cpu"], T[NEG_SAFE_ARM]  # runtime NEG guard re-fit
    s = cscore(t["vmaf"]) if t and t["ok"] else 0.0
    util = {a: (cscore(x["vmaf"]) if x["ok"] else 0.0) - lam * x["cpu"] / 40.0 for a, x in T.items()}
    oracle = max((cscore(x["vmaf"]) for x in T.values() if x["ok"]), default=0.0)
    return {"pick": pick, "used": used, "score": s, "oracle": oracle, "cpu": cpu,
            "u": s - lam * cpu / 40.0, "u_oracle": max(util.values()) if util else 0.0}


def const_pick(clips: dict, te: list[str], arm: str) -> dict[str, dict]:
    return {i: outcome(clips[i], arm) for i in te}


# ---------------------------------------------------------------- evaluation
def summarise(clips: dict, res: dict[str, dict]) -> dict:
    def agg(ids):
        if not ids:
            return None
        r = [res[i] for i in ids]
        return {"n": len(r), "score": float(np.mean([x["score"] for x in r])),
                "regret": float(np.mean([x["oracle"] - x["score"] for x in r])),
                "uregret": float(np.mean([x["u_oracle"] - x["u"] for x in r])),
                "cpu": float(np.mean([x["cpu"] for x in r])),
                "big_miss": int(sum(x["oracle"] - x["score"] > 0.05 for x in r))}
    ids = [i for i in res if clips[i]["kind"] in REL]
    out = {"all": agg(ids)}
    for k in REL:
        out[k] = agg([i for i in ids if clips[i]["kind"] == k])
    for t in ("easy", "medium", "hard"):
        out[t] = agg([i for i in ids if clips[i]["tier"] == t])
    out["hd"] = agg([i for i in ids if clips[i]["hd"]])
    out["sd"] = agg([i for i in ids if not clips[i]["hd"]])
    out["hfr"] = agg([i for i in ids if clips[i]["fps"] > 31])
    out["picks"] = dict(pd.Series([res[i]["pick"] for i in ids]).value_counts())
    out["picks"] = {k: int(v) for k, v in out["picks"].items()}
    return out


def run_one(args) -> tuple[str, dict]:
    name, cfg, data, reps = args
    clips = load(Path(data))
    ids = sorted(clips)
    tr = [i for i in ids if clips[i]["split"] != "test"]
    te = [i for i in ids if clips[i]["split"] == "test" and clips[i]["kind"] in REL]
    const = cfg.get("const")
    res_t = const_pick(clips, [i for i in te if clips[i]["pcrf"] is not None], const) if const else fit_pick(clips, tr, te, cfg)
    out = {"cfg": cfg, "test": summarise(clips, res_t)}
    cvs = []
    fams = sorted({clips[i]["family"] for i in ids})
    per_clip = defaultdict(list)
    for rep in range(reps):
        rng = np.random.default_rng(100 + rep)
        fold = dict(zip(fams, rng.permutation(len(fams)) % 5))
        res = {}
        for k in range(5):
            trk = [i for i in ids if fold[clips[i]["family"]] != k]
            tek = [i for i in ids if fold[clips[i]["family"]] == k and clips[i]["kind"] in REL]
            res.update(const_pick(clips, [i for i in tek if clips[i]["pcrf"] is not None], const) if const
                       else fit_pick(clips, trk, tek, cfg))
        cvs.append(summarise(clips, res))
        for i, r in res.items():
            per_clip[i].append(r["u_oracle"] - r["u"])
    keys = [k for k in cvs[0] if k != "picks"]
    out["cv"] = {k: ({m: float(np.mean([c[k][m] for c in cvs])) for m in cvs[0][k]} if cvs[0][k] else None) for k in keys}
    out["cv_sd_uregret"] = float(np.std([c["all"]["uregret"] for c in cvs]))
    out["cv_picks"] = cvs[0]["picks"]
    out["cv_clip_uregret"] = {i: float(np.mean(v)) for i, v in per_clip.items()}
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))
    return name, out


EXPERIMENTS: dict[str, dict] = {
    # ---- references
    "const_p2": {"const": "svt_crf_p2_qm_la"}, "const_p3": {"const": "svt_crf_p3_qm_la"},
    "const_p3_t0": {"const": "svt_crf_p3_t0_qm_la"}, "const_p4": {"const": "svt_crf_p4_qm_la"},
    "const_p5": {"const": "svt_crf_p5_qm_la"},
    "base": {},                                            # current training: unweighted, absolute VMAF, stage 2
    "base_stage1": {"stage2": False},
    # ---- mixing
    "nat_0.5": {"w_natural": 0.5}, "nat_0.25": {"w_natural": 0.25}, "nat_0.1": {"w_natural": 0.1}, "nat_0": {"w_natural": 0.0},
    "gen_0": {"w_genhard": 0.0},
    "kindbal": {"kind_balance": True}, "kindbal_nat_0.25": {"kind_balance": True, "w_natural": 0.25},
    "fambal_0.5": {"family_balance": 0.5}, "fambal_1": {"family_balance": 1.0},
    "spread_1": {"spread_k": 1.0}, "spread_3": {"spread_k": 3.0},
    "hd_2": {"hd_w": 2.0}, "hard_2": {"hard_w": 2.0},
    # ---- strategy
    "delta": {"target": "delta"}, "delta_ref_p2": {"target": "delta", "ref": "svt_crf_p2_qm_la"},
    "score_target": {"target": "score"},
    "abs_loss": {"loss": "absolute_error"},
    "gbr_small": {"gbr": {"max_leaf_nodes": 4, "max_iter": 400}},
    "gbr_big": {"gbr": {"max_leaf_nodes": 16, "max_iter": 300}},
    "gbr_slow": {"gbr": {"learning_rate": 0.02, "max_iter": 800}},
    "gbr_fast": {"gbr": {"learning_rate": 0.1, "max_iter": 150}},
    "gbr_leaf4": {"gbr": {"min_samples_leaf": 4}}, "gbr_leaf16": {"gbr": {"min_samples_leaf": 16}},
    "gbr_l2_0": {"gbr": {"l2_regularization": 0.0}}, "gbr_l2_10": {"gbr": {"l2_regularization": 10.0}},
    "gbr_depth3": {"gbr": {"max_depth": 3, "max_leaf_nodes": 8, "max_iter": 500}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/encoder_rank_all"))
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", default="", help="JSON dict name -> cfg of additional experiments")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="re-run experiments that already have a result file")
    a = ap.parse_args()
    exps = dict(EXPERIMENTS)
    if a.extra:
        exps.update(json.loads(a.extra))
    names = [n for n in exps if not a.only or n in a.only.split(",")]
    load(Path(a.data), write_geo=True)  # fill the geometry cache once
    done = {n: json.loads((OUT / f"{n}.json").read_text()) for n in names
            if not a.force and (OUT / f"{n}.json").exists()}
    todo = [n for n in names if n not in done]
    print(f"{len(done)} experiments already done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        results = {**done, **dict(ex.map(run_one, [(n, exps[n], a.data, a.reps) for n in todo]))}
    print(f"{'experiment':20} | test n score regret uregret cpu miss | cv score regret uregret(sd) cpu | cv uregret ai/render/screen | hard  hd")
    for n in names:
        r = results[n]; t, c = r["test"]["all"], r["cv"]["all"]
        g = lambda k: f"{r['cv'][k]['uregret']:.4f}" if r["cv"].get(k) else "  -  "  # noqa: E731
        print(f"{n:20} | {t['n']:3d} {t['score']:.4f} {t['regret']:.4f} {t['uregret']:.4f} {t['cpu']:5.0f} {t['big_miss']:3d} | "
              f"{c['score']:.4f} {c['regret']:.4f} {c['uregret']:.4f}({r['cv_sd_uregret']:.4f}) {c['cpu']:5.0f} | "
              f"{g('ai')} {g('render')} {g('screen')} | {g('hard')} {g('hd')}", flush=True)


if __name__ == "__main__":
    main()
