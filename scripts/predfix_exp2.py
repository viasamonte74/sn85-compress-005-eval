#!/usr/bin/env python3
"""Prediction fix, experiment 2 (2026-11-12): is "measure the pilot" accurate with what the runtime really has?

Held-out clips of the rc3 4-clip batched run: re-encode the p8 pilots at the runtime's pilot CRFs (whole clip,
same hard prefilter the runtime pilots had), measure budget use (validator packet rules) and quick VMAF
(subsample 5, 8 decode + 4 libvmaf threads, as quick_vmaf_gap), estimate the p8 VMAF at budget (two pilots:
linear in ln(use); one: vmaf_per_log_bits slope), feed variant D (exp1) and compare with the labels.
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics as st
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("STREAMENC_QUICK_VMAF_THREADS", "8,4")
SUB = int(os.environ.get("PREDFIX_SUB", "5"))            # quick VMAF subsample
NOFILTER = os.environ.get("PREDFIX_NOFILTER", "0") == "1"  # pilots without the stage-1 hard prefilter
TAG = os.environ.get("PREDFIX_TAG", "exp2")
from predfix_diag import parse_run  # noqa: E402
from predfix_exp1 import AI_HARD, PILOT, SLOPE, gbr  # noqa: E402
from streamenc.encode.crf_pipeline import NEG_SUBSAMPLE, vmaf_per_log_bits  # noqa: E402
from streamenc.encode.ratefit import budget_use  # noqa: E402
from streamenc.encoders.arms import FFPROBE, build_commands, get_arm  # noqa: E402
from streamenc.features.extract import FEATURE_NAMES  # noqa: E402
from streamenc.filters.policy import filter_vf  # noqa: E402
from streamenc.ml.crf_predictor import geometry_features  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import read_packets_ffprobe, streaming_target_kbps  # noqa: E402
from streamenc.validate.vmaf import quick_vmaf_gap  # noqa: E402

import pandas as pd  # noqa: E402


def pilot_points(run: Path) -> dict:
    """clip -> (pilot points [(crf, use)], pilots prefiltered?) from the run log."""
    import ast, re
    rows = list(map(json.loads, open(run / "per_clip.jsonl")))
    batches, cur, it = [], None, None
    for line in open(run / "contender.log"):
        if line.startswith("===== batch"):
            cur = []; batches.append(cur); it = None; continue
        m = re.search(r"crf_pipeline\] item \w+: \S+ crf=\S+ util=(\S+) ", line)
        if m:
            it = {"util": m.group(1), "h": []}; cur.append(it); continue
        if it is not None and "{'tag'" in line:
            it["h"].append(ast.literal_eval(line[line.index("{"):].strip()))
        elif it is not None and "crf_pipeline" not in line:
            it = None
    out = {}
    for r in rows:
        name = Path(r["ref"]).stem
        if name in out or r.get("util") is None:
            continue
        c = [i for i in batches[r["batch"]] if i["util"] == str(round(r["util"], 4))]
        if len(c) != 1:
            continue
        h = c[0]["h"]
        pts = [(x["crf"], x["use"]) for x in h if x["tag"] == "pilot" and "use" in x]
        pre = any(x["tag"] == "hard_prefilter" and x.get("stage") == "stage1" for x in h)
        if pts:
            out[name] = (r["ref"], pts, pre)
    return out


def main() -> int:
    run = ROOT / "results" / "e2e_batched" / "rc3_heldout"
    pp = pilot_points(run)
    work = Path(tempfile.mkdtemp(prefix="predfix2-", dir=ROOT / "work"))
    jobs = [(name, src, crf, pre) for name, (src, pts, pre) in pp.items() for crf, _ in pts]
    print(f"{len(pp)} clips with runtime pilots, {len(jobs)} pilot encodes", flush=True)

    def enc(job):
        name, src, crf, pre = job
        meta = probe_video(src, count_frames=True)
        gop = gop_for_timestamps(packet_pts(src), meta.fps, 2.0)
        target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
        vf = filter_vf("cblur4_ds85").format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction)) if pre and not NOFILTER else None
        o = work / f"{name}_{int(round(crf * 100))}.mp4"
        for cmd in build_commands(get_arm(PILOT), src, str(o), target, gop, str(work / f"p_{name}_{crf}"), vf, crf=crf):
            r = run_command(cmd, timeout=1800)
            assert r.ok, r.stderr[-300:]
        sz = [p.size_bytes for p in read_packets_ffprobe(str(o), ffprobe=FFPROBE)]
        n, fps = len(sz), meta.fps
        avg = 8.0 * sum(sz) / (n / fps) / 1000.0
        win = max(1, round(2.0 * fps)); run_, best = sum(sz[:win]), sum(sz[:win])
        for k in range(win, n):
            run_ += sz[k] - sz[k - win]; best = max(best, run_)
        peak = 8.0 * best / (win / fps) / 1000.0 if n > win else avg
        use = budget_use(avg, peak, target, 1.5)
        q = quick_vmaf_gap(src, str(o), subsample=SUB, reference_meta=meta)
        os.remove(o)
        return name, crf, use, q.vmaf, q.wall_seconds

    with ThreadPoolExecutor(6) as ex:
        meas = list(ex.map(enc, jobs))
    by = collections.defaultdict(list)
    for name, crf, use, v, w in meas:
        by[name].append((use, v))
    est, est1 = {}, {}
    for name, pts in by.items():
        u, v = min(pts, key=lambda p: abs(math.log(p[0] / 0.985)))
        est1[name] = v + vmaf_per_log_bits(v) * math.log(0.985 / u)  # one pilot + slope rule
        pts.sort()
        if len(pts) >= 2 and abs(math.log(pts[-1][0] / pts[0][0])) > 0.02:
            (u0, v0), (u1, v1) = pts[0], pts[-1]
            s = (v1 - v0) / math.log(u1 / u0)
            u, v = min(pts, key=lambda p: abs(math.log(p[0] / 0.985)))
            est[name] = v + s * math.log(0.985 / u)
        else:
            u, v = min(pts, key=lambda p: abs(math.log(p[0] / 0.985)))
            est[name] = v + vmaf_per_log_bits(v) * math.log(0.985 / u)
    # variant D trained on the train split (p8 VMAF at budget from the trials), applied with the runtime estimate
    d = ROOT / "data" / "encoder_rank_all"
    feats = pd.read_csv(d / "features.csv").drop_duplicates("clip_id", keep="last").set_index("clip_id")
    labels = {json.loads(l)["clip_id"]: json.loads(l) for l in open(d / "labels.jsonl")}
    trials = collections.defaultdict(dict)
    for l in open(d / "trials.jsonl"):
        t = json.loads(l)
        if t.get("passed") and t.get("vmaf") is not None:
            trials[t["clip_id"]][t["config_id"]] = t
    geo = {}

    def X(c):
        if c not in geo:
            m = probe_video(feats.loc[c, "path"], count_frames=True)
            geo[c] = [float(feats.loc[c, n]) for n in FEATURE_NAMES] + geometry_features(m.width, m.height, m.fps, m.nb_frames, 0.05)
        return geo[c]

    def crf8(c):
        p = trials[c][PILOT]
        return float(p["aim"]) + math.log(max(float(p["util"]), 1e-3) / 0.985) / SLOPE
    train = [c for c in feats.index if c in labels and labels[c]["kind"] != "other" and labels[c]["split"] != "test"
             and PILOT in trials.get(c, {})]
    test = [c for c in est if c in feats.index and PILOT in trials.get(c, {})]
    errs = collections.defaultdict(list)
    for a in sorted({a for c in train for a in trials[c]}):
        tr = [c for c in train if a in trials[c]]
        Xtr = np.array([X(c) + [crf8(c), trials[c][PILOT]["vmaf"], math.log(trials[c][PILOT]["util"])] for c in tr])
        ytr = np.array([trials[c][a]["vmaf"] - trials[c][PILOT]["vmaf"] for c in tr])
        m = gbr().fit(Xtr, ytr)
        for c in test:
            if a not in trials[c]:
                continue
            g = "ai_hard" if c.startswith(AI_HARD) else labels[c]["kind"]
            lab = trials[c][a]["vmaf"]
            ideal = trials[c][PILOT]["vmaf"] + m.predict(np.array([X(c) + [crf8(c), trials[c][PILOT]["vmaf"], math.log(trials[c][PILOT]["util"])]]))[0]
            real = est[c] + m.predict(np.array([X(c) + [crf8(c), est[c], math.log(0.985)]]))[0]
            real1 = est1[c] + m.predict(np.array([X(c) + [crf8(c), est1[c], math.log(0.985)]]))[0]
            errs[(g, "ideal")].append(ideal - lab); errs[(g, "runtime")].append(real - lab); errs[(g, "rt_1pt")].append(real1 - lab)
    for nm, E in (("2-pilot", est), ("1-pilot+slope", est1)):
        v8e = [E[c] - trials[c][PILOT]["vmaf"] for c in test]
        print(f"p8 VMAF at budget, {nm} estimate - label: n={len(v8e)} bias {st.mean(v8e):+.2f} MAE {st.mean(map(abs, v8e)):.2f} max {max(map(abs, v8e)):.2f}")
    print(f"settings: subsample {SUB}, pilots unfiltered {NOFILTER}, pairing {os.environ.get('STREAMENC_QUICK_VMAF_PAIRING', 'old')}")
    for g in sorted({k[0] for k in errs}):
        for v in ("ideal", "runtime", "rt_1pt"):
            e = errs[(g, v)]
            print(f"  {g:8s} {v:8s} n={len(e):4d} bias {st.mean(e):+5.2f} MAE {st.mean(map(abs, e)):5.2f} |err|>5: {sum(abs(x) > 5 for x in e)}")
    walls = [w for *_, w in meas]
    print(f"quick VMAF per pilot: mean {st.mean(walls):.2f} s, max {max(walls):.2f} s (6 encodes in parallel)")
    (ROOT / "results" / "predfix" / f"{TAG}.json").write_text(json.dumps({"meas": meas, "est": est, "est1": est1}, indent=1))
    os.rmdir(work) if not any(work.iterdir()) else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
