#!/usr/bin/env python3
"""Q1 (2026-11-12, user: "try fix weak spots"): can a fast encode predict whether the hard prefilter (cblur4_ds85)
helps a clip? Truth = actual filtered final VMAF - unfiltered label of the same arm (results/predfix/filter_gain_truth.json).
Proxy per arm (p8 = pilot arm, p6): unfiltered at the arm's budget CRF; filtered at the same CRF and 1.0 lower;
exact VMAF (validator pairing, every frame); filtered VMAF interpolated in ln(use) to the unfiltered use."""
from __future__ import annotations

import collections, json, math, os, statistics as st, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encode.ratefit import budget_use  # noqa: E402
from streamenc.encoders.arms import FFPROBE, build_commands, get_arm  # noqa: E402
from streamenc.filters.policy import filter_vf  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import read_packets_ffprobe, streaming_target_kbps  # noqa: E402
from streamenc.validate.vmaf import quick_vmaf_gap  # noqa: E402

SLOPE = 0.14
truth = json.load(open(ROOT / "results/predfix/filter_gain_truth.json"))
lab = collections.defaultdict(dict)
for l in open(ROOT / "data/encoder_rank_all/trials.jsonl"):
    t = json.loads(l)
    if t.get("passed") and t.get("vmaf") is not None:
        lab[t["clip_id"]][t["config_id"]] = t
work = Path(tempfile.mkdtemp(prefix="fgain-", dir=ROOT / "work"))
ARMS = os.environ.get("FGAIN_ARMS", "svt_crf_p8_t1,svt_crf_p6_qm_la").split(",")
sys.path.insert(0, str(ROOT))
from streamenc.encode.crf_pipeline import vmaf_per_log_bits  # noqa: E402


def one(job):
    name, arm, crf, filt = job
    src = truth[name]["path"]
    meta = probe_video(src, count_frames=True)
    gop = gop_for_timestamps(packet_pts(src), meta.fps, 2.0)
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
    vf = filter_vf("cblur4_ds85").format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction)) if filt else None
    o = work / f"{name}_{arm}_{int(round(crf * 100))}_{int(filt)}.mp4"
    for cmd in build_commands(get_arm(arm), src, str(o), target, gop, str(o) + ".pass", vf, crf=crf):
        r = run_command(cmd, timeout=1800)
        assert r.ok, r.stderr[-300:]
    sz = [p.size_bytes for p in read_packets_ffprobe(str(o), ffprobe=FFPROBE)]
    n, fps = len(sz), meta.fps
    avg = 8.0 * sum(sz) / (n / fps) / 1000.0
    win = max(1, round(2.0 * fps)); run_, best = sum(sz[:win]), sum(sz[:win])
    for k in range(win, n):
        run_ += sz[k] - sz[k - win]; best = max(best, run_)
    peak = 8.0 * best / (win / fps) / 1000.0 if n > win else avg
    q = quick_vmaf_gap(src, str(o), subsample=1, reference_meta=meta, threads=(8, 8), exact_pairing=True)
    os.remove(o)
    return name, arm, crf, filt, budget_use(avg, peak, target, 1.5), q.vmaf


jobs = []
for name in truth:
    for arm in ARMS:
        t = lab[name].get(arm)
        if t is None:
            continue
        c = round((float(t["aim"]) + math.log(max(float(t["util"]), 1e-3) / 0.985) / SLOPE) * 4) / 4
        jobs += [(name, arm, c, False), (name, arm, c, True), (name, arm, c - 1.0, True)]
with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(one, jobs))
by = collections.defaultdict(dict)
for name, arm, crf, filt, use, v in res:
    by[(name, arm)][(crf, filt)] = (use, v)
rows = []
for (name, arm), d in by.items():
    (uu, vu), = [v for (c, f), v in d.items() if not f]
    fl = sorted(v for (c, f), v in d.items() if f)
    (u0, v0), (u1, v1) = fl[0], fl[-1]
    s = (v1 - v0) / math.log(u1 / u0) if abs(math.log(u1 / u0)) > 0.01 else 0.55 * (100 - v1)
    vf_at_u = v1 + s * math.log(uu / u1)
    (uf, vff), = [v for (c, f), v in d.items() if f and abs(c - [c_ for (c_, f_) in d if not f_][0]) < 1e-9]
    rule = vff + vmaf_per_log_bits(vff) * math.log(uu / uf) - vu                 # one pair + slope rule
    rows.append((name, arm, round(vf_at_u - vu, 2), truth[name]["gain"], round(rule, 2)))
print(f"{'clip':26s} {'proxy':18s} proxy_gain  real_gain")
agree = collections.Counter()
agree_r = collections.Counter()
for name, arm, pg, rg, rl in sorted(rows, key=lambda r: (r[1], r[3])):
    ok = (pg > 0) == (rg > 0) or abs(rg) < 0.2
    okr = (rl > 0) == (rg > 0) or abs(rg) < 0.2
    agree[arm] += ok; agree_r[arm] += okr
    print(f"{name:26s} {arm:18s} {pg:+8.2f}  {rg:+8.2f} {'ok' if ok else 'WRONG'}   1-pair rule {rl:+6.2f} {'ok' if okr else 'WRONG'}")
for arm in ARMS:
    sel = [(pg, rg) for n, a, pg, rg, rl in rows if a == arm]
    selr = [(rl, rg) for n, a, pg, rg, rl in rows if a == arm]
    print(f"{arm}: 1-pair rule sign agrees {agree_r[arm]}/{len(selr)}; gain if filter when rule>0: {sum(rg for rl, rg in selr if rl > 0):+.2f}")
    # value of deciding by the proxy: sum of real gains where proxy says "filter" vs always / never
    take = sum(rg for pg, rg in sel if pg > 0)
    print(f"{arm}: sign agrees {agree[arm]}/{len(sel)}; total real gain if filter only when proxy>0: {take:+.2f} "
          f"(always filter {sum(rg for _, rg in sel):+.2f}, oracle {sum(max(rg, 0) for _, rg in sel):+.2f})")
json.dump(rows, open(ROOT / "results/predfix/filter_gain_proxy.json", "w"), indent=1)
