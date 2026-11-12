#!/usr/bin/env python3
"""Single-segment prefilter probe, offline check (2026-11-12, user: "start single-segment prefilter").

For each clip with a known real filter gain (results/predfix/filter_gain_truth.json): one 2 s keyframe segment (the
middle one when it is full length, else the first) encoded with p8 at the pilot arm's budget CRF, with and without
cblur4_ds85, in parallel; exact VMAF of each against the same source frames (quick_vmaf_gap ref_trim); gain at equal
bits = VMAF_f + vmaf_per_log_bits(VMAF_f) * ln(bytes_u / bytes_f) - VMAF_u. Sanity check first: ref_trim VMAF ==
validator-equivalent VMAF against a lossless copy of the segment.
"""
from __future__ import annotations

import collections, json, math, os, statistics as st, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encode.crf_pipeline import vmaf_per_log_bits  # noqa: E402
from streamenc.encoders.arms import FFMPEG, FFPROBE, build_commands, get_arm  # noqa: E402
from streamenc.filters.policy import filter_vf  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import read_packets_ffprobe, streaming_target_kbps  # noqa: E402
from streamenc.validate.vmaf import compute_vmaf_pair, quick_vmaf_gap  # noqa: E402

SLOPE = 0.14
PILOT = "svt_crf_p8_t1"
truth = json.load(open(ROOT / "results/predfix/filter_gain_truth.json"))
lab = collections.defaultdict(dict)
for l in open(ROOT / "data/encoder_rank_all/trials.jsonl"):
    t = json.loads(l)
    if t.get("passed") and t.get("vmaf") is not None:
        lab[t["clip_id"]][t["config_id"]] = t
work = Path(tempfile.mkdtemp(prefix="fseg-", dir=ROOT / "work"))


def seg_of(meta, gop):
    bounds = [(a, min(a + gop, meta.nb_frames)) for a in range(0, meta.nb_frames, gop)]
    m = bounds[len(bounds) // 2]
    return m if m[1] - m[0] == gop else bounds[0]


def enc(src, meta, gop, target, crf, a, b, filt, tag):
    vf = f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS"
    if filt:
        vf += "," + filter_vf("cblur4_ds85").format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction))
    o = work / f"{tag}.mp4"
    t0 = time.perf_counter()
    for cmd in build_commands(get_arm(PILOT), src, str(o), target, gop, str(o) + ".pass", vf, crf=crf):
        r = run_command(cmd, timeout=900)
        assert r.ok, r.stderr[-300:]
    return str(o), sum(p.size_bytes for p in read_packets_ffprobe(str(o), ffprobe=FFPROBE)), time.perf_counter() - t0


# sanity: ref_trim == validator VMAF on a lossless segment copy
name0 = next(iter(truth)); src0 = truth[name0]["path"]
m0 = probe_video(src0, count_frames=True); g0 = gop_for_timestamps(packet_pts(src0), m0.fps, 2.0); a0, b0 = seg_of(m0, g0)
t0 = streaming_target_kbps(width=m0.width, height=m0.height, fps=m0.fps, bits_per_pixel=0.05)
o0, _, _ = enc(src0, m0, g0, t0, 40.0, a0, b0, False, "sanity")
refseg = str(work / "refseg.mkv")
subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src0, "-vf", f"trim=start_frame={a0}:end_frame={b0},setpts=PTS-STARTPTS",
                "-c:v", "ffv1", "-level", "3", refseg], check=True)
qv = quick_vmaf_gap(src0, o0, subsample=1, reference_meta=m0, threads=(8, 8), exact_pairing=True, ref_trim=(a0, b0))
full = compute_vmaf_pair(refseg, o0)
print(f"sanity {name0} seg {a0}-{b0}: ref_trim quick {qv.vmaf:.4f} / NEG {qv.vmaf_neg:.4f} | validator on segment copy {full.vmaf:.4f} / {full.vmaf_neg:.4f}", flush=True)


def one(name):
    src = truth[name]["path"]
    meta = probe_video(src, count_frames=True)
    gop = gop_for_timestamps(packet_pts(src), meta.fps, 2.0)
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
    t = lab[name][PILOT]
    crf = round((float(t["aim"]) + math.log(max(float(t["util"]), 1e-3) / 0.985) / SLOPE) * 4) / 4
    a, b = seg_of(meta, gop)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as ex:
        (ou, bu, _), (of, bf, _) = ex.map(lambda f: enc(src, meta, gop, target, crf, a, b, f, f"{name}_{int(f)}"), [False, True])
    t1 = time.perf_counter()
    with ThreadPoolExecutor(2) as ex:
        qu, qf = ex.map(lambda o: quick_vmaf_gap(src, o, subsample=1, reference_meta=meta, threads=(8, 8), exact_pairing=True,
                                                 ref_trim=(a, b)), [ou, of])
    t2 = time.perf_counter()
    gain = qf.vmaf + vmaf_per_log_bits(qf.vmaf) * math.log(bu / bf) - qu.vmaf
    return name, round(gain, 2), truth[name]["gain"], round(t1 - t0, 2), round(t2 - t1, 2), f"{meta.width}x{meta.height}@{meta.fps:g}"


rows = [one(n) for n in truth]  # one clip at a time: per-probe wall like the runtime (other clips aside)
print(f"{'clip':26s} {'geom':16s} seg_gain real_gain  enc_s vmaf_s")
ok = 0
for n, g, r, te, tv, geo in sorted(rows, key=lambda x: x[2]):
    good = (g > 0) == (r > 0) or abs(r) < 0.2
    ok += good
    print(f"{n:26s} {geo:16s} {g:+7.2f} {r:+8.2f}  {te:5.2f} {tv:5.2f} {'ok' if good else 'WRONG'}")
take = sum(r for n, g, r, *_ in rows if g > 0.1)
print(f"sign agrees {ok}/{len(rows)}; real gain if filter when seg_gain > 0.1: {take:+.2f} (always {sum(r for n, g, r, *_ in rows):+.2f}, "
      f"oracle {sum(max(r, 0) for n, g, r, *_ in rows):+.2f}); probe wall mean {st.mean(x[3] + x[4] for x in rows):.2f} s max {max(x[3] + x[4] for x in rows):.2f} s")
json.dump(rows, open(ROOT / "results/predfix/filter_gain_seg.json", "w"), indent=1)
