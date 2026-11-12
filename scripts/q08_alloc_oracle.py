#!/usr/bin/env python3
"""q08 segment-allocation oracle (2026-11-12, user: "solve q08 like the NW-AV1 research plan").

The runtime encodes each 2 s keyframe segment independently and joins them without re-encoding. The validator's
VMAF is the mean of per-frame scores, and a frame's score depends only on that frame (plus reference-only motion),
so the clip VMAF of ANY per-segment CRF mix is the frame-weighted mean of the segment scores measured on uniform-CRF
joins. Packet sizes are the concatenation of the segment packets, so average and 2 s peak are exact too.

For one clip and one (arm, hard prefilter): encode every segment at every CRF of a quarter-step grid exactly like
crf_pipeline._segmented_final, join each uniform level, measure packets + per-frame VMAF / VMAF-NEG the validator's
way (canonical y4m, vmaf_v0.6.1 / v0.6.1neg, n_subsample=1, mean), then search all mixes under the real limits.
Finally the best mixes are joined for real and scored with compute_vmaf_pair to prove the prediction is exact.

  python scripts/q08_alloc_oracle.py --clip /workspace/dataset-src/practice_pack_005/q08/clip.mkv --tag q08_cb4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import FFMPEG, FFPROBE, build_commands, get_arm  # noqa: E402
from streamenc.filters.policy import filter_vf  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import read_packets_ffprobe, streaming_target_kbps  # noqa: E402
from streamenc.validate.vmaf import (DEFAULT_VMAF_MODEL, SECONDARY_VMAF_MODEL, _escape_filter_option,  # noqa: E402
                                     canonicalize, compute_vmaf_pair, default_ffmpeg_with_vmaf)


def per_frame_vmaf(ref_y4m: str, cand_y4m: str, model: str) -> tuple[list[float], float]:
    """compute_vmaf's exact graph, returning the per-frame scores and the pooled mean."""
    ff = default_ffmpeg_with_vmaf()
    with tempfile.TemporaryDirectory(prefix="vmafpf-") as tmp:
        log = os.path.join(tmp, "v.json")
        graph = (f"[0:v:0][1:v:0]libvmaf=model={_escape_filter_option(model)}"
                 f":log_fmt=json:log_path={_escape_filter_option(log)}:n_threads=1:n_subsample=1:pool=mean")
        r = run_command([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", cand_y4m, "-i", ref_y4m,
                         "-filter_complex", graph, "-f", "null", "-"], timeout=3600)
        if not r.ok:
            raise RuntimeError(r.stderr[-300:])
        doc = json.loads(Path(log).read_text())
    return [float(f["metrics"]["vmaf"]) for f in doc["frames"]], float(doc["pooled_metrics"]["vmaf"]["mean"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--arm", default="svt_crf_p3_qm_la_lqb")
    ap.add_argument("--prefilter", default="cblur4_ds85")
    ap.add_argument("--crf-lo", type=float, default=55.0)
    ap.add_argument("--crf-hi", type=float, default=58.5)
    ap.add_argument("--slack", type=float, default=0.008)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--verify", type=int, default=3)
    args = ap.parse_args()

    work = ROOT / "work" / "alloc_oracle" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    src = args.clip
    meta = probe_video(src, count_frames=True)
    n = meta.nb_frames
    pts = packet_pts(src)
    gop = gop_for_timestamps(pts, meta.fps, 2.0)
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
    bounds = [(a, min(a + gop, n)) for a in range(0, n, gop)]
    vf_hard = filter_vf(args.prefilter).format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction)) \
        if args.prefilter != "none" else None
    arm = get_arm(args.arm)
    q = 0.25
    grid = [round(args.crf_lo + q * i, 2) for i in range(int(round((args.crf_hi - args.crf_lo) / q)) + 1)]
    fr = Fraction(str(meta.fps_fraction)) if getattr(meta, "fps_fraction", None) else Fraction(meta.fps).limit_denominator(1001)
    print(f"{args.tag}: {meta.width}x{meta.height} {meta.fps} fps, {n} frames, gop {gop}, segments {bounds}, "
          f"target {target:.2f} kbps, grid {grid[0]}..{grid[-1]} ({len(grid)})", flush=True)

    def enc(job):
        i, c = job
        a, b = bounds[i]
        vf = f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS" + (f",{vf_hard}" if vf_hard else "")
        o = work / f"s{i:03d}_{int(round(c * 100))}.ivf"
        if o.exists() and o.stat().st_size > 0:
            return
        for cmd in build_commands(arm, src, str(o), target, gop, str(work / f"pass{i}_{int(c * 100)}"), vf, crf=c):
            cmd[len(cmd) - 1 - cmd[::-1].index("-f") + 1] = "ivf"
            r = run_command(cmd, timeout=3600)
            if not r.ok:
                raise RuntimeError(r.stderr[-300:])

    def join(combo: tuple[float, ...], name: str) -> str:
        lst = work / f"{name}.txt"
        lst.write_text("".join(f"file '{work / f's{i:03d}_{int(round(c * 100))}.ivf'}'\n" for i, c in enumerate(combo)))
        o = str(work / f"{name}.mp4")
        r = run_command([FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                         "-i", str(lst), "-map", "0:v:0", "-c", "copy", "-video_track_timescale", str(fr.numerator),
                         "-f", "mp4", o], timeout=600)
        if not r.ok:
            raise RuntimeError(r.stderr[-300:])
        return o

    with ThreadPoolExecutor(args.par) as ex:
        list(ex.map(enc, [(i, c) for c in grid for i in range(len(bounds))]))
    ref_y4m = str(work / "ref.y4m")
    if not os.path.exists(ref_y4m):
        canonicalize(src, ref_y4m)

    def level(c):
        o = join(tuple([c] * len(bounds)), f"u{int(round(c * 100))}")
        sizes = [p.size_bytes for p in read_packets_ffprobe(o, ffprobe=FFPROBE)]
        cy = str(work / f"u{int(round(c * 100))}.y4m")
        canonicalize(o, cy)
        v, vm = per_frame_vmaf(ref_y4m, cy, DEFAULT_VMAF_MODEL)
        g, gm = per_frame_vmaf(ref_y4m, cy, SECONDARY_VMAF_MODEL)
        os.remove(cy)
        assert len(sizes) == len(v) == n, (len(sizes), len(v), n)
        return c, sizes, v, g, vm, gm

    with ThreadPoolExecutor(min(args.par, len(grid))) as ex:
        levels = list(ex.map(level, grid))
    L = {c: (s, v, g) for c, s, v, g, _, _ in levels}
    for c, s, v, g, vm, gm in levels:
        assert abs(sum(v) / n - vm) < 1e-4, "pooled VMAF is not the per-frame mean"
    dur = n / meta.fps
    win = max(1, round(2.0 * meta.fps))
    cap_avg = target * (1.0 + args.slack)

    def evaluate(combo):
        sizes, v, g = [], [], []
        for i, c in enumerate(combo):
            a, b = bounds[i]
            s_, v_, g_ = L[c]
            sizes += s_[a:b]; v += v_[a:b]; g += g_[a:b]
        avg = 8.0 * sum(sizes) / dur / 1000.0
        run = sum(sizes[:win]); best = run
        for k in range(win, len(sizes)):
            run += sizes[k] - sizes[k - win]; best = max(best, run)
        peak = 8.0 * best / (win / meta.fps) / 1000.0 if len(sizes) > win else avg
        vm, gm = sum(v) / n, sum(g) / n
        return {"combo": combo, "avg_kbps": avg, "util": avg / target, "peak_ratio": peak / target, "vmaf": vm,
                "neg": gm, "gap": vm - gm}

    allc = [evaluate(c) for c in itertools.product(grid, repeat=len(bounds))]
    ok = [r for r in allc if r["avg_kbps"] <= cap_avg and r["peak_ratio"] <= 1.5 and r["gap"] <= 3.0]
    ok.sort(key=lambda r: -r["vmaf"])
    uni = [evaluate(tuple([c] * len(bounds))) for c in grid]
    uni_ok = [r for r in uni if r["avg_kbps"] <= cap_avg and r["peak_ratio"] <= 1.5]
    best_uni = max(uni_ok, key=lambda r: r["vmaf"]) if uni_ok else None
    print("uniform levels (crf: util, vmaf, gap):", flush=True)
    for r in uni:
        print(f"  {r['combo'][0]:6.2f}: util {r['util']:.4f} peak {r['peak_ratio']:.3f} vmaf {r['vmaf']:.3f} gap {r['gap']:.3f}")
    print(f"best uniform under cap: {best_uni and (best_uni['combo'][0], round(best_uni['util'], 4), round(best_uni['vmaf'], 3))}")
    print("best mixes under cap (avg <= %.4f x target, peak <= 1.5, gap <= 3):" % (1 + args.slack))
    for r in ok[:8]:
        print(f"  {r['combo']}: util {r['util']:.4f} peak {r['peak_ratio']:.3f} vmaf {r['vmaf']:.3f} gap {r['gap']:.3f}")
    # equal-CRF-spread view: best mix with max CRF spread <= 0.25 / 0.5
    for sp in (0.25, 0.5):
        cand = [r for r in ok if max(r["combo"]) - min(r["combo"]) <= sp + 1e-9]
        if cand:
            print(f"best mix with CRF spread <= {sp}: {cand[0]['combo']} util {cand[0]['util']:.4f} vmaf {cand[0]['vmaf']:.3f}")
    ver = []
    for k, r in enumerate(ok[:args.verify]):
        o = join(r["combo"], f"verify{k}")
        sz = [p.size_bytes for p in read_packets_ffprobe(o, ffprobe=FFPROBE)]
        pair = compute_vmaf_pair(src, o)
        avg = 8.0 * sum(sz) / dur / 1000.0
        ver.append({"combo": r["combo"], "pred_vmaf": r["vmaf"], "real_vmaf": pair.vmaf, "pred_neg": r["neg"],
                    "real_neg": pair.vmaf_neg, "pred_util": r["util"], "real_util": avg / target})
        print(f"verify {r['combo']}: vmaf pred {r['vmaf']:.6f} real {pair.vmaf:.6f} | neg pred {r['neg']:.6f} real "
              f"{pair.vmaf_neg:.6f} | util pred {r['util']:.6f} real {avg / target:.6f}", flush=True)
    out = ROOT / "results" / "alloc_oracle" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clip": src, "arm": args.arm, "prefilter": args.prefilter, "target_kbps": target,
                               "bounds": bounds, "grid": grid, "uniform": uni, "best_uniform": best_uni,
                               "best_mixes": ok[:50], "verify": ver}, indent=1, default=list))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
