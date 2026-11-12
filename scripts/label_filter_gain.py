#!/usr/bin/env python3
"""Label the hard-prefilter gain of every non-natural labelled clip (2026-11-12, user: "Go ahead").

Label = the single-segment probe of scripts/filter_gain_seg.py (sign right on 13 of 14 clips with a known real gain):
p8 (pilot arm) at its budget CRF, one 2 s keyframe segment (middle one if full length, else the first), with and
without cblur4_ds85; exact VMAF against the same source frames; gain at equal bits. Appends to
data/filter_gain/labels.jsonl (resumable). Used to train a zero-runtime-cost prefilter decision model.
"""
from __future__ import annotations

import collections, json, math, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encode.crf_pipeline import vmaf_per_log_bits  # noqa: E402
from streamenc.encoders.arms import FFPROBE, build_commands, get_arm  # noqa: E402
from streamenc.filters.policy import filter_vf  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import read_packets_ffprobe, streaming_target_kbps  # noqa: E402
from streamenc.validate.vmaf import quick_vmaf_gap  # noqa: E402

SLOPE = 0.14
PILOT = "svt_crf_p8_t1"
FILTER = "cblur4_ds85"
D = ROOT / "data" / "encoder_rank_all"
OUT = ROOT / "data" / "filter_gain" / "labels.jsonl"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["clip_id"] for l in open(OUT)} if OUT.exists() else set()
    labels = {}
    for l in open(D / "labels.jsonl"):
        r = json.loads(l); labels[r["clip_id"]] = r
    p8 = {}
    for l in open(D / "trials.jsonl"):
        t = json.loads(l)
        if t["config_id"] == PILOT and t.get("passed") and t.get("vmaf") is not None:
            p8[t["clip_id"]] = t
    feats = pd.read_csv(D / "features.csv").drop_duplicates("clip_id", keep="last")
    todo = [(r.clip_id, r.path) for r in feats.itertuples()
            if r.clip_id in labels and labels[r.clip_id]["kind"] != "other" and r.clip_id in p8 and r.clip_id not in done
            and os.path.exists(r.path)]
    print(f"{len(done)} done, {len(todo)} to label", flush=True)
    work = Path(tempfile.mkdtemp(prefix="fglab-", dir=ROOT / "work"))

    def one(job):
        cid, src = job
        t0 = time.perf_counter()
        meta = probe_video(src, count_frames=True)
        gop = gop_for_timestamps(packet_pts(src), meta.fps, 2.0)
        target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
        t = p8[cid]
        crf = round((float(t["aim"]) + math.log(max(float(t["util"]), 1e-3) / 0.985) / SLOPE) * 4) / 4
        crf = min(63.0, max(1.0, crf))
        bounds = [(a, min(a + gop, meta.nb_frames)) for a in range(0, meta.nb_frames, gop)]
        sa, sb = bounds[len(bounds) // 2]
        if sb - sa != gop:
            sa, sb = bounds[0]
        fvf = filter_vf(FILTER).format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction))
        res = {}
        for filt in (False, True):
            vf = f"trim=start_frame={sa}:end_frame={sb},setpts=PTS-STARTPTS" + (f",{fvf}" if filt else "")
            o = work / f"{cid}_{int(filt)}.mp4"
            for cmd in build_commands(get_arm(PILOT), src, str(o), target, gop, str(o) + ".pass", vf, crf=crf):
                r = run_command(cmd, timeout=1800)
                if not r.ok:
                    raise RuntimeError(r.stderr[-200:])
            b = sum(p.size_bytes for p in read_packets_ffprobe(str(o), ffprobe=FFPROBE))
            q = quick_vmaf_gap(src, str(o), subsample=1, reference_meta=meta, threads=(4, 4), exact_pairing=True, ref_trim=(sa, sb))
            res[filt] = (b, q.vmaf, q.vmaf_neg)
            os.remove(o)
        (bu, vu, nu), (bf, vf_, nf) = res[False], res[True]
        gain = vf_ + vmaf_per_log_bits(vf_) * math.log(max(bu, 1) / max(bf, 1)) - vu
        return {"clip_id": cid, "family": labels[cid]["family"], "kind": labels[cid]["kind"], "split": labels[cid]["split"],
                "crf": crf, "seg": [sa, sb], "bytes_u": bu, "bytes_f": bf, "vmaf_u": round(vu, 4), "vmaf_f": round(vf_, 4),
                "neg_u": round(nu, 4), "neg_f": round(nf, 4), "gain": round(gain, 4), "seconds": round(time.perf_counter() - t0, 2)}

    n_ok = n_err = 0
    with ThreadPoolExecutor(int(os.environ.get("FGL_PAR", "6"))) as ex, open(OUT, "a") as f:
        futs = {ex.submit(one, j): j for j in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result(); f.write(json.dumps(rec) + "\n"); f.flush(); n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_err += 1; print("ERR", futs[fut][0], str(exc)[-150:], flush=True)
            if (n_ok + n_err) % 50 == 0:
                print(f"{n_ok + n_err}/{len(todo)} ({n_err} errors)", flush=True)
    print(f"labelled {n_ok}, errors {n_err}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
