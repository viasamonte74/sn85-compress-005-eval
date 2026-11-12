#!/usr/bin/env python3
"""Broad encoder exploration on hard clips (all encoders in the bundled ffmpeg n8.1).

For each (clip, arm): encode at the streaming budget (aim ladder until the
official streaming gates pass), then VMAF + VMAF-NEG. Writes one CSV row per
trial so the label catalog can be rebuilt from measured winners.

  python scripts/hard_encoder_explore.py --clips ai_hard_01,ai_hard_18 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.probe import probe_video  # noqa: E402
from streamenc.score import clip_score  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import (  # noqa: E402
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)
from streamenc.validate.vmaf import compute_vmaf_pair  # noqa: E402

FFBIN = ROOT / "third_party" / "ffmpeg-n8.1-linux64-gpl" / "bin"
FFMPEG = str(FFBIN / "ffmpeg")
FFPROBE = str(FFBIN / "ffprobe")
AIMS = (0.985,)
MAX_FIT_ATTEMPTS = 5
FIT_GOAL = 0.99  # target utilisation of the streaming budget
FIT_LOW = 0.97  # accept once a valid encode reaches this utilisation
THREADS = int(os.environ.get("EXPLORE_THREADS", "8"))

DATASET = Path("/workspace/dataset")


def clip_path(cid: str) -> Path:
    for sub in ("ai", "ai1080", "render", "screen", "generated_hard", "ai_gen2", "open_clips"):
        p = DATASET / sub / f"{cid}.mkv"
        if p.exists():
            return p
    raise FileNotFoundError(cid)


# ---------------------------------------------------------------- arms
# Each arm: (codec, two_pass, params). Builders below turn this into argv lists.
ARMS: dict[str, dict] = {
    # SVT-AV1 v4.3
    "svt_p6_t1": {"codec": "libsvtav1", "preset": 6, "svt": {"tune": 1}},
    "svt_p5_t1": {"codec": "libsvtav1", "preset": 5, "svt": {"tune": 1}},
    "svt_p4_t1": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1}},
    "svt_p3_t1": {"codec": "libsvtav1", "preset": 3, "svt": {"tune": 1}},
    "svt_p2_t1": {"codec": "libsvtav1", "preset": 2, "svt": {"tune": 1}},
    "svt_p4_t0": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 0}},
    "svt_p4_t1_la120": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120}},
    "svt_p4_t1_vboost": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-variance-boost": 1}},
    "svt_p4_t1_fg8": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "film-grain": 8, "film-grain-denoise": 1}},
    "svt_p4_t1_fg8nd": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "film-grain": 8, "film-grain-denoise": 0}},
    "svt_p4_t1_sr12": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "superres-mode": 1, "superres-denom": 12}},
    "svt_p4_t1_sr_auto": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "superres-mode": 4}},
    "svt_p4_t1_tf0": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-tf": 0}},
    "svt_p4_t1_qm": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0}},
    # SVT-AV1 v4.3 combinations / extra knobs
    "svt_p3_combo": {"codec": "libsvtav1", "preset": 3, "svt": {"tune": 1, "lookahead": 120, "superres-mode": 4, "enable-qm": 1, "qm-min": 0}},
    "svt_p4_combo": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "superres-mode": 4, "enable-qm": 1, "qm-min": 0}},
    "svt_p4_la120_qm": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "enable-qm": 1, "qm-min": 0}},
    "svt_p4_la120_tfs3": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "tf-strength": 3}},
    "svt_p4_la120_open": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "irefresh-type": 1}},
    "svt_p4_la120_10b": {"codec": "libsvtav1", "preset": 4, "pix_fmt": "yuv420p10le", "svt": {"tune": 1, "lookahead": 120}},
    "svt_p4_la120_t0": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 0, "lookahead": 120}},
    # SVT-AV1 v1.7 (system ffmpeg 6.1) — old label winners, now with rate fit
    "svt17_p5_t1": {"codec": "libsvtav1", "bin": "system", "preset": 5, "svt": {"tune": 1}},
    "svt17_p5_t0": {"codec": "libsvtav1", "bin": "system", "preset": 5, "svt": {"tune": 0}},
    "svt17_p4_t0": {"codec": "libsvtav1", "bin": "system", "preset": 4, "svt": {"tune": 0}},
    # libaom (2-pass VBR)
    "aom_c4_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 4, "tune": "psnr"},
    "aom_c3_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 3, "tune": "psnr"},
    "aom_c4_ssim": {"codec": "libaom-av1", "two_pass": True, "cpu": 4, "tune": "ssim"},
    "aom_c5_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 5, "tune": "psnr"},
    "aom_c4_psnr_dn": {"codec": "libaom-av1", "two_pass": True, "cpu": 4, "tune": "psnr", "aom": {"denoise-noise-level": 10}},
    # x265 (2-pass)
    "x265_slow_psnr": {"codec": "libx265", "two_pass": True, "preset": "slow", "tune": "psnr"},
    "x265_slower_psnr": {"codec": "libx265", "two_pass": True, "preset": "slower", "tune": "psnr"},
    "x265_medium_psnr": {"codec": "libx265", "two_pass": True, "preset": "medium", "tune": "psnr"},
    "x265_slow_aq3": {"codec": "libx265", "two_pass": True, "preset": "slow", "x265": {"aq-mode": 3}},
    # VP9 (2-pass)
    "vp9_c2": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 2},
    "vp9_c1": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 1},
    "vp9_c3": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 3},
    # x264 (2-pass) baseline
    "x264_veryslow_psnr": {"codec": "libx264", "two_pass": True, "preset": "veryslow", "tune": "psnr"},
    # rav1e
    # NVENC (bundled SDK features)
    "nv_av1_p7_uhq": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32},
    "nv_av1_p7_hq_full": {"codec": "av1_nvenc", "preset": "p7", "tune": "hq", "multipass": "fullres", "la": 32},
    "nv_av1_p7_hq_full_noaq": {"codec": "av1_nvenc", "preset": "p7", "tune": "hq", "multipass": "fullres", "la": 32, "aq": False},
    "nv_av1_p7_uhq_tf": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32, "tf_level": 4},
    "nv_hevc_p7_uhq": {"codec": "hevc_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32},
    "nv_hevc_p7_hq_full": {"codec": "hevc_nvenc", "preset": "p7", "tune": "hq", "multipass": "fullres", "la": 32},
}


def _head(inp: str, vf: str | None, binary: str = FFMPEG) -> list[str]:
    a = [binary, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", inp,
         "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough"]
    if vf:
        a += ["-vf", vf]
    return a


def _tail(out: str, pix_fmt: str = "yuv420p") -> list[str]:
    return ["-pix_fmt", pix_fmt, "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv", "-f", "mp4", out]


def build_commands(arm: dict, inp: str, out: str, kbps: float, gop: int, passlog: str,
                   vf: str | None = None) -> list[list[str]]:
    c = arm["codec"]
    maxrate, bufsize = kbps * 1.45, kbps * 2.0
    rate = ["-b:v", f"{kbps:.2f}k"]
    vbv = ["-maxrate", f"{maxrate:.2f}k", "-bufsize", f"{bufsize:.2f}k"]
    if c == "libsvtav1":
        svt = {"keyint": gop, "scd": 0, "lp": THREADS, **arm.get("svt", {})}
        enc = ["-c:v", c, "-preset", str(arm["preset"]), *rate, "-g", str(gop),
               "-svtav1-params", ":".join(f"{k}={v}" for k, v in svt.items())]
        binary = "ffmpeg" if arm.get("bin") == "system" else FFMPEG
        return [_head(inp, vf, binary) + enc + _tail(out, arm.get("pix_fmt", "yuv420p"))]
    if c in ("av1_nvenc", "hevc_nvenc"):
        aq = arm.get("aq", True)
        enc = ["-c:v", c, "-preset", arm["preset"], "-tune", arm["tune"], "-rc", "vbr", *rate, *vbv,
               "-g", str(gop), "-forced-idr", "1", "-strict_gop", "1", "-no-scenecut", "1",
               "-rc-lookahead", str(arm.get("la", 32)), "-multipass", arm.get("multipass", "fullres"),
               "-spatial-aq", "1" if aq else "0", "-temporal-aq", "1" if aq else "0"]
        if "tf_level" in arm:
            enc += ["-tf_level", str(arm["tf_level"])]
        return [_head(inp, vf) + enc + _tail(out)]
    if c == "librav1e":
        enc = ["-c:v", c, "-speed", str(arm["speed"]), *rate, "-g", str(gop), "-keyint_min", str(gop),
               "-rav1e-params", f"threads={THREADS}:no_scene_detection=true"]
        return [_head(inp, vf) + enc + _tail(out)]

    # 2-pass software encoders
    common: list[str]
    if c == "libaom-av1":
        aomp = {"enable-keyframe-filtering": 0, **arm.get("aom", {})}
        common = ["-c:v", c, "-cpu-used", str(arm["cpu"]), "-row-mt", "1", "-threads", str(THREADS),
                  "-tiles", "2x1", "-lag-in-frames", "35", "-auto-alt-ref", "1", "-tune", arm["tune"],
                  "-g", str(gop), "-keyint_min", str(gop), "-aom-params",
                  ":".join(f"{k}={v}" for k, v in aomp.items()), *rate, *vbv, "-undershoot-pct", "5",
                  "-overshoot-pct", "5"]
    elif c == "libx265":
        xp = {"keyint": gop, "min-keyint": gop, "scenecut": 0, "open-gop": 0,
              "vbv-maxrate": int(maxrate), "vbv-bufsize": int(bufsize), "pools": THREADS,
              "stats": passlog + ".x265", **arm.get("x265", {})}
        common = ["-c:v", c, "-preset", arm["preset"], *rate]
        if arm.get("tune"):
            common += ["-tune", arm["tune"]]
        common += ["-x265-params", ":".join(f"{k}={v}" for k, v in xp.items())]
    elif c == "libvpx-vp9":
        common = ["-c:v", c, "-deadline", "good", "-cpu-used", str(arm["cpu"]), "-row-mt", "1",
                  "-threads", str(THREADS), "-tile-columns", "2", "-lag-in-frames", "25",
                  "-auto-alt-ref", "1", "-arnr-maxframes", "7", "-arnr-strength", "4",
                  "-g", str(gop), "-keyint_min", str(gop), *rate, *vbv]
    elif c == "libx264":
        common = ["-c:v", c, "-preset", arm["preset"], "-tune", arm.get("tune", "psnr"), *rate, *vbv,
                  "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0", "-threads", str(THREADS)]
    else:
        raise ValueError(c)

    if c == "libx265":
        p1 = _head(inp, vf) + common[:-1] + [common[-1] + ":pass=1", "-f", "null", "-"]
        p2 = _head(inp, vf) + common[:-1] + [common[-1] + ":pass=2"] + _tail(out)
        return [p1, p2]
    p1 = _head(inp, vf) + common + ["-pass", "1", "-passlogfile", passlog, "-f", "null", "-"]
    p2 = _head(inp, vf) + common + ["-pass", "2", "-passlogfile", passlog] + _tail(out)
    return [p1, p2]


# ---------------------------------------------------------------- trial
_csv_lock = threading.Lock()
FIELDS = ["clip_id", "arm", "codec", "prefilter", "aim", "attempts", "passed_streaming", "codes",
          "avg_kbps", "peak_kbps", "util", "vmaf", "vmaf_neg", "vmaf_delta", "clip_score",
          "encode_seconds", "spf", "error"]


def run_trial(cid: str, arm_name: str, arm: dict, work: Path, vf: str | None, pf_name: str) -> dict:
    inp = str(clip_path(cid))
    meta = probe_video(inp, count_frames=True)
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
    gop = max(1, int(2.0 * meta.fps + 1e-9))
    d = Path(tempfile.mkdtemp(prefix=f"{cid}-{arm_name}-", dir=work))
    row = {"clip_id": cid, "arm": arm_name, "codec": arm["codec"], "prefilter": pf_name,
           "passed_streaming": False, "encode_seconds": 0.0}
    try:
        # Closed-loop rate fit: encoders under/overshoot the requested bitrate by up to ~25%
        # on hard content, so rescale the request from the measured utilisation.
        aim, best, best_util = AIMS[0], None, -1.0
        for i in range(1, MAX_FIT_ATTEMPTS + 1):
            out = str(d / f"out{i}.mp4")
            t0 = time.perf_counter()
            err = ""
            for cmd in build_commands(arm, inp, out, target * aim, gop, str(d / "pass"), vf):
                r = run_command(cmd, timeout=7200)
                if not r.ok:
                    err = (r.stderr or r.stdout or "")[-300:]
                    break
            row["encode_seconds"] += time.perf_counter() - t0
            row["attempts"] = i
            if err or not os.path.exists(out):
                row["error"] = err.replace("\n", " ")
                break
            m = measure_streaming(read_packets_ffprobe(out, ffprobe=FFPROBE), fps=meta.fps,
                                  frames=meta.nb_frames, peak_window_seconds=2.0)
            vs = streaming_violations(m, target_kbps=target, peak_ratio=1.5, peak_window_seconds=2.0,
                                      max_keyframe_interval_seconds=2.0, fps=meta.fps)
            codes = {v.code for v in vs}
            util = m.avg_kbps / target
            if not vs and util > best_util:
                best, best_util = (out, aim, m), util
            if "STREAMING_KEYFRAME_INTERVAL" in codes:
                row["codes"] = "|".join(sorted(codes))
                break
            if not vs and util >= FIT_LOW:
                break
            if "STREAMING_PEAK_EXCEEDED" in codes and "STREAMING_BITRATE_EXCEEDED" not in codes:
                aim *= 0.93
            else:
                aim *= FIT_GOAL / max(util, 1e-3)
            aim = min(aim, 2.5)
        if best is not None:
            out, aim, m = best
            row.update({"aim": round(aim, 4), "avg_kbps": round(m.avg_kbps, 2), "peak_kbps": round(m.peak_kbps, 2),
                        "util": round(m.avg_kbps / target, 4), "passed_streaming": True, "codes": "", "error": ""})
            pair = compute_vmaf_pair(inp, out, ffmpeg=FFMPEG)
            row.update({"vmaf": round(pair.vmaf, 4), "vmaf_neg": round(pair.vmaf_neg, 4),
                        "vmaf_delta": round(pair.delta, 4), "clip_score": round(clip_score(pair.vmaf), 4)})
        row["spf"] = round(row["encode_seconds"] / max(1, meta.nb_frames), 4)
        return row
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)[:300]
        return row
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--prefilters", default="none")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "hard_explore" / "trials.csv"))
    args = ap.parse_args()

    from streamenc.filters.policy import FILTER_CATALOG
    extra_pf = {
        "blur05": "gblur=sigma=0.5",
        "ds75": "scale=iw*3/4:-2:flags=bicubic,scale=iw*4/3:-2:flags=lanczos",
        "hqdn_mild": "hqdn3d=1.5:1.5:3:3",
        "cas03": "cas=strength=0.3",
        "ds85": "scale=iw*17/20:-2:flags=bicubic,scale=iw*20/17:-2:flags=lanczos",
    }
    clips = args.clips.split(",")
    arms = list(ARMS) if args.arms == "all" else args.arms.split(",")
    pfs = args.prefilters.split(",")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work" / "hard_explore"
    work.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with open(out) as f:
            for r in csv.DictReader(f):
                done.add((r["clip_id"], r["arm"], r["prefilter"]))
    jobs = [(c, a, p) for p in pfs for c in clips for a in arms if (c, a, p) not in done]
    print(f"{len(jobs)} trials ({len(done)} done)", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {}
            for c, a, p in jobs:
                vf = None if p == "none" else (extra_pf.get(p) or FILTER_CATALOG.get(p))
                futs[ex.submit(run_trial, c, a, ARMS[a], work, vf, p)] = (c, a, p)
            for fu in as_completed(futs):
                r = fu.result()
                with _csv_lock:
                    w.writerow(r)
                    f.flush()
                print(f"{r['clip_id']:<18} {r['arm']:<24} {p if (p:=r['prefilter'])!='none' else '':<9} "
                      f"vmaf={r.get('vmaf')} util={r.get('util')} t={r.get('encode_seconds', 0):.0f}s "
                      f"{r.get('codes') or ''} {r.get('error', '')[:80]}", flush=True)


if __name__ == "__main__":
    main()
