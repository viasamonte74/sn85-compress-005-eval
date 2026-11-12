#!/usr/bin/env python3
"""q08 lab (2026-11-12, user: raise q08 to VMAF >= 75; try untried methods on q08 only).

Each config: whole-clip encode, CRF searched (quarter steps; libaom: 2-pass VBR bitrate) until the average lands in
[0.990, 1.007] x budget with the 2 s peak <= 1.5 x budget; then validate_clip (validator-equivalent: VMAF, VMAF-NEG,
streaming, keyframes, chroma / tone / grayscale gates). Appends results/q08_lab/results.jsonl.
  python scripts/q08_lab.py base 10bit chroma24 ...
"""
from __future__ import annotations

import json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import FFMPEG, FFPROBE, build_commands  # noqa: E402
from streamenc.filters.policy import filter_vf  # noqa: E402
from streamenc.probe import probe_video  # noqa: E402
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402
from streamenc.validate.official_streaming import read_packets_ffprobe, streaming_target_kbps  # noqa: E402
from streamenc.validate.validate_clip import validate_clip  # noqa: E402

CLIP = os.environ.get("Q08_CLIP", "/workspace/dataset-src/practice_pack_005/q08/clip.mkv")
OUT = ROOT / "results" / "q08_lab"
LQB = {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120, "luminance-qp-bias": 5}
_DS = lambda f: ("boxblur=luma_radius=0:luma_power=0:chroma_radius=4:chroma_power=1,"  # noqa: E731
                 f"scale=trunc(iw*{f}/2)*2:trunc(ih*{f}/2)*2:flags=bicubic,scale={{w}}:{{h}}:flags=lanczos")
CH = lambda o: {f"chroma-{p}-{b}-qindex-offset": o for p in "uv" for b in ("dc", "ac")}  # noqa: E731
CONFIGS = {
    # name: (codec, preset/cpu, svt params, pix_fmt, prefilter, extra)
    "base":        ("svt", 3, LQB, None, "cblur4_ds85", {}),
    "base_nopf":   ("svt", 3, LQB, None, None, {}),
    "10bit":       ("svt", 3, LQB, "yuv420p10le", "cblur4_ds85", {}),
    "chroma24":    ("svt", 3, {**LQB, **CH(24)}, None, "cblur4_ds85", {}),
    "chroma48":    ("svt", 3, {**LQB, **CH(48)}, None, "cblur4_ds85", {}),
    "chroma32_nopf": ("svt", 3, {**LQB, **CH(32)}, None, "ds85", {}),
    "notf":        ("svt", 3, {**LQB, "enable-tf": 0}, None, "cblur4_ds85", {}),
    "hier5":       ("svt", 3, {**LQB, "hierarchical-levels": 5}, None, "cblur4_ds85", {}),
    "fgs8":        ("svt", 3, {**LQB, "film-grain": 8}, None, "cblur4_ds85", {}),
    "p1_10bit":    ("svt", 1, LQB, "yuv420p10le", "cblur4_ds85", {}),
    "p0_10bit_ch": ("svt", 0, {**LQB, **CH(24)}, "yuv420p10le", "cblur4_ds85", {}),
    "aom3":        ("aom", 3, {}, None, "cblur4_ds85", {}),
    # lower effective resolution (down/up before encoding; chroma box blur kept like cblur4_ds85)
    "ds70":        ("svt", 3, LQB, None, "raw:" + _DS(0.70), {}),
    "ds60":        ("svt", 3, LQB, None, "raw:" + _DS(0.60), {}),
    "ds50":        ("svt", 3, LQB, None, "raw:" + _DS(0.50), {}),
    "cqm15":       ("svt", 3, {**LQB, "chroma-qm-min": 15, "chroma-qm-max": 15}, None, "cblur4_ds85", {}),
    "hier3":       ("svt", 3, {**LQB, "hierarchical-levels": 3}, None, "cblur4_ds85", {}),
    "p1_10b_h3":       ("svt", 1, {**LQB, "hierarchical-levels": 3}, "yuv420p10le", "cblur4_ds85", {}),
    "p1_10b_h3_cb16":  ("svt", 1, {**LQB, "hierarchical-levels": 3}, "yuv420p10le", "cblur16_ds85", {}),
    "p3_h3_cb16":      ("svt", 3, {**LQB, "hierarchical-levels": 3}, None, "cblur16_ds85", {}),
    "p0_10b_h3_cb16":  ("svt", 0, {**LQB, "hierarchical-levels": 3}, "yuv420p10le", "cblur16_ds85", {}),
    "p2":              ("svt", 2, LQB, None, "cblur4_ds85", {}),
    "p2_h3_cb16":      ("svt", 2, {**LQB, "hierarchical-levels": 3}, None, "cblur16_ds85", {}),
    "p2_10b_h3_cb16":  ("svt", 2, {**LQB, "hierarchical-levels": 3}, "yuv420p10le", "cblur16_ds85", {}),
    "sres12":      ("svt", 3, {**LQB, "superres-mode": 1, "superres-denom": 12, "superres-kf-denom": 12}, None, "cblur4_ds85", {}),
    "sres16":      ("svt", 3, {**LQB, "superres-mode": 1, "superres-denom": 16, "superres-kf-denom": 16}, None, None, {}),
    "aom3_10bit":  ("aom", 3, {}, "yuv420p10le", "cblur4_ds85", {}),
    "aom1_10bit":  ("aom", 1, {}, "yuv420p10le", "cblur4_ds85", {}),
}


def measure(path, meta, target):
    sz = [p.size_bytes for p in read_packets_ffprobe(path, ffprobe=FFPROBE)]
    n, fps = len(sz), meta.fps
    avg = 8.0 * sum(sz) / (meta.nb_frames / fps) / 1000.0
    win = max(1, round(2.0 * fps)); run_, best = sum(sz[:win]), sum(sz[:win])
    for k in range(win, n):
        run_ += sz[k] - sz[k - win]; best = max(best, run_)
    peak = 8.0 * best / (win / fps) / 1000.0 if n > win else avg
    return avg / target, peak / target


def encode(name, meta, gop, target, x, work):
    codec, pre, svt, pix, pf, extra = CONFIGS[name]
    if pf and pf.startswith("raw:"):  # lab-only filter strings: down/up scale by a factor
        vf = pf[4:].format(w=meta.width, h=meta.height)
    else:
        vf = filter_vf(pf).format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction)) if pf else None
    o = str(work / f"{name}_{x:.3f}.mp4")
    t0 = time.perf_counter(); cpu = 0.0
    if codec == "svt":
        arm = {"codec": "libsvtav1", "rc": "crf", "preset": pre, "svt": dict(svt), "pix_fmt": pix or "yuv420p"}
        cmds = build_commands(arm, CLIP, o, target, gop, str(work / f"{name}.pass"), vf, crf=x)
        for cmd in cmds:
            r = run_command(cmd, timeout=7200); cpu += r.cpu_seconds
            assert r.ok, r.stderr[-400:]
    else:  # libaom 2-pass VBR at x * budget
        kb = int(round(x * target))
        base = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", CLIP, "-map", "0:v:0"]
        if vf:
            base += ["-vf", vf]
        enc = ["-c:v", "libaom-av1", "-cpu-used", str(pre), "-row-mt", "1", "-tiles", "2x1", "-threads", "16",
               "-b:v", f"{kb}k", "-maxrate", f"{int(kb * 1.35)}k", "-bufsize", f"{int(kb * 2)}k",
               "-g", str(gop), "-keyint_min", str(gop), "-lag-in-frames", "35", "-aq-mode", "0",
               "-undershoot-pct", "2", "-overshoot-pct", "2", "-pix_fmt", pix or "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
        pl = str(work / f"{name}_aom")
        for ps, dst in ((1, "/dev/null"), (2, o)):
            cmd = base + enc + ["-pass", str(ps), "-passlogfile", pl] + (["-f", "null"] if ps == 1 else ["-f", "mp4"]) + [dst]
            r = run_command(cmd, timeout=14400); cpu += r.cpu_seconds
            assert r.ok, r.stderr[-400:]
    return o, time.perf_counter() - t0, cpu


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work" / "q08_lab"; work.mkdir(parents=True, exist_ok=True)
    meta = probe_video(CLIP, count_frames=True)
    gop = gop_for_timestamps(packet_pts(CLIP), meta.fps, 2.0)
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=0.05)
    for name in sys.argv[1:]:
        codec = CONFIGS[name][0]
        tries, best = [], None
        x = 56.0 if codec == "svt" else 0.985
        for _ in range(8):
            o, wall, cpu = encode(name, meta, gop, target, x, work)
            u, pk = measure(o, meta, target)
            tries.append((round(x, 3), round(u, 4), round(pk, 3), round(wall, 1)))
            print(f"{name}: x={x:.3f} use {u:.4f} peak {pk:.3f} ({wall:.1f} s)", flush=True)
            ok = u <= 1.007 and pk <= 1.5
            if ok and (best is None or u > best[1]):
                if best: os.remove(best[0])
                best = (o, u, pk, x, wall, cpu)
            elif o != (best or [None])[0]:
                os.remove(o)
            if ok and u >= 0.990:
                break
            if codec == "svt":  # ln(bits) ~ -0.1 per CRF unit near here; quarter steps
                k = 0.095
                nx = x + math.log(max(u, 1e-6) / 0.9985) / k
                nx = round(nx * 4) / 4
                if nx == x:
                    nx = x + (0.25 if u > 1.007 else -0.25)
                if any(abs(t[0] - nx) < 1e-6 for t in tries):
                    break
                x = nx
            else:
                x = x * 0.9985 / max(u, 1e-6)
        if best is None:
            print(f"{name}: no encode inside the budget", flush=True); continue
        rep = validate_clip(CLIP, best[0], run_vmaf=True, run_perceptual=True)
        m = rep.metrics
        rec = {"config": name, "x": best[3], "use": round(best[1], 4), "peak": round(best[2], 3), "vmaf": m.get("vmaf"),
               "vmaf_neg": m.get("vmaf_neg"), "gap": (m.get("vmaf") or 0) - (m.get("vmaf_neg") or 0),
               "passed": bool(rep.passed), "violations": [v.code for v in rep.violations], "clip_score": m.get("clip_score"),
               "encode_s": round(best[4], 1), "cpu_s": round(best[5], 1), "tries": tries,
               "chroma": {k: v for k, v in m.items() if "chroma" in k or "tone" in k or "energy" in k}}
        print(f"RESULT {name}: VMAF {rec['vmaf']:.3f} NEG {rec['vmaf_neg']:.3f} gap {rec['gap']:.2f} use {rec['use']} "
              f"passed {rec['passed']} {rec['violations']} encode {rec['encode_s']} s", flush=True)
        with open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        os.remove(best[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
