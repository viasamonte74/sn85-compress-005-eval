#!/usr/bin/env python3
"""Wide encoder-config search for hard clips (2026-11-10, user: "increase configs; once one config scores, use it
as the reference").  Two stages per clip, every trial a normal label trial (rate fit at 0.05 bpp + full validation):
  A. one-factor screen: base svt_crf_p4_qm_la plus ONE change (SVT tool / preset / prefilter / bit depth), ~55 configs;
  B. reference search: the best config of A becomes the reference; the other tools that gained are added one at a
     time (largest gain first) and kept when the combination beats the current reference (forward selection).
Metric: VMAF of a gate-valid encode; for clips with no valid encode, the lowest bitrate/budget ratio at CRF 63.
Output: results/hard_rescue/search_trials.jsonl (+ search_best.jsonl, one line per clip). Resumable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamenc.encoders.arms import ARMS  # noqa: E402
from streamenc.ml.encoder_rank.labeling import run_label_trial, source_cache  # noqa: E402

BASE = {"preset": 4, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}}
SR = lambda d: {"superres-mode": 1, "superres-denom": d, "superres-kf-denom": d}  # noqa: E731
# tool name -> (group, change). Tools of one group are alternatives (never combined).
TOOLS: dict[str, tuple[str, dict]] = {
    "p2": ("preset", {"preset": 2}), "p1": ("preset", {"preset": 1}), "p0": ("preset", {"preset": 0}),
    "p3": ("preset", {"preset": 3}),
    "tune0": ("tune", {"svt": {"tune": 0}}), "tune2": ("tune", {"svt": {"tune": 2}}),
    "sharpm2": ("sharp", {"svt": {"sharpness": -2}}), "sharpm5": ("sharp", {"svt": {"sharpness": -5}}),
    "sharp2": ("sharp", {"svt": {"sharpness": 2}}), "sharp4": ("sharp", {"svt": {"sharpness": 4}}),
    "lqb5": ("lqb", {"svt": {"luminance-qp-bias": 5}}), "lqb10": ("lqb", {"svt": {"luminance-qp-bias": 10}}),
    "lqb20": ("lqb", {"svt": {"luminance-qp-bias": 20}}),
    "vb1": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 1}}),
    "vb2": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 2}}),
    "vb3": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 3}}),
    "vb4": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 4}}),
    "vb2o4": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 2, "variance-octile": 4}}),
    "tfoff": ("tf", {"svt": {"enable-tf": 0}}), "tfadapt": ("tf", {"svt": {"enable-tf": 2}}),
    "tf0": ("tf", {"svt": {"tf-strength": 0}}), "tf2": ("tf", {"svt": {"tf-strength": 2}}),
    "tf3": ("tf", {"svt": {"tf-strength": 3}}), "tf4": ("tf", {"svt": {"tf-strength": 4}}),
    "cdef0": ("cdef", {"svt": {"enable-cdef": 0}}), "rest0": ("rest", {"svt": {"enable-restoration": 0}}),
    "dlf2": ("dlf", {"svt": {"enable-dlf": 2}}), "overlay": ("overlay", {"svt": {"enable-overlays": 1}}),
    "hl3": ("hl", {"svt": {"hierarchical-levels": 3}}), "hl4": ("hl", {"svt": {"hierarchical-levels": 4}}),
    "hl5": ("hl", {"svt": {"hierarchical-levels": 5}}),
    "aq0": ("aq", {"svt": {"aq-mode": 0}}), "aq1": ("aq", {"svt": {"aq-mode": 1}}),
    "fg6": ("fg", {"svt": {"film-grain": 6, "film-grain-denoise": 1}}),
    "fg12": ("fg", {"svt": {"film-grain": 12, "film-grain-denoise": 1}}),
    "fg20": ("fg", {"svt": {"film-grain": 20, "film-grain-denoise": 1}}),
    "sr10": ("sr", {"svt": SR(10)}), "sr12": ("sr", {"svt": SR(12)}), "sr14": ("sr", {"svt": SR(14)}),
    "sr16": ("sr", {"svt": SR(16)}),
    "qpc1": ("qpc", {"svt": {"qp-scale-compress-strength": 1}}), "qpc2": ("qpc", {"svt": {"qp-scale-compress-strength": 2}}),
    "qpc3": ("qpc", {"svt": {"qp-scale-compress-strength": 3}}),
    "mfmv": ("mfmv", {"svt": {"enable-mfmv": 1}}), "smg2": ("smg", {"svt": {"startup-mg-size": 2}}),
    "mtx32": ("mtx", {"svt": {"max-tx-size": 32}}),
    "cqm0": ("cqm", {"svt": {"chroma-qm-min": 0}}), "qmoff": ("qm", {"svt": {"enable-qm": 0}}),
    "qmmin4": ("qm", {"svt": {"qm-min": 4}}), "acb1": ("acb", {"svt": {"ac-bias": 1.0}}),
    "acb3": ("acb", {"svt": {"ac-bias": 3.0}}),
    "iref1": ("iref", {"svt": {"irefresh-type": 1}}),
    "b10": ("depth", {"pix_fmt": "yuv420p10le"}),
    "f_hqlight": ("pre", {"prefilter": "hqdn3d_light"}), "f_hqspatial": ("pre", {"prefilter": "hqdn3d_spatial"}),
    "f_ds85": ("pre", {"prefilter": "ds85"}), "f_ds67": ("pre", {"prefilter": "ds67"}),
    "f_tmix3": ("pre", {"prefilter": "tmix3"}), "f_hold2": ("pre", {"prefilter": "hold2"}),
}
# ---- wide search 2 (2026-11-11, user: "extend encoder configs more, use NVENC too"); enabled with --wide2.
# VMAF is luma-only, so chroma quantiser offsets move bits to luma (the perceptual chroma gates still apply).
TOOLS2: dict[str, tuple[str, dict]] = {
    "cro8": ("cro", {"svt": {"chroma-u-dc-qindex-offset": 8, "chroma-u-ac-qindex-offset": 8, "chroma-v-dc-qindex-offset": 8, "chroma-v-ac-qindex-offset": 8}}), "cro16": ("cro", {"svt": {"chroma-u-dc-qindex-offset": 16, "chroma-u-ac-qindex-offset": 16, "chroma-v-dc-qindex-offset": 16, "chroma-v-ac-qindex-offset": 16}}), "cro32": ("cro", {"svt": {"chroma-u-dc-qindex-offset": 32, "chroma-u-ac-qindex-offset": 32, "chroma-v-dc-qindex-offset": 32, "chroma-v-ac-qindex-offset": 32}}), "cro48": ("cro", {"svt": {"chroma-u-dc-qindex-offset": 48, "chroma-u-ac-qindex-offset": 48, "chroma-v-dc-qindex-offset": 48, "chroma-v-ac-qindex-offset": 48}}),
    "crom8": ("cro", {"svt": {"chroma-u-dc-qindex-offset": -8, "chroma-u-ac-qindex-offset": -8, "chroma-v-dc-qindex-offset": -8, "chroma-v-ac-qindex-offset": -8}}),
    "ydcm4": ("ydc", {"svt": {"luma-y-dc-qindex-offset": -4}}), "ydc4": ("ydc", {"svt": {"luma-y-dc-qindex-offset": 4}}),
    "kfq8": ("kfq", {"svt": {"key-frame-qindex-offset": -8}}), "kfq16": ("kfq", {"svt": {"key-frame-qindex-offset": -16}}),
    "kfcq16": ("kfcq", {"svt": {"key-frame-chroma-qindex-offset": 16}}),
    "fixq": ("fixq", {"svt": {"use-fixed-qindex-offsets": 1}}),
    "sqo8": ("sqo", {"svt": {"startup-qp-offset": -8}}), "sqo16": ("sqo", {"svt": {"startup-qp-offset": -16}}),
    "qmmax8": ("qmmax", {"svt": {"qm-max": 8}}), "qmmax12": ("qmmax", {"svt": {"qm-max": 12}}),
    "cqmmax8": ("cqmmax", {"svt": {"chroma-qm-max": 8}}),
    "hl2": ("hl", {"svt": {"hierarchical-levels": 2}}),
    "lowdelay": ("pred", {"svt": {"pred-struct": 1}}),
    "scm0": ("scm", {"svt": {"scm": 0}}), "scm1": ("scm", {"svt": {"scm": 1}}),
    "dg0": ("dg", {"svt": {"enable-dg": 0}}), "fastdec": ("fastdec", {"svt": {"fast-decode": 1}}),
    "tiles1": ("tiles", {"svt": {"tile-columns": 1}}), "recode4": ("recode", {"svt": {"recode-loop": 4}}),
    "tune4": ("tune", {"svt": {"tune": 4}}),
    "vb2c1": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 2, "variance-boost-curve": 1}}),
    "vb2o2": ("vb", {"svt": {"enable-variance-boost": 1, "variance-boost-strength": 2, "variance-octile": 2}}),
    "afg12": ("fg", {"svt": {"film-grain": 12, "film-grain-denoise": 1, "adaptive-film-grain": 1}}),
    "la60": ("la", {"svt": {"lookahead": 60}}), "la32": ("la", {"svt": {"lookahead": 32}}),
    "scd1": ("scd", {"svt": {"scd": 1}}), "acb6": ("acb", {"svt": {"ac-bias": 6.0}}),
    "tf1": ("tf", {"svt": {"tf-strength": 1}}), "maxqp58": ("maxqp", {"svt": {"max-qp": 58}}),
    "sharpm7": ("sharp", {"svt": {"sharpness": -7}}), "sharp7": ("sharp", {"svt": {"sharpness": 7}}),
    "f_ds75": ("pre", {"prefilter": "ds75"}), "f_ds60": ("pre", {"prefilter": "ds60"}),
    "f_ds50": ("pre", {"prefilter": "ds50"}), "f_ds67l": ("pre", {"prefilter": "ds67l"}),
    "f_ds67a": ("pre", {"prefilter": "ds67a"}), "f_ds67usm": ("pre", {"prefilter": "ds67_usm"}),
    "f_hqds67": ("pre", {"prefilter": "hq_ds67"}), "f_dsw67": ("pre", {"prefilter": "dsw67"}),
    "f_dsh67": ("pre", {"prefilter": "dsh67"}), "f_nlm": ("pre", {"prefilter": "nlm"}),
    "f_cblur": ("pre", {"prefilter": "cblur"}), "f_gblur05": ("pre", {"prefilter": "gblur05"}),
    "f_gblur08": ("pre", {"prefilter": "gblur08"}), "f_hqhard": ("pre", {"prefilter": "hqdn3d_hard"}),
    "f_atad": ("pre", {"prefilter": "atadenoise_hard"}), "f_harddn": ("pre", {"prefilter": "hard_denoise"}),
}
# other encoders (stage A only): NVENC on the GPU (AV1 / HEVC), libaom, x265, VP9, x264, SVT-AV1 VBR
OTHER2 = ["nv_av1_p7_uhq", "nv_av1_p7_hq_full", "nv_av1_p7_hq_full_noaq", "nv_av1_p7_uhq_tf", "nv_av1_p7_uhq_10b",
          "nv_av1_p7_uhq_10b_tf", "nv_av1_p7_uhq_la3", "nv_av1_p7_uhq_la60", "nv_av1_p7_uhq_aq15", "nv_av1_p7_uhq_noaq",
          "nv_av1_p7_uhq+ds67", "nv_av1_p7_uhq+ds50", "nv_av1_p7_uhq+hqdn3d_light", "nv_av1_p7_uhq_10b+ds67",
          "nv_hevc_p7_uhq", "nv_hevc_p7_hq_full", "nv_hevc_p7_uhq_tf", "nv_hevc_p7_uhq+ds67",
          "aom_c2_psnr", "aom_c3_ssim", "aom_c4_ssim", "aom_c4_psnr_dn", "aom_c3_psnr+ds67",
          "x265_slower_psnr", "x265_veryslow_psnr", "x265_slow_aq3", "x265_slower_psnr+ds67",
          "vp9_c1", "vp9_c0", "x264_veryslow_psnr", "svt_p2_tuned", "svt_p4_tuned"]
OTHER = ["aom_c3_psnr", "svt_crf_p8_t1"]  # other-encoder / speed references (stage A only)
OUT = ROOT / "results/hard_rescue/search_trials.jsonl"
BEST = ROOT / "results/hard_rescue/search_best.jsonl"


def label_id(tools: tuple[str, ...]) -> str:
    """Register the arm for a tool set and return its label id."""
    arm = {"codec": "libsvtav1", "rc": "crf", "preset": BASE["preset"], "svt": dict(BASE["svt"])}
    pre = None
    for t in tools:
        ch = TOOLS[t][1]
        arm["svt"].update(ch.get("svt", {}))
        if "preset" in ch: arm["preset"] = ch["preset"]
        if "pix_fmt" in ch: arm["pix_fmt"] = ch["pix_fmt"]
        pre = ch.get("prefilter", pre)
    name = "x_" + ("-".join(t for t in tools if not t.startswith("f_")) or "base")
    ARMS[name] = arm
    return name + (f"+{pre}" if pre else "")


def metric(t: dict) -> float | None:
    if t.get("passed") and t.get("vmaf") is not None:
        return float(t["vmaf"])
    return None


def over(t: dict) -> float | None:
    """bitrate / budget of the last fit attempt (for clips with no valid encode)."""
    try:
        return float(str(t.get("fit_history") or "").split()[-1].split(":")[1])
    except (IndexError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-add", type=int, default=6)
    ap.add_argument("--wide2", action="store_true", help="add the TOOLS2 / OTHER2 configs (incl. NVENC)")
    a = ap.parse_args()
    if a.wide2:
        TOOLS.update(TOOLS2)
        OTHER.extend(OTHER2)
    labels, pilot = {}, {}
    dirs = [ROOT / "data/encoder_rank_all"] + sorted((ROOT / "data").glob("encoder_rank_v14*/shard*"))
    for d in dirs:
        for l in (open(d / "labels.jsonl") if (d / "labels.jsonl").exists() else []):
            r = json.loads(l); labels[r["clip_id"]] = r
        for l in (open(d / "trials.jsonl") if (d / "trials.jsonl").exists() else []):
            t = json.loads(l)
            if t["config_id"] == "svt_crf_p5_qm_la" and t.get("crf_estimate") is not None:
                pilot[t["clip_id"]] = float(t["crf_estimate"])
    done: dict[tuple[str, str], dict] = {}
    if OUT.exists():
        for l in open(OUT):
            t = json.loads(l); done[(t["clip_id"], t["config_id"])] = t
    work = ROOT / "work/hard_search"

    for cid in a.clips.split(","):
        path = Path(labels[cid]["path"])
        raw, raw_dir = source_cache(path)
        hint = pilot.get(cid)

        def run(lids: list[str]) -> list[dict]:
            todo = [l for l in lids if (cid, l) not in done]
            with ThreadPoolExecutor(a.workers) as ex:
                for t in ex.map(lambda l: run_label_trial(path, l, work, crf_hint=hint if l.startswith(("x_", "svt")) else None,
                                                          encode_input=raw), todo):
                    done[(cid, t["config_id"])] = t
                    with open(OUT, "a") as f:
                        f.write(json.dumps(t) + "\n")
                    print(f"  {t['config_id']:40} pass={t['passed']} vmaf={t['vmaf']} neg={t.get('vmaf_neg')} over={over(t)} "
                          f"cpu={t.get('enc_cpu_seconds')} {t['codes'][:50]}", flush=True)
            return [done[(cid, l)] for l in lids]

        try:
            print(f"== {cid}: stage A ({len(TOOLS) + 1 + len(OTHER)} configs), CRF hint {hint}", flush=True)
            names = list(TOOLS)
            res = run([label_id(())] + [label_id((n,)) for n in names] + OTHER)
            base, singles = res[0], dict(zip(names, res[1:1 + len(names)]))
            valid = metric(base) is not None or any(metric(t) is not None for t in singles.values())
            if valid:   # quality mode: VMAF of gate-valid encodes
                score = lambda t: metric(t) if metric(t) is not None else -1e9  # noqa: E731
                eps = 0.05
            else:       # fit mode: bitrate / budget at the highest CRF, lower is better
                score = lambda t: -(over(t) or 1e9)  # noqa: E731
                eps = 0.01
            b0 = score(base)
            gains = sorted(((score(t) - b0, n) for n, t in singles.items() if score(t) - b0 > eps), reverse=True)
            print(f"  stage A: base {b0:.3f} ({'VMAF' if valid else '-over'}); gains: "
                  + ", ".join(f"{n} {g:+.2f}" for g, n in gains[:15]), flush=True)
            cur, cur_s, used = (), b0, set()
            for g, n in gains:   # stage B: forward selection from the best single config
                if len(cur) >= a.max_add or TOOLS[n][0] in used:
                    continue
                cand = cur + (n,)
                t = singles[n] if not cur else run([label_id(cand)])[0]
                s = score(t)
                keep = s > cur_s + eps
                print(f"  stage B: {'+'.join(cand)} -> {s:.3f} ({'keep' if keep else 'drop'})", flush=True)
                if keep:
                    cur, cur_s = cand, s
                    used.add(TOOLS[n][0])
            best = done[(cid, label_id(cur))]
            rec = {"clip_id": cid, "mode": "vmaf" if valid else "fit", "base": b0, "best": cur_s, "tools": list(cur),
                   "config_id": label_id(cur), "arm": ARMS[label_id(cur).split("+")[0]], "passed": best.get("passed"),
                   "vmaf": best.get("vmaf"), "clip_score": best.get("clip_score"), "over": over(best),
                   "label_max_vmaf": labels[cid].get("max_vmaf")}
            with open(BEST, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  BEST {cid}: {'+'.join(cur) or 'base'} {cur_s:.3f} (base {b0:.3f}, label best {labels[cid].get('max_vmaf')})", flush=True)
        finally:
            if raw_dir:
                shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
