#!/usr/bin/env python3
"""Encoder-config search for clean content: renders and screen clips (2026-11-13, user: "test normal render, screen
clips to improve more, to make them get close to 100 VMAF").

Same machinery as scripts/hard_config_search.py (every trial = a normal label trial: rate fit at 0.05 bpp + full
validator-equivalent validation), but
  * tools chosen for clean, high-VMAF content: presets p1-p3, tune, qm, variance boost, temporal filter, loop
    filters, hierarchy, lookahead, keyframe / start-up QP offsets, screen-content mode, 10-bit, chroma blur
    (VMAF is luma-only). Left out: superres, film grain, down-scaling, denoise / frame-hold prefilters (they lose
    detail, which costs VMAF at 95+), SVT sharpness (sharpening is out), chroma qindex offsets (ignored by SVT 4.3);
  * the metric is VMAF normalised to 0.99 of the budget with the measured slope 0.55 x (100 - VMAF) per ln(bits),
    so a config is not credited for landing a quarter CRF step closer to the cap.
Base = svt_crf_p4_qm_la (tune 1, qm, qm-min 0, lookahead 120). Stage A one-factor, stage B forward selection.
Output: results/clean_search/trials.jsonl, best.jsonl. Resumable.
  python scripts/clean_config_search.py --clips rnd_hard_06,film_bbb_30_01 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import hard_config_search as hcs  # noqa: E402
from streamenc.encoders.arms import ARMS  # noqa: E402
from streamenc.ml.encoder_rank.labeling import run_label_trial, source_cache  # noqa: E402

hcs.TOOLS.update(hcs.TOOLS2)
# chroma QP: SVT 4.3 ignores the chroma-u/v-*-qindex-offset options; use-fixed-qindex-offsets=2 (chroma only) applies
# key-frame-chroma-qindex-offset + chroma-qindex-offsets (checked 2026-11-13: 0.5-11 % fewer bits, luma PSNR unchanged).
CHQ = lambda o: {"use-fixed-qindex-offsets": 2, "key-frame-chroma-qindex-offset": o,  # noqa: E731
                 "chroma-qindex-offsets": ",".join([str(o)] * 6)}
hcs.TOOLS.update({"chq32": ("chq", {"svt": CHQ(32)}), "chq63": ("chq", {"svt": CHQ(63)})})
USE = ["p1", "p2", "p3", "tune0", "tune2", "tune4", "lqb5", "lqb10", "vb1", "vb2", "vb3", "vb2o4", "vb2c1", "vb2o2",
       "tfoff", "tfadapt", "tf0", "tf1", "tf2", "tf3", "cdef0", "rest0", "dlf2", "overlay", "hl3", "hl4", "hl5",
       "aq0", "aq1", "qpc1", "qpc2", "mfmv", "smg2", "mtx32", "cqm0", "qmoff", "qmmin4", "qmmax8", "qmmax12",
       "cqmmax8", "acb1", "acb3", "iref1", "b10", "kfq8", "kfq16", "kfcq16", "fixq", "sqo8", "scm0", "scm1", "dg0",
       "recode4", "la60", "scd1", "maxqp58", "f_cblur", "chq32", "chq63"]
OUT = ROOT / "results/clean_search/trials.jsonl"
BEST = ROOT / "results/clean_search/best.jsonl"
AIM = 0.99


def metric(t: dict) -> float | None:
    if not (t.get("passed") and t.get("vmaf") is not None and t.get("util")):
        return None
    v, u = float(t["vmaf"]), float(t["util"])
    return v + min(25.0, max(0.0, 0.55 * (100.0 - v))) * math.log(AIM / max(u, 1e-3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-add", type=int, default=6)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    labels, pilot = {}, {}
    for l in open(ROOT / "data/encoder_rank_all/labels.jsonl"):
        r = json.loads(l); labels[r["clip_id"]] = r
    for l in open(ROOT / "data/encoder_rank_all/trials.jsonl"):
        t = json.loads(l)
        if t["config_id"] == "svt_crf_p4_qm_la" and t.get("crf_estimate") is not None:
            pilot[t["clip_id"]] = float(t["crf_estimate"])
    done: dict[tuple[str, str], dict] = {}
    if OUT.exists():
        for l in open(OUT):
            t = json.loads(l); done[(t["clip_id"], t["config_id"])] = t
    work = ROOT / "work/clean_search"
    for cid in a.clips.split(","):
        assert labels[cid]["split"] != "test", f"{cid} is a test-split clip"
        path = Path(labels[cid]["path"])
        raw, raw_dir = source_cache(path)
        hint = pilot.get(cid)

        def run(lids: list[str]) -> list[dict]:
            todo = [l for l in lids if (cid, l) not in done]
            with ThreadPoolExecutor(a.workers) as ex:
                for t in ex.map(lambda l: run_label_trial(path, l, work, crf_hint=hint, encode_input=raw), todo):
                    done[(cid, t["config_id"])] = t
                    with open(OUT, "a") as f:
                        f.write(json.dumps(t) + "\n")
                    m = metric(t)
                    print(f"  {t['config_id']:34} pass={t['passed']} vmaf={t['vmaf']} util={t.get('util') and round(t['util'], 4)} "
                          f"norm={m and round(m, 3)} cpu={t.get('enc_cpu_seconds')} {t['codes'][:50]}", flush=True)
            return [done[(cid, l)] for l in lids]

        try:
            print(f"== {cid} ({labels[cid]['kind']}): stage A ({len(USE) + 1} configs), CRF hint {hint}", flush=True)
            res = run([hcs.label_id(())] + [hcs.label_id((n,)) for n in USE])
            base, singles = res[0], dict(zip(USE, res[1:]))
            score = lambda t: metric(t) if metric(t) is not None else -1e9  # noqa: E731
            b0 = score(base)
            gains = sorted(((score(t) - b0, n) for n, t in singles.items() if score(t) - b0 > 0.05), reverse=True)
            print(f"  stage A: base {b0:.3f}; gains: " + ", ".join(f"{n} {g:+.2f}" for g, n in gains[:20]), flush=True)
            cur, cur_s, used = (), b0, set()
            for g, n in gains:
                if len(cur) >= a.max_add or hcs.TOOLS[n][0] in used:
                    continue
                cand = cur + (n,)
                t = singles[n] if not cur else run([hcs.label_id(cand)])[0]
                s = score(t)
                keep = s > cur_s + 0.05
                print(f"  stage B: {'+'.join(cand)} -> {s:.3f} ({'keep' if keep else 'drop'})", flush=True)
                if keep:
                    cur, cur_s = cand, s
                    used.add(hcs.TOOLS[n][0])
            best = done[(cid, hcs.label_id(cur))]
            rec = {"clip_id": cid, "kind": labels[cid]["kind"], "base": b0, "best": cur_s, "tools": list(cur),
                   "config_id": hcs.label_id(cur), "vmaf": best.get("vmaf"), "util": best.get("util"),
                   "enc_cpu_seconds": best.get("enc_cpu_seconds"), "base_cpu": base.get("enc_cpu_seconds")}
            with open(BEST, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  BEST {cid}: {'+'.join(cur) or 'base'} {cur_s:.3f} (base {b0:.3f})", flush=True)
        finally:
            if raw_dir:
                shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
