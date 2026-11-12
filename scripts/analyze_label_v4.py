#!/usr/bin/env python3
"""Merge the v4 label shards (fixed VMAF) and summarise: per-clip best config, NVENC vs SVT vs
libaom, hard-clip picture, official-score comparison, and a speed-aware view (each doubling of
total run time costs 5.08 % of the final score)."""
from __future__ import annotations
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/workspace/dataset/official_scores.jsonl")

def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "encoder_rank_v4" / "shard*" / pattern))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows

def fam(cid):
    if cid.startswith("nv_av1"): return "nvenc_av1"
    if cid.startswith("nv_hevc"): return "nvenc_hevc"
    if cid.startswith("svt17"): return "svt17"
    if cid.startswith("svt"): return "svt"
    if cid.startswith("aom"): return "aom"
    if cid.startswith("x265"): return "x265"
    if cid.startswith("vp9"): return "vp9"
    return cid

def main():
    labels = load("labels.jsonl"); trials = load("trials.jsonl")
    off = {json.loads(l)["clip"]: json.loads(l) for l in open(OFFICIAL)} if OFFICIAL.exists() else {}
    by_clip = defaultdict(dict)
    for t in trials:
        by_clip[t["clip_id"]][t["config_id"]] = t
    clips = sorted(by_clip)
    print(f"{len(clips)} clips, {len(trials)} trials\n")
    print(f"{'clip':13}{'best config':22}{'vmaf':>7}{'score':>7} | {'best svt':>8} {'best nvenc':>10} {'best aom':>8} | {'official':>8}  nv-svt")
    agg = defaultdict(list); fam_best = defaultdict(list); zero = 0
    for c in clips:
        tr = by_clip[c]; valid = {k: v for k, v in tr.items() if v.get("passed") and v.get("passed_streaming") and v.get("vmaf") is not None}
        if not valid: print(f"{c:13} NO VALID TRIAL"); continue
        best = max(valid.values(), key=lambda v: v["vmaf"])
        fb = {}
        for f in ("svt", "nvenc_av1", "aom"):
            cand = [v for k, v in valid.items() if fam(k) == f]
            fb[f] = max(v["vmaf"] for v in cand) if cand else float("nan")
            if cand: fam_best[f].append(fb[f])
        o = off.get(c, {}).get("vmaf", float("nan"))
        if best["clip_score"] <= 0: zero += 1
        print(f"{c:13}{best['config_id']:22}{best['vmaf']:7.2f}{best['clip_score']:7.3f} | {fb['svt']:8.2f} {fb['nvenc_av1']:10.2f} {fb['aom']:8.2f} | {o:8.2f}  {fb['nvenc_av1']-fb['svt']:+.2f}")
        for k, v in valid.items(): agg[k].append((v["vmaf"], v["encode_seconds"], v["clip_score"]))
    print(f"\nclips scoring 0 (VMAF<60) even with the best config: {zero}/{len(clips)}")
    print("\nPer-config over all clips where valid (n, mean VMAF, mean clip score, mean encode s incl. rate-fit attempts):")
    for k, v in sorted(agg.items(), key=lambda kv: -sum(x[0] for x in kv[1]) / len(kv[1])):
        n = len(v); print(f"  {k:22} n={n:2d} vmaf={sum(x[0] for x in v)/n:6.2f} score={sum(x[2] for x in v)/n:.3f} enc={sum(x[1] for x in v)/n:6.1f}s")
    print("\nFamily best-per-clip means:", {f: round(sum(v)/len(v), 2) for f, v in fam_best.items()})
    hard = [c for c in clips if "_hard_" in c]
    print(f"\nHard clips ({len(hard)}): best-VMAF distribution:",
          sorted(round(max(v['vmaf'] for v in by_clip[c].values() if v.get('vmaf') is not None), 1) for c in hard))

if __name__ == "__main__":
    main()
