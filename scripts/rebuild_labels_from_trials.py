#!/usr/bin/env python3
"""Rebuild labels.jsonl from an existing trials.jsonl (same assembly as labeling.label_clip,
without re-encoding). Usage: rebuild_labels_from_trials.py <shard_dir> [<shard_dir> ...]"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import streamenc.ml.encoder_rank.labeling as L

def rebuild(shard: Path, policy_version: int) -> int:
    trials = [json.loads(l) for l in (shard / "trials.jsonl").open() if l.strip()]
    by = defaultdict(list)
    for t in trials: by[t["clip_id"]].append(t)
    paths = {}
    for sub in ("ai", "ai1080", "ai_gen2", "render", "screen", "generated_hard", "open_clips"):
        for p in (Path("/workspace/dataset") / sub).glob("*.mkv"): paths.setdefault(p.stem, p)
    out = (shard / "labels.jsonl").open("w"); n = 0
    for clip_id, tr in by.items():
        path = paths[clip_id]; t0 = time.perf_counter()
        analysis = L.analyze_video(path, mode="sampled", max_samples=24, max_width=640)
        feats = analysis.features; diff = L.difficulty_score(feats)
        regime = L.hard_feature_regime(feats); hard = bool(diff.is_hard or L.is_hard_content(feats))
        family = L.clip_family(path.stem, path); split = L.assign_split(family)
        winner, quality = L.select_winner(tr)
        valid = [t for t in tr if t.get("passed") and t.get("passed_streaming")]
        label = L.empty_label(
            clip_id=path.stem, path=str(path), kind=L.clip_kind(path), tier=L.clip_tier(path.stem), family=family, split=split,
            winner_config_id=winner["config_id"] if winner else None,
            winner_config_index=winner["config_index"] if winner else None,
            winner_prefilter=winner["prefilter"] if winner else None,
            winner_aim=winner.get("aim") if winner else None, winner_vmaf=winner.get("vmaf") if winner else None,
            winner_clip_score=float(winner.get("clip_score") or 0.0) if winner else 0.0,
            winner_util=winner.get("util") if winner else None,
            winner_passed=bool(winner.get("passed")) if winner else False,
            winner_passed_streaming=bool(winner.get("passed_streaming")) if winner else False,
            n_trials=len(tr), n_streaming_pass=sum(1 for t in tr if t.get("passed_streaming")),
            max_vmaf=max((t["vmaf"] for t in valid if t.get("vmaf") is not None), default=None),
            max_clip_score=max((float(t.get("clip_score") or 0.0) for t in valid), default=0.0),
            difficulty_score=float(diff.score), is_hard=hard, regime=regime,
            label_seconds=sum(float(t.get("encode_seconds") or 0.0) for t in tr),
        )
        label["trainable"] = bool(quality["trainable"]); label["winner_confidence"] = float(quality["winner_confidence"])
        label["n_gate_valid"] = int(quality["n_valid"]); label["reject_reason"] = quality.get("reject_reason")
        label["tie_with"] = quality.get("tie_with"); label["efficient_config_id"] = quality.get("efficient_config_id")
        label["label_policy_version"] = policy_version; label["rebuilt_from_trials"] = True
        out.write(json.dumps(label) + "\n"); n += 1
    out.close(); return n

if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, rebuild(Path(d), 3), "labels rebuilt")
