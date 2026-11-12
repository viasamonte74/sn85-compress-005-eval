#!/usr/bin/env python3
"""End-to-end contender evaluation on a clip list: run.sh under a CPU set like the sandbox (32 CPUs),
then validator-equivalent scoring of every output (validate_clip == vidaio scoring, see official_score.py).

  python scripts/e2e_eval.py --clips data/e2e_test16.txt --model models/crf_predictor_eval.pkl \
      --cpus 0-31 --out results/e2e/<tag>
Writes summary.json + per_clip.jsonl and the contender log; inputs are hard links with 64-hex names.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.score.competition_score import final_score, speed_score  # noqa: E402


def _score(args: tuple[str, str]) -> dict:
    from streamenc.validate.validate_clip import validate_clip

    ref, cand = args
    if not os.path.exists(cand):
        return {"ref": ref, "exists": False, "passed": False, "clip_score": 0.0}
    rep = validate_clip(ref, cand, run_vmaf=True, run_perceptual=True)
    m = rep.metrics
    return {"ref": ref, "exists": True, "passed": bool(rep.passed), "violations": [v.code for v in rep.violations],
            "vmaf": m.get("vmaf"), "vmaf_neg": m.get("vmaf_neg"), "clip_score": float(m.get("clip_score", 0.0)) if rep.passed else 0.0,
            "target_kbps": m.get("target_kbps"), "avg_kbps": (m.get("streaming") or {}).get("avg_kbps"),
            "peak_kbps": (m.get("streaming") or {}).get("peak_kbps"),
            "util": ((m.get("streaming") or {}).get("avg_kbps") or 0) / (m.get("target_kbps") or 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", default=str(ROOT / "models" / "crf_predictor.pkl"))
    ap.add_argument("--cpus", default="0-31")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score-procs", type=int, default=20)  # untimed scoring after run.sh: 1 thread per VMAF process, CPU otherwise idle
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=str(ROOT / "work" / "e2e"))
    args = ap.parse_args()

    clips = [l.strip() for l in open(args.clips) if l.strip() and not l.startswith("#")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    inp, outd = work / "input", work / "output"
    inp.mkdir(parents=True)
    outd.mkdir()
    mapping = {}
    for i, c in enumerate(clips):
        name = f"{i + 1:064x}"
        try:
            os.link(c, inp / name)
        except OSError:
            shutil.copy2(c, inp / name)
        mapping[name] = c
    env = dict(os.environ, BATCH_WORKERS=str(args.workers), CONTENDER_FLAGS=f"--model {args.model}")
    t0 = time.perf_counter()
    with open(out / "contender.log", "w") as lf:
        rc = subprocess.run(["taskset", "-c", args.cpus, "/bin/sh", str(ROOT / "run.sh"), str(inp), str(outd)],
                            env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0
    from streamenc.probe import probe_video

    frames = sum(probe_video(c, count_frames=True).nb_frames for c in clips)
    with ProcessPoolExecutor(args.score_procs) as ex:
        rows = list(ex.map(_score, [(mapping[n], str(outd / n)) for n in mapping]))
    import hashlib  # 2026-11-11 stability runs: identical outputs across repeats?
    for n, r in zip(mapping, rows):
        f_ = outd / n
        r["sha256"] = hashlib.sha256(f_.read_bytes()).hexdigest() if f_.exists() else None
    q = sum(r["clip_score"] for r in rows) / len(rows)
    spd = speed_score(wall, frames)
    summary = {"rc": rc, "n": len(rows), "frames": frames, "wall_seconds": round(wall, 2),
               "spf": wall / frames, "runtime_3428": round(wall / frames * 3428, 1), "speed": round(spd, 4),
               "quality": round(q, 4), "final": round(final_score(q, wall, frames), 4),
               "n_passed": sum(r["passed"] for r in rows), "model": args.model, "cpus": args.cpus,
               "workers": args.workers}
    with open(out / "per_clip.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{Path(r['ref']).stem:26s} pass={r['passed']} vmaf={r.get('vmaf') and round(r['vmaf'], 2)} "
              f"score={r['clip_score']:.4f} {','.join(r.get('violations') or [])}")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
