#!/usr/bin/env python3
"""End-to-end evaluation in validator-style batches (2026-11-12, rc3; external review finding 1).

Like e2e_eval.py (same input staging, scoring, final formula), but every consecutive group of --batch clips of each
list is its own run.sh call (fresh process, like a fresh sandbox per batch) and the batch walls are summed, the way
the validator times a submission. The single-call e2e_eval lets 4 workers pull from one queue, so it hides the
"each batch waits for its slowest clip" cost.

  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
      --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/<tag>
"""
from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(ROOT / "scripts"))

from e2e_eval import _score  # noqa: E402
from streamenc.score.competition_score import final_score, speed_score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="clip lists; batches never mix lists")
    ap.add_argument("--model", default=str(ROOT / "models" / "crf_predictor.pkl"))
    ap.add_argument("--cpus", default="0-31")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--score-procs", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=str(ROOT / "work" / "e2e_batched"))
    args = ap.parse_args()

    batches = []
    for lst in args.clips:
        clips = [l.strip() for l in open(lst) if l.strip() and not l.startswith("#")]
        batches += [(Path(lst).stem, clips[i:i + args.batch]) for i in range(0, len(clips), args.batch)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    env = dict(os.environ, BATCH_WORKERS=str(args.batch), CONTENDER_FLAGS=f"--model {args.model}")
    rows_in, bres = [], []
    with open(out / "contender.log", "w") as lf:
        for b, (lst, clips) in enumerate(batches):
            inp, outd = work / f"b{b:02d}" / "input", work / f"b{b:02d}" / "output"
            inp.mkdir(parents=True)
            outd.mkdir()
            for i, c in enumerate(clips):
                name = f"{b + 1:032x}{i + 1:032x}"
                try:
                    os.link(c, inp / name)
                except OSError:
                    shutil.copy2(c, inp / name)
                rows_in.append((b, lst, c, str(outd / name)))
            lf.write(f"===== batch {b} ({lst}, {len(clips)} clips)\n")
            lf.flush()
            t0 = time.perf_counter()
            rc = subprocess.run(["taskset", "-c", args.cpus, "/bin/sh", str(ROOT / "run.sh"), str(inp), str(outd)],
                                env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            wall = time.perf_counter() - t0
            bres.append({"batch": b, "list": lst, "n": len(clips), "rc": rc, "wall_seconds": round(wall, 3)})
            print(f"batch {b} {lst} n={len(clips)} rc={rc} wall={wall:.2f}s", flush=True)
    from streamenc.probe import probe_video

    frames_by = {c: probe_video(c, count_frames=True).nb_frames for _, _, c, _ in rows_in}
    with ProcessPoolExecutor(args.score_procs) as ex:
        rows = list(ex.map(_score, [(c, o) for _, _, c, o in rows_in]))
    for (b, lst, c, o), r in zip(rows_in, rows):
        r["batch"], r["list"], r["frames"] = b, lst, frames_by[c]
        r["sha256"] = hashlib.sha256(Path(o).read_bytes()).hexdigest() if os.path.exists(o) else None
    with open(out / "per_clip.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {"batches": bres}
    for name, sel in [("all", None)] + [(l, l) for l in dict.fromkeys(x["list"] for x in bres)]:
        rs = [r for r in rows if sel is None or r["list"] == sel]
        wall = sum(x["wall_seconds"] for x in bres if sel is None or x["list"] == sel)
        frames = sum(r["frames"] for r in rs)
        q = sum(r["clip_score"] for r in rs) / len(rs)
        summ[name] = {"n": len(rs), "frames": frames, "wall_seconds": round(wall, 2),
                      "runtime_3428": round(wall / frames * 3428, 1), "speed": round(speed_score(wall, frames), 4),
                      "quality": round(q, 4), "final": round(final_score(q, wall, frames), 4),
                      "n_passed": sum(r["passed"] for r in rs)}
    summ.update({"model": args.model, "cpus": args.cpus, "batch": args.batch})
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "batches"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
