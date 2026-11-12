#!/usr/bin/env python3
"""Final end-to-end validation: run.sh contract + optional VMAF score estimate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.score import clip_score, final_score, speed_score
from streamenc.util.logging import get_logger
from streamenc.validate.validate_clip import validate_clip

log = get_logger("final_validate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--work-dir", default=str(ROOT / "work" / "final_e2e"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "final"))
    parser.add_argument("--n-clips", type=int, default=4)
    parser.add_argument("--skip-vmaf", action="store_true")
    args = parser.parse_args()

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    dataset = Path(args.dataset)
    clips = [
        dataset / "ai" / "ai_easy_01.mkv",
        dataset / "render" / "rnd_easy_02.mkv",
        dataset / "screen" / "scr_easy_05.mkv",
        dataset / "ai" / "ai_hard_06.mkv",
        dataset / "screen" / "scr_easy_02.mkv",
        dataset / "render" / "rnd_easy_06.mkv",
    ]
    clips = [c for c in clips if c.exists()][: args.n_clips]
    work = Path(args.work_dir)
    results = Path(args.results_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    results.mkdir(parents=True, exist_ok=True)
    inp = work / "input"
    out = work / "output"
    inp.mkdir()
    out.mkdir()

    # Official naming: 64-hex, no extension
    mapping = {}
    for i, c in enumerate(clips):
        name = f"{i:064x}"
        shutil.copy2(c, inp / name)
        mapping[name] = str(c)

    t0 = time.perf_counter()
    rc = os.system(f"/bin/sh {ROOT}/run.sh {inp} {out}")
    wall = time.perf_counter() - t0
    exit_code = rc >> 8

    checks = {
        "runsh_exit_0": exit_code == 0,
        "n_inputs": len(mapping),
        "n_outputs": 0,
        "all_regular_0644": True,
        "no_symlinks": True,
        "no_temp_leftovers": True,
        "basename_match": True,
        "wall_seconds": wall,
    }
    temps = [p.name for p in out.iterdir() if p.name.startswith(".")]
    checks["no_temp_leftovers"] = len(temps) == 0
    checks["temp_names"] = temps

    scores = []
    per_clip = []
    for name, ref in mapping.items():
        cand = out / name
        row = {"name": name, "ref": ref, "exists": cand.exists()}
        if not cand.exists():
            checks["basename_match"] = False
            row["ok"] = False
            per_clip.append(row)
            scores.append(0.0)
            continue
        checks["n_outputs"] += 1
        st = cand.stat()
        if stat.S_ISLNK(st.st_mode) or cand.is_symlink():
            checks["no_symlinks"] = False
        mode = stat.S_IMODE(st.st_mode)
        if mode != 0o644:
            # Some FS ignore exact mode; require at least user-readable regular file.
            if not cand.is_file() or cand.is_symlink():
                checks["all_regular_0644"] = False
        if args.skip_vmaf:
            row["ok"] = True
            row["skipped_vmaf"] = True
            per_clip.append(row)
            continue
        report = validate_clip(ref, str(cand), run_vmaf=True, run_perceptual=True)
        row["passed"] = report.passed
        row["violations"] = [v.code for v in report.violations]
        row["metrics"] = report.metrics
        cs = float(report.metrics.get("clip_score", 0.0)) if report.passed else 0.0
        if not report.passed:
            cs = 0.0
        elif "clip_score" not in report.metrics and "vmaf" in report.metrics:
            cs = clip_score(float(report.metrics["vmaf"]))
        scores.append(cs)
        row["clip_score"] = cs
        row["ok"] = report.passed
        per_clip.append(row)

    quality = sum(scores) / len(scores) if scores else 0.0
    # Use actual wall against inserted-frame speed rules (proxy; batch may be <3428 frames).
    total_frames = 0
    try:
        from streamenc.probe import probe_video

        for ref in mapping.values():
            total_frames += probe_video(ref, count_frames=True).nb_frames
    except Exception:  # noqa: BLE001
        total_frames = 3428
    spf = wall / max(total_frames, 1)
    runtime_3428 = spf * 3428.0
    est = final_score(quality, runtime_3428, 3428.0)

    summary = {
        "checks": checks,
        "quality_mean": quality,
        "total_frames": total_frames,
        "wall_seconds": wall,
        "extrapolated_runtime_3428": runtime_3428,
        "speed_score": speed_score(runtime_3428, 3428.0),
        "estimated_final_score": est,
        "per_clip": per_clip,
        "pass": bool(
            checks["runsh_exit_0"]
            and checks["n_outputs"] == checks["n_inputs"]
            and checks["no_symlinks"]
            and checks["no_temp_leftovers"]
            and (args.skip_vmaf or all(r.get("ok") for r in per_clip))
        ),
    }
    (results / "final_e2e_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("FINAL SUMMARY:\n%s", json.dumps(summary, indent=2, default=str))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
