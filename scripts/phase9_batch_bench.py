#!/usr/bin/env python3
"""Phase 9: compare serial vs parallel 4-clip batch wall times on L40S."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.batch.executor import run_batch_parallel, run_batch_serial
from streamenc.util.logging import get_logger

log = get_logger("phase9")


def _pick_batch(dataset: Path) -> list[Path]:
    # Prefer ~diverse short clips totaling a meaningful frame count.
    candidates = [
        dataset / "ai" / "ai_easy_01.mkv",
        dataset / "render" / "rnd_easy_02.mkv",
        dataset / "screen" / "scr_easy_05.mkv",
        dataset / "ai" / "ai_easy_12.mkv",
    ]
    return [p for p in candidates if p.exists()][:4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/dataset")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "phase9"))
    parser.add_argument("--work-dir", default=str(ROOT / "work" / "phase9"))
    args = parser.parse_args()

    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    clips = _pick_batch(Path(args.dataset))
    if len(clips) < 2:
        log.error("need at least 2 clips")
        return 1
    results_dir = Path(args.results_dir)
    work = Path(args.work_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def pairs(tag: str):
        out = work / tag
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        return [(str(c), str(out / c.stem)) for c in clips]

    kwargs = {"prefer_quality": True, "allow_filter": True, "use_decision": True}
    log.info("serial batch (%d clips)", len(clips))
    serial = run_batch_serial(pairs("serial"), process_kwargs=kwargs)
    log.info("parallel-2 batch")
    parallel = run_batch_parallel(pairs("parallel2"), workers=2, process_kwargs=kwargs)

    # Also measure run.sh wall clock on a fresh dir copy with hashed names.
    runsh_dir_in = work / "runsh_in"
    runsh_dir_out = work / "runsh_out"
    if runsh_dir_in.exists():
        shutil.rmtree(runsh_dir_in)
    if runsh_dir_out.exists():
        shutil.rmtree(runsh_dir_out)
    runsh_dir_in.mkdir(parents=True)
    runsh_dir_out.mkdir(parents=True)
    for i, c in enumerate(clips):
        # Extensionless sha-like names
        name = f"{'a'*63}{i}"
        shutil.copy2(c, runsh_dir_in / name)
    t0 = time.perf_counter()
    rc = os.system(
        f"cd {ROOT} && /bin/sh ./run.sh {runsh_dir_in} {runsh_dir_out} > {results_dir}/runsh_batch.log 2>&1"
    )
    runsh_wall = time.perf_counter() - t0
    outs = list(runsh_dir_out.iterdir())
    # Scrub check
    temps = [p for p in outs if p.name.startswith(".")]

    summary = {
        "n_clips": len(clips),
        "clip_ids": [c.stem for c in clips],
        "serial": serial.to_dict(),
        "parallel2": parallel.to_dict(),
        "runsh": {
            "wall_seconds": runsh_wall,
            "exit_code": rc >> 8,
            "n_outputs": len([p for p in outs if p.is_file() and not p.name.startswith(".")]),
            "temp_leftovers": len(temps),
        },
        "winner": "serial"
        if serial.wall_seconds <= parallel.wall_seconds
        else "parallel2",
        "recommendation": (
            "serial"
            if serial.wall_seconds <= parallel.wall_seconds * 1.05
            else "parallel2"
        ),
    }
    # Extrapolate serial to 3428 frames
    if serial.total_frames > 0:
        spf = serial.wall_seconds / serial.total_frames
        summary["serial_extrapolated_3428_s"] = spf * 3428.0
        summary["speed_saturation_target_s"] = 114.0
    (results_dir / "phase9_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Phase9 summary:\n%s", json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
