"""Competition contender entry: one input → one output (Phase 7–10 pipeline).

Used by ``run.sh`` via ``python -m streamenc.encode.contender INPUT OUTPUT``.
Always exits 0/1 per item; ``run.sh`` itself always exits 0 for the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure NVENC encode library is visible in sandbox/host images that
# installed display-driver userspace under /opt/nvidia-drivers.
_enc = "/opt/nvidia-drivers/lib64"
if Path(_enc).is_dir():
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(":") if p]
    if _enc not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([_enc, *parts])

from streamenc.pipeline.item import process_item
from streamenc.util.logging import get_logger

log = get_logger("streamenc.contender")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits-per-pixel", type=float, default=0.05)
    parser.add_argument("--max-keyframe-interval", type=float, default=2.0)
    parser.add_argument("--peak-window", type=float, default=2.0)
    parser.add_argument("--peak-ratio", type=float, default=1.5)
    parser.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0=unlimited")
    parser.add_argument("--no-decision", action="store_true", help="force fixed NVENC p6 baseline")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--prefer-speed", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)

    result = process_item(
        args.input,
        args.output,
        prefer_quality=not args.prefer_speed,
        allow_filter=not args.no_filter,
        use_decision=not args.no_decision,
        bits_per_pixel=args.bits_per_pixel,
        max_keyframe_interval=args.max_keyframe_interval,
        peak_window=args.peak_window,
        peak_ratio=args.peak_ratio,
        time_budget_seconds=args.time_budget or None,
    )
    # Never leave temp junk in the output directory.
    out_dir = Path(args.output).parent
    for pat in (".corr-partial-*", ".corr-valid-*.mp4", ".streaming-partial-*", ".bench-partial-*"):
        for p in out_dir.glob(pat):
            try:
                p.unlink()
            except OSError:
                pass

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    if result.ok:
        log.info(
            "ok %s config=%s rule=%s prefilter=%s attempts=%d t=%.2fs",
            result.clip_id,
            result.config_id,
            result.rule_id,
            result.prefilter,
            result.attempts,
            result.total_seconds,
        )
        # Ensure mode 0644 plain file.
        try:
            os.chmod(args.output, 0o644)
        except OSError:
            pass
        return 0
    log.error("FAILED %s: %s", result.clip_id, result.reason or "encode failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
