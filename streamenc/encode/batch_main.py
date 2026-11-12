"""Batch driver for run.sh — serial or bounded-parallel item processing."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

_enc = "/opt/nvidia-drivers/lib64"
if Path(_enc).is_dir():
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(":") if p]
    if _enc not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([_enc, *parts])

from streamenc.batch.executor import list_input_items, run_batch_parallel, run_batch_serial
from streamenc.util import deadline as _deadline
from streamenc.util.logging import get_logger

log = get_logger("streamenc.batch_main")


class _BackgroundPredictor:
    """2026-11-12 (rc3, review finding 4): unpickling the 15 MB model takes ~1.5 s, serial before any encode.
    Load it on a background thread instead; every item computes its content features first (~3-4 s, no model
    needed), then calls this, which waits for the load. Same model object, same plans, same outputs.
    A failed load falls back to the rules plan, as the old synchronous load did."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fn = None
        self._done = threading.Event()
        threading.Thread(target=self._load, name="model-load", daemon=True).start()

    def _load(self) -> None:
        t = time.perf_counter()
        try:
            from streamenc.ml.crf_predictor import make_predictor

            self._fn = make_predictor(self._path)
            log.info("crf predictor loaded in background (%.2fs): %s", time.perf_counter() - t, self._path)
        except Exception as exc:  # noqa: BLE001
            log.warning("crf predictor unavailable (%s): rules fallback", exc)
        finally:
            self._done.set()

    def __call__(self, features, meta):
        self._done.wait()
        if self._fn is None:
            from streamenc.encode.crf_pipeline import rule_predictor

            return rule_predictor(features, meta)
        return self._fn(features, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bits-per-pixel", type=float, default=0.05)
    parser.add_argument("--max-keyframe-interval", type=float, default=2.0)
    parser.add_argument("--peak-window", type=float, default=2.0)
    parser.add_argument("--peak-ratio", type=float, default=1.5)
    parser.add_argument("--batch-budget", type=float, default=3300.0)
    parser.add_argument("--item-timeout", type=float, default=300.0)
    parser.add_argument("--batch-deadline", type=float, default=420.0,
                        help="hard limit per round of --workers items (0 = off): no child process runs past it")
    parser.add_argument("--no-decision", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--prefer-speed", action="store_true")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "models" / "crf_predictor.pkl"),
                        help="CRF predictor pickle (crf engine); rules fallback when missing")
    parser.add_argument("--engine", choices=("legacy", "crf"), default="crf",
                        help="crf: SVT-AV1 fractional-CRF pilot/final pipeline (default); legacy: VBR controller")
    args = parser.parse_args(argv)

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = list_input_items(inp)
    if not items:
        log.error("no input items")
        return 66

    rounds = math.ceil(len(items) / max(1, args.workers)) if args.mode == "parallel" else len(items)
    _deadline.start(args.batch_deadline * rounds if args.batch_deadline > 0 else None)

    pairs = [(str(p), str(out / p.name)) for p in items]
    # Longest-first scheduling: input size tracks pixels x frames x complexity, so the slowest items
    # start first and the batch tail shrinks (outputs are matched by name, order is irrelevant).
    pairs.sort(key=lambda pr: -os.path.getsize(pr[0]) if os.path.exists(pr[0]) else 0)
    kwargs = {
        "prefer_quality": not args.prefer_speed,
        "allow_filter": not args.no_filter,
        "use_decision": not args.no_decision,
        "bits_per_pixel": args.bits_per_pixel,
        "max_keyframe_interval": args.max_keyframe_interval,
        "peak_window": args.peak_window,
        "peak_ratio": args.peak_ratio,
    }

    process_fn = None
    if args.engine == "crf":
        from streamenc.encode.crf_pipeline import process_item_crf

        process_fn = process_item_crf
        if args.model and Path(args.model).is_file():
            kwargs["predictor"] = _BackgroundPredictor(args.model)
        else:
            log.info("no crf predictor at %s: rules fallback", args.model)
    extra = {"process_fn": process_fn} if process_fn else {}

    t0 = time.perf_counter()
    if args.mode == "parallel":
        # Per-item budget still applied inside process_item when set.
        kwargs["time_budget_seconds"] = args.item_timeout
        result = run_batch_parallel(pairs, workers=args.workers, process_kwargs=kwargs, **extra)
    else:
        result = run_batch_serial(
            pairs,
            process_kwargs=kwargs,
            item_time_budget=args.item_timeout,
            batch_deadline=args.batch_budget,
            **extra,
        )

    # Scrub temps + chmod outputs
    for p in out.glob(".*"):
        if p.name.startswith(".corr-") or p.name.startswith(".streaming-") or p.name.startswith(".bench-"):
            try:
                p.unlink()
            except OSError:
                pass
    done = failed = 0
    for item in result.items:
        if item.ok and item.output and Path(item.output).is_file():
            try:
                os.chmod(item.output, 0o644)
            except OSError:
                pass
            done += 1
            log.info("done %s", Path(item.input).name)
        else:
            failed += 1
            # Ensure no partial/zero score file remains
            cand = out / Path(item.input).name
            if cand.exists():
                try:
                    cand.unlink()
                except OSError:
                    pass
            log.error("FAILED %s: %s", Path(item.input).name, item.reason or "no output")

    wall = time.perf_counter() - t0
    log.info(
        "contender batch mode=%s: %d done, %d failed, wall=%.2fs (executor reported %.2fs)",
        result.mode,
        done,
        failed,
        wall,
        result.wall_seconds,
    )
    if result.abandoned:
        # Items past the batch deadline: their threads are still alive and would block interpreter exit
        # (concurrent.futures joins its workers at exit). Kill any child left, flush the logs, leave now.
        killed = _deadline.kill_children()
        import shutil

        for d in out.glob(".crf-*"):  # work dirs of abandoned items (normally removed by the item itself)
            shutil.rmtree(d, ignore_errors=True)
        log.error("batch deadline: %d item(s) abandoned, %d child process(es) killed; exiting 0",
                  result.abandoned, killed)
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    # Always 0 so one failure does not zero the batch.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
