"""Four-clip batch executor tuned for L40S / 32 CPU.

run.sh uses the parallel mode (BATCH_WORKERS=4 threads, one clip each, SVT-AV1 on the CPU) with the batch
deadline of ``streamenc.util.deadline``; the serial mode is kept for experiments.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from streamenc.pipeline.item import ItemResult, process_item
from streamenc.score import final_score
from streamenc.util import deadline as _deadline
from streamenc.util.logging import get_logger

log = get_logger("streamenc.batch")


@dataclass
class BatchResult:
    items: list[ItemResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    mode: str = "serial"
    total_frames: int = 0
    mean_clip_score_proxy: float | None = None  # filled if scores provided
    estimated_final_score: float | None = None
    abandoned: int = 0  # items still running when the batch deadline passed (their threads are left behind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "wall_seconds": self.wall_seconds,
            "total_frames": self.total_frames,
            "n_ok": sum(1 for i in self.items if i.ok),
            "n_fail": sum(1 for i in self.items if not i.ok),
            "mean_clip_score_proxy": self.mean_clip_score_proxy,
            "estimated_final_score": self.estimated_final_score,
            "items": [i.to_dict() for i in self.items],
        }


def list_input_items(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    items = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        items.append(p)
    return items


def run_batch_serial(
    pairs: list[tuple[str, str]],
    *,
    process_kwargs: dict[str, Any] | None = None,
    item_time_budget: float | None = None,
    batch_deadline: float | None = None,
    process_fn: Callable[..., ItemResult] = process_item,
) -> BatchResult:
    kwargs = dict(process_kwargs or {})
    t0 = time.perf_counter()
    items: list[ItemResult] = []
    for inp, out in pairs:
        if batch_deadline is not None:
            left = batch_deadline - (time.perf_counter() - t0)
            if left < 5.0:
                items.append(
                    ItemResult(
                        input=inp,
                        output="",
                        ok=False,
                        clip_id=Path(inp).name,
                        config_id="",
                        rule_id="SKIP",
                        prefilter="none",
                        feature_seconds=0.0,
                        encode_seconds=0.0,
                        total_seconds=0.0,
                        attempts=0,
                        reason="batch time budget exhausted",
                    )
                )
                continue
            kwargs["time_budget_seconds"] = min(item_time_budget or left, left) if item_time_budget else left
        elif item_time_budget is not None:
            kwargs["time_budget_seconds"] = item_time_budget
        log.info("batch item %s → %s", inp, out)
        items.append(process_fn(inp, out, **kwargs))
    wall = time.perf_counter() - t0
    total_frames = sum(i.frames for i in items)
    return BatchResult(items=items, wall_seconds=wall, mode="serial", total_frames=total_frames)


#: seconds the parallel executor keeps waiting after the batch deadline: by then run_command has killed every
#: child and refuses new ones, so a normal item returns within a second or two
DEADLINE_GRACE_SECONDS = 10.0


def _failed_item(inp: str, rule_id: str, reason: str) -> ItemResult:
    return ItemResult(
        input=inp,
        output="",
        ok=False,
        clip_id=Path(inp).name,
        config_id="",
        rule_id=rule_id,
        prefilter="none",
        feature_seconds=0.0,
        encode_seconds=0.0,
        total_seconds=0.0,
        attempts=0,
        reason=reason[:240],
    )


def run_batch_parallel(
    pairs: list[tuple[str, str]],
    *,
    workers: int = 2,
    process_kwargs: dict[str, Any] | None = None,
    process_fn: Callable[..., ItemResult] = process_item,
) -> BatchResult:
    """Bounded parallel encode. Honours the batch deadline (``streamenc.util.deadline``): items still running
    DEADLINE_GRACE_SECONDS after it are reported failed and left behind (``BatchResult.abandoned``)."""
    kwargs = dict(process_kwargs or {})
    t0 = time.perf_counter()
    items: list[ItemResult] = []
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futs = {pool.submit(process_fn, inp, out, **kwargs): (inp, out) for inp, out in pairs}
    pending = set(futs)
    while pending:
        left = _deadline.remaining()
        wait_s = None if left is None else max(0.0, left + DEADLINE_GRACE_SECONDS)
        done, pending = wait(pending, timeout=wait_s, return_when=FIRST_COMPLETED)
        for fut in done:
            try:
                items.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                inp, out = futs[fut]
                log.error("parallel item failed %s: %s", inp, exc)
                items.append(_failed_item(inp, "ERROR", str(exc)))
        if not done:  # deadline + grace passed with items still running
            break
    for fut in pending:
        fut.cancel()
        inp, _ = futs[fut]
        log.error("batch deadline: abandoning %s", inp)
        items.append(_failed_item(inp, "DEADLINE", "batch deadline passed"))
    pool.shutdown(wait=not pending, cancel_futures=True)
    # Preserve input order
    order = {inp: i for i, (inp, _) in enumerate(pairs)}
    items.sort(key=lambda r: order.get(r.input, 0))
    wall = time.perf_counter() - t0
    total_frames = sum(i.frames for i in items)
    return BatchResult(items=items, wall_seconds=wall, mode=f"parallel:{workers}", total_frames=total_frames,
                       abandoned=len(pending))


def attach_score_estimate(batch: BatchResult, clip_scores: list[float]) -> BatchResult:
    if not clip_scores:
        return batch
    quality = sum(clip_scores) / len(clip_scores)
    frames = batch.total_frames or 3428
    # Extrapolate wall time to 3428 frames using observed spf.
    spf = batch.wall_seconds / max(frames, 1)
    runtime = spf * 3428.0
    batch.mean_clip_score_proxy = quality
    batch.estimated_final_score = final_score(quality, runtime, 3428.0)
    return batch
