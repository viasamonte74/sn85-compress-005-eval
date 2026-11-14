"""Which items of the current batch are still running (2026-11-12, free-time refill).

The validator runs one batch's clips as threads of one process and the batch lasts as long as its slowest clip, so
work a clip does while another clip is still running costs no wall time. ``run_batch_parallel`` registers each item;
an item can ask how many others are still running and get an Event that is set as soon as it is the last one left
(``run_command(..., cancel=event)`` then kills its child, so optional work never extends the batch).
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_active: dict[str, threading.Event] = {}


def enter(key: str) -> None:
    with _lock:
        _active[key] = threading.Event()
        if len(_active) == 1:
            _active[key].set()  # alone: nothing is free
        else:
            for ev in _active.values():  # nobody is alone any more
                ev.clear()


def leave(key: str) -> None:
    with _lock:
        _active.pop(key, None)
        if len(_active) == 1:
            next(iter(_active.values())).set()


def others_active(key: str) -> int:
    with _lock:
        return max(0, len(_active) - (1 if key in _active else 0))


def last_one_event(key: str) -> threading.Event | None:
    """Set once ``key`` is the only item still running (None when ``key`` is not registered)."""
    with _lock:
        return _active.get(key)


def _reset_for_tests() -> None:
    with _lock:
        _active.clear()
