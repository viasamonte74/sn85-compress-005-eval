"""Batch-wide hard deadline for every child process (2026-11-12, rc3; external review finding 3).

The validator times the whole ``run.sh`` and charges a failed or timed-out batch its full timeout (3600 s,
``epoch_evidence.py``), so one stuck encoder would zero the batch's 4 clips AND add ~3600 s to the summed time.
A missing output only zeroes that clip. So: ``run_command`` caps every child at the batch deadline (and starts
none after it), the parallel executor stops waiting a few seconds later, and ``batch_main`` exits 0 with the
outputs that are done. Normal batches finish far inside the deadline and are unaffected.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

_lock = threading.Lock()
_deadline: float | None = None  # time.monotonic() value; None = no deadline
_children: set[Any] = set()     # live subprocess.Popen objects started by run_command


def start(seconds: float | None) -> None:
    """Set the deadline ``seconds`` from now (None or <= 0 clears it)."""
    global _deadline
    with _lock:
        _deadline = None if not seconds or seconds <= 0 else time.monotonic() + float(seconds)


def remaining() -> float | None:
    """Seconds left before the deadline (negative once passed), or None when no deadline is set."""
    d = _deadline
    return None if d is None else d - time.monotonic()


def register(proc: Any) -> None:
    with _lock:
        _children.add(proc)


def unregister(proc: Any) -> None:
    with _lock:
        _children.discard(proc)


def kill_children() -> int:
    """SIGKILL every live child started by run_command; returns how many were signalled."""
    with _lock:
        procs = list(_children)
    n = 0
    for p in procs:
        # os.kill by pid, never Popen.poll()/kill(): those may reap the child and break run_command's own
        # os.wait4. A registered pid is still unreaped (run_command unregisters only after wait4), so it
        # cannot have been reused.
        try:
            os.kill(p.pid, signal.SIGKILL)
            n += 1
        except OSError:
            pass
    return n
