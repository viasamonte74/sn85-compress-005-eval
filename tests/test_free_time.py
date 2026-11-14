"""2026-11-12: batch state (who is still running) and cancellable commands for free-time work."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.batch import state  # noqa: E402
from streamenc.util.subprocess_run import CancelledCommand, run_command  # noqa: E402


def test_last_one_event():
    state._reset_for_tests()
    for k in ("a", "b", "c"):
        state.enter(k)
    ev = state.last_one_event("a")
    assert state.others_active("a") == 2 and not ev.is_set()
    state.leave("b")
    assert not ev.is_set()
    state.leave("c")
    assert ev.is_set() and state.others_active("a") == 0
    state.leave("a")
    assert state.last_one_event("a") is None


def test_late_arrival_clears_alone():
    state._reset_for_tests()
    state.enter("x")
    ex = state.last_one_event("x")
    assert ex.is_set()
    state.enter("y")
    assert not ex.is_set() and not state.last_one_event("y").is_set()
    state.leave("x")
    assert state.last_one_event("y").is_set()
    state.leave("y")


def test_alone_is_never_free():
    state._reset_for_tests()
    state.enter("solo")
    try:
        assert state.last_one_event("solo").is_set()
    finally:
        state.leave("solo")


def test_cancel_kills_the_child():
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t = time.monotonic()
    with pytest.raises(CancelledCommand):
        run_command(["sleep", "30"], timeout=60, cancel=ev)
    assert time.monotonic() - t < 2.0
    assert isinstance(CancelledCommand(["x"], 0), subprocess.TimeoutExpired)


def test_cancel_not_set_runs_normally():
    r = run_command(["sh", "-c", "sleep 0.3; echo ok"], timeout=30, cancel=threading.Event())
    assert r.ok and r.stdout.strip() == "ok"
