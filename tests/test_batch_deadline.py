"""rc3 (2026-11-12): batch deadline + background model load (external review findings 3 and 4)."""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamenc.batch import executor  # noqa: E402
from streamenc.encode import batch_main  # noqa: E402
from streamenc.util import deadline  # noqa: E402
from streamenc.util.subprocess_run import run_command  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_deadline():
    deadline.start(None)
    yield
    deadline.start(None)


def test_no_deadline_runs_normally():
    r = run_command(["sh", "-c", "echo hi"], timeout=30)
    assert r.ok and r.stdout.strip() == "hi"
    assert deadline.remaining() is None


def test_deadline_caps_a_stuck_child():
    deadline.start(1.0)
    t = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "30"], timeout=3600)
    assert time.monotonic() - t < 3.0
    assert not deadline._children  # unregistered after wait4


def test_no_child_started_after_deadline():
    deadline.start(0.05)
    time.sleep(0.1)
    t = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "30"])
    assert time.monotonic() - t < 0.5


def test_kill_children_ends_a_running_command():
    box = {}

    def _run():
        box["r"] = run_command(["sleep", "30"], timeout=60)

    th = threading.Thread(target=_run)
    th.start()
    for _ in range(100):
        if deadline._children:
            break
        time.sleep(0.02)
    assert deadline.kill_children() == 1
    th.join(5)
    assert not th.is_alive() and box["r"].returncode == -9


def _ok_item(inp: str, out: str, **_kw):
    return dataclasses.replace(executor._failed_item(inp, "OK", ""), ok=True, output=out)


def test_parallel_executor_abandons_items_past_the_deadline(monkeypatch):
    monkeypatch.setattr(executor, "DEADLINE_GRACE_SECONDS", 0.2)
    release = threading.Event()

    def fn(inp, out, **kw):
        if inp == "hang":
            release.wait(30)  # a hang outside any child process
        return _ok_item(inp, out)

    deadline.start(0.5)
    t = time.monotonic()
    res = executor.run_batch_parallel([("a", "oa"), ("hang", "oh"), ("b", "ob")], workers=4, process_fn=fn)
    release.set()
    assert time.monotonic() - t < 3.0
    assert res.abandoned == 1
    by = {i.input: i for i in res.items}
    assert by["a"].ok and by["b"].ok and not by["hang"].ok and by["hang"].rule_id == "DEADLINE"
    assert [i.input for i in res.items] == ["a", "hang", "b"]


def test_parallel_executor_without_deadline_waits_for_all():
    res = executor.run_batch_parallel([("a", "oa"), ("b", "ob")], workers=2, process_fn=_ok_item)
    assert res.abandoned == 0 and all(i.ok for i in res.items)


def test_background_predictor_waits_for_the_load(monkeypatch):
    import streamenc.ml.crf_predictor as cp

    def slow_make(path):
        time.sleep(0.3)
        return lambda f, m: ("plan", path, f, m)

    monkeypatch.setattr(cp, "make_predictor", slow_make)
    p = batch_main._BackgroundPredictor("model.pkl")
    assert p("feat", "meta") == ("plan", "model.pkl", "feat", "meta")


def test_background_predictor_falls_back_to_rules(monkeypatch):
    import streamenc.ml.crf_predictor as cp
    from streamenc.encode.crf_pipeline import Plan

    def bad_make(path):
        raise RuntimeError("corrupt model")

    monkeypatch.setattr(cp, "make_predictor", bad_make)
    p = batch_main._BackgroundPredictor("model.pkl")
    assert p(None, None) == Plan()
