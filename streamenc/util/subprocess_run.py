from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from streamenc.util import deadline as _deadline


class CancelledCommand(subprocess.TimeoutExpired):
    """The child was killed because its ``cancel`` event was set (free-time work no longer free)."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cpu_seconds: float = 0.0  # user+sys CPU time of the child (contention-free cost measure)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = False,
    text: bool = True,
    cancel: "threading.Event | None" = None,
) -> CommandResult:
    """Run ``argv`` to completion, capturing output and the child's CPU time (``os.wait4``).

    CPU time is per child (not process-wide RUSAGE_CHILDREN), so it stays correct when several
    commands run concurrently from different threads.

    The batch deadline (``streamenc.util.deadline``) caps ``timeout``; once it has passed no new child is
    started (``subprocess.TimeoutExpired`` is raised at once, like any other timeout).
    """
    left = _deadline.remaining()
    if left is not None:
        if left <= 0.0:
            raise subprocess.TimeoutExpired(list(argv), 0.0)
        timeout = left if timeout is None else min(timeout, left)
    proc = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    chunks: dict[str, object] = {}

    def _drain(name: str, stream) -> None:
        try:
            chunks[name] = stream.read()
        finally:
            stream.close()

    threads = [
        threading.Thread(target=_drain, args=("out", proc.stdout), daemon=True),
        threading.Thread(target=_drain, args=("err", proc.stderr), daemon=True),
    ]
    _deadline.register(proc)
    try:
        for t in threads:
            t.start()
        timed_out = False
        end = None if timeout is None else time.monotonic() + timeout  # one budget for both pipes
        cancelled = False
        for t in threads:
            if cancel is None:  # the default path, unchanged
                t.join(None if end is None else max(0.0, end - time.monotonic()))
            else:  # join in slices so the ``cancel`` event (free-time work) can stop the child at once
                while t.is_alive() and not cancel.is_set():
                    if end is not None and time.monotonic() >= end:
                        break
                    t.join(0.2 if end is None else max(0.0, min(0.2, end - time.monotonic())))
                cancelled = t.is_alive() and cancel.is_set()
            if t.is_alive():
                timed_out = True
                break
        if timed_out:
            try:
                os.kill(proc.pid, signal.SIGKILL)  # not proc.kill(): its poll() could reap before our wait4
            except OSError:
                pass
            for t in threads:
                t.join()
        _, status, rusage = os.wait4(proc.pid, 0)
    finally:
        _deadline.unregister(proc)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if timed_out:
        if cancelled:
            raise CancelledCommand(list(argv), timeout or 0.0)
        raise subprocess.TimeoutExpired(list(argv), timeout or 0.0)
    out = chunks.get("out") or ("" if text else b"")
    err = chunks.get("err") or ("" if text else b"")
    result = CommandResult(
        argv=tuple(argv),
        returncode=proc.returncode,
        stdout=out if text else out.decode("utf-8", "replace"),
        stderr=err if text else err.decode("utf-8", "replace"),
        cpu_seconds=float(rusage.ru_utime + rusage.ru_stime),
    )
    if check and not result.ok:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr}"
        )
    return result
