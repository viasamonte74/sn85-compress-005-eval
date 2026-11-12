"""One-shot NVENC availability probe (cached per process).

The evaluation sandbox may expose the GPU without the NVENC userspace library
(``libnvidia-encode.so.1``), or run on a stack where hardware encoding is absent.
``nvenc_available`` encodes 8 synthetic frames with the requested NVENC encoder once;
any failure (missing encoder, missing library, driver error, timeout) means "no NVENC"
and the pipeline substitutes a CPU (SVT-AV1) configuration.

``STREAMENC_FORCE_CPU=1`` forces the CPU path (for tests and for a deliberate CPU-only
entry).
"""

from __future__ import annotations

import os
import threading

from streamenc.util.subprocess_run import run_command

_lock = threading.Lock()
_cache: dict[str, bool] = {}

#: NVENC config -> CPU config substitution used when NVENC is unavailable.
CPU_FALLBACK_DEFAULT = "svt_av1_p6_psnr"
CPU_FALLBACK_HARD = "hard_svt_p5_psnr"


def nvenc_available(encoder: str = "av1_nvenc", *, ffmpeg: str = "ffmpeg", timeout: float = 30.0) -> bool:
    if os.environ.get("STREAMENC_FORCE_CPU", "") not in ("", "0", "false", "no"):
        return False
    with _lock:
        if encoder in _cache:
            return _cache[encoder]
        argv = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", "testsrc2=size=256x128:rate=24:duration=0.34",
            "-pix_fmt", "yuv420p", "-c:v", encoder, "-preset", "p1", "-b:v", "200k",
            "-f", "null", "-",
        ]
        try:
            res = run_command(argv, timeout=timeout)
            ok = bool(res.ok)
        except Exception:  # noqa: BLE001 - any failure means "not available"
            ok = False
        _cache[encoder] = ok
        return ok


def reset_cache() -> None:
    with _lock:
        _cache.clear()
