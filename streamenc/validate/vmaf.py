"""Official-style VMAF / VMAF-NEG measurement via ffmpeg libvmaf.

Pinned models and filter options match miner-vidaio ``backends_real.py``:
- primary: ``version=vmaf_v0.6.1``
- secondary: ``version=vmaf_v0.6.1neg``
- ``n_threads=1``, ``n_subsample=1``, ``pool=mean``
- filter input order: candidate first, reference second
- BOTH files are first canonicalized exactly like validator-vidaio
  ``vidaio/scoring/canonicalize.py`` + ``scoring_worker/service.py``: first video
  stream, ``format=yuv420p,setpts=PTS-STARTPTS``, ``-fps_mode cfr``, bt709 tags,
  untagged matrix/range filled as bt709/tv, written to y4m. Measuring on the raw
  MKV/MP4 instead mispairs frames (ms-rounded MKV timestamps) and under-reads VMAF
  by up to ~18 points on high-motion clips (measured 2026-11-08 on ai_hard_04).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from streamenc.util.paths import project_root
from streamenc.util.subprocess_run import run_command

DEFAULT_VMAF_MODEL = "version=vmaf_v0.6.1"
SECONDARY_VMAF_MODEL = "version=vmaf_v0.6.1neg"
VMAF_MODEL_DELTA_MAX = 3.0


def default_ffmpeg_with_vmaf() -> str:
    bundled = project_root() / "third_party" / "ffmpeg-n8.1-linux64-gpl" / "bin" / "ffmpeg"
    if bundled.is_file():
        return str(bundled)
    return "ffmpeg"


CANONICAL_PIX_FMT = "yuv420p"
CANONICAL_COLOR = "bt709"
CANONICAL_RANGE = "tv"
CANONICAL_TIMESCALE = 90000
_UNSPECIFIED_TAGS = frozenset({"", "unknown", "unspecified", "reserved"})


def relabel_vf(pix_fmt: str | None, color_space: str | None, color_range: str | None) -> str | None:
    """2026-11-13: the filter that makes an encode keep an untagged source's pixels. The validator relabels an untagged
    reference as BT.709 (canonicalization below); the encoder writes BT.709 tags, so without this ffmpeg 8 CONVERTS
    unknown -> BT.709 through RGB and the output no longer matches the reference (luma 29 vs 39 dB on a Wan clip).
    None for tagged sources (their encodes are unchanged)."""
    fill = untagged_color_fill(pix_fmt, color_space or "unknown", color_range or "unknown")
    kv = ([f"colorspace={CANONICAL_COLOR}"] if fill.get("fill_color_space") else []) + \
         ([f"range={CANONICAL_RANGE}"] if fill.get("fill_color_range") else [])
    return "setparams=" + ":".join(kv) if kv else None


def untagged_color_fill(pix_fmt: str | None, color_space: str | None, color_range: str | None) -> dict[str, bool]:
    """Mirror of validator ``untagged_color_fill``: fill only *stated-but-unspecified*
    tags of a YUV stream; ``None`` (probe says nothing) fills nothing."""
    if not (pix_fmt or "").startswith("yuv"):
        return {}
    fill: dict[str, bool] = {}
    if color_space is not None and color_space in _UNSPECIFIED_TAGS:
        fill["fill_color_space"] = True
    if color_range is not None and color_range in _UNSPECIFIED_TAGS:
        fill["fill_color_range"] = True
    return fill


def build_canonicalization_plan(
    input_path: str,
    output_path: str,
    *,
    ffmpeg: str = "ffmpeg",
    fill_color_space: bool = False,
    fill_color_range: bool = False,
) -> list[str]:
    """Same argv as validator ``build_canonicalization_plan`` (no scale variant)."""
    vf: list[str] = []
    fill = []
    if fill_color_space:
        fill.append(f"colorspace={CANONICAL_COLOR}")
    if fill_color_range:
        fill.append(f"range={CANONICAL_RANGE}")
    if fill:
        vf.append("setparams=" + ":".join(fill))
    vf.extend((f"format={CANONICAL_PIX_FMT}", "setpts=PTS-STARTPTS"))
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", input_path, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", ",".join(vf), "-fps_mode", "cfr",
        "-color_primaries", CANONICAL_COLOR, "-color_trc", CANONICAL_COLOR,
        "-colorspace", CANONICAL_COLOR, "-video_track_timescale", str(CANONICAL_TIMESCALE),
        "-y", output_path,
    ]


def canonicalize(path: str, output_path: str, *, ffmpeg: str | None = None, timeout: float = 3600.0) -> str:
    """Write the validator-canonical y4m of ``path`` to ``output_path``."""
    ffmpeg = ffmpeg or default_ffmpeg_with_vmaf()
    from streamenc.probe import probe_video

    info = probe_video(path, count_frames=False)
    # validator probe (vidaio backends_real.probe): a missing tag is reported as "unknown" -> filled (relabelled)
    fill = untagged_color_fill(info.pix_fmt, info.color_space or "unknown", info.color_range or "unknown")
    argv = build_canonicalization_plan(path, output_path, ffmpeg=ffmpeg, **fill)
    result = run_command(argv, timeout=timeout)
    if not result.ok or not os.path.isfile(output_path):
        raise RuntimeError(f"canonicalization failed: {result.stderr.strip() or result.stdout.strip()}")
    return output_path


def canonicalize_pair(reference: str, candidate: str, tmp: str, *, ffmpeg: str | None = None, timeout: float = 3600.0) -> tuple[str, str]:
    ref = canonicalize(reference, os.path.join(tmp, "reference.y4m"), ffmpeg=ffmpeg, timeout=timeout)
    cand = canonicalize(candidate, os.path.join(tmp, "candidate.y4m"), ffmpeg=ffmpeg, timeout=timeout)
    return ref, cand


def _escape_filter_option(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


@dataclass(frozen=True)
class VmafPair:
    vmaf: float
    vmaf_neg: float
    delta: float
    model_primary: str
    model_secondary: str
    libvmaf_version: str | None
    ffmpeg_path: str

    @property
    def delta_ok(self) -> bool:
        return self.delta <= VMAF_MODEL_DELTA_MAX


def compute_vmaf(
    reference: str,
    candidate: str,
    *,
    model: str = DEFAULT_VMAF_MODEL,
    ffmpeg: str | None = None,
    timeout: float = 3600.0,
) -> tuple[float, str | None]:
    ffmpeg = ffmpeg or default_ffmpeg_with_vmaf()
    with tempfile.TemporaryDirectory(prefix="vmaf-") as tmp:
        log_path = os.path.join(tmp, "vmaf.json")
        graph = (
            "[0:v:0][1:v:0]libvmaf="
            f"model={_escape_filter_option(model)}"
            f":log_fmt=json:log_path={_escape_filter_option(log_path)}"
            ":n_threads=1:n_subsample=1:pool=mean"
        )
        argv = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            candidate,
            "-i",
            reference,
            "-filter_complex",
            graph,
            "-f",
            "null",
            "-",
        ]
        result = run_command(argv, timeout=timeout)
        if not result.ok:
            raise RuntimeError(f"libvmaf failed: {result.stderr.strip() or result.stdout.strip()}")
        doc = json.loads(Path(log_path).read_text())
        score = float(doc["pooled_metrics"]["vmaf"]["mean"])
        version = doc.get("version") if isinstance(doc.get("version"), str) else None
        if not 0.0 <= score <= 100.0:
            raise RuntimeError(f"pooled VMAF {score!r} outside [0, 100]")
        return score, version


def compute_vmaf_pair(
    reference: str,
    candidate: str,
    *,
    ffmpeg: str | None = None,
    timeout: float = 3600.0,
    canonicalize_inputs: bool = True,
) -> VmafPair:
    """VMAF + VMAF-NEG the validator's way. ``canonicalize_inputs=False`` only when
    both paths are already canonical y4m files (see :func:`canonicalize_pair`)."""
    ffmpeg = ffmpeg or default_ffmpeg_with_vmaf()
    with tempfile.TemporaryDirectory(prefix="canon-") as tmp:
        if canonicalize_inputs:
            reference, candidate = canonicalize_pair(reference, candidate, tmp, ffmpeg=ffmpeg, timeout=timeout)
        vmaf, ver = compute_vmaf(reference, candidate, model=DEFAULT_VMAF_MODEL, ffmpeg=ffmpeg, timeout=timeout)
        vmaf_neg, ver2 = compute_vmaf(
            reference, candidate, model=SECONDARY_VMAF_MODEL, ffmpeg=ffmpeg, timeout=timeout
        )
    return VmafPair(
        vmaf=vmaf,
        vmaf_neg=vmaf_neg,
        delta=abs(vmaf - vmaf_neg),
        model_primary=DEFAULT_VMAF_MODEL,
        model_secondary=SECONDARY_VMAF_MODEL,
        libvmaf_version=ver or ver2,
        ffmpeg_path=ffmpeg,
    )


@dataclass(frozen=True)
class QuickGap:
    vmaf: float
    vmaf_neg: float
    gap: float
    frames: int
    cpu_seconds: float
    wall_seconds: float


def _quick_threads() -> tuple[int, int]:
    """2026-11-12 research (utilisation): (decoder threads, libvmaf threads) of quick_vmaf_gap.

    The in-pipeline VMAF / VMAF-NEG check was bound by single-threaded decoding: q07 4.47 s -> 0.76 s and q01 (1080p)
    16.77 s -> 2.52 s with 8 decoder threads + 4 libvmaf threads, CPU-seconds about the same, pooled VMAF and NEG
    bit-identical. Default "8,4" since 1.0.0rc4 (user yes 2026-11-12); STREAMENC_QUICK_VMAF_THREADS="1,1" = old.
    compute_vmaf_pair keeps the validator's single-thread settings either way."""
    raw = os.environ.get("STREAMENC_QUICK_VMAF_THREADS", "8,4")
    try:
        d, v = (max(1, int(x)) for x in raw.split(","))
    except ValueError:
        d, v = 1, 1
    return d, v


def _quick_pairing(rmeta: object | None, force: bool = False) -> str:
    """2026-11-12 research (prediction fix): how quick_vmaf_gap pairs frames.

    ``setpts=N/FRAME_RATE/TB`` (old) is NOT the validator's pairing on every clip: vs the validator's canonical-y4m
    VMAF it read 90.462 instead of 96.575 on a 50 fps render (1 ms-timebase FFV1 vs a 1/12800 mp4) and 98.654 instead
    of 98.996 on practice q05 (30 fps); exact at 24 fps. Raw timestamps (setpts=PTS-STARTPTS) fail at 24/30 fps.
    STREAMENC_QUICK_VMAF_PAIRING=rate uses an explicit timebase from the reference frame rate on both inputs
    (``settb=1/num,setpts=N*den``): exact on all three (and at 60 fps). Default "rate" since 1.0.0rc4 (user yes
    2026-11-12); STREAMENC_QUICK_VMAF_PAIRING=old = the old pairing."""
    if not (force or os.environ.get("STREAMENC_QUICK_VMAF_PAIRING", "rate") == "rate") or rmeta is None:
        return "setpts=N/FRAME_RATE/TB"
    try:
        from fractions import Fraction

        fr = Fraction(str(getattr(rmeta, "fps_fraction", None) or getattr(rmeta, "fps"))).limit_denominator(1000000)
        return f"settb=1/{fr.numerator},setpts=N*{fr.denominator}"
    except Exception:  # noqa: BLE001
        return "setpts=N/FRAME_RATE/TB"


def quick_vmaf_gap(
    reference: str,
    candidate: str,
    *,
    subsample: int = 5,
    ffmpeg: str | None = None,
    timeout: float = 600.0,
    reference_meta: object | None = None,
    threads: tuple[int, int] | None = None,
    exact_pairing: bool = False,
    ref_trim: tuple[int, int] | None = None,
) -> QuickGap:
    """Runtime estimate of the validator's |VMAF - VMAF-NEG| gap: one ffmpeg pass, both models,
    every ``subsample``-th frame. Frames are paired by index (``setpts=N/FRAME_RATE/TB`` on both
    inputs, the equivalent of the validator's CFR y4m canonicalization; raw-timestamp pairing
    misaligns a 1 ms-timebase FFV1 reference against an mp4). Measured vs the official two full
    passes: |error| <= 0.07 at subsample 5 (720p and 1080p, 2026-11-09).

    Colour: the validator's canonicalization writes bt709/tv tags, and ffmpeg 8 then auto-inserts a
    swscale conversion for any input not tagged bt709 (untagged sources included: probe gives None,
    so nothing is filled). ``format=yuv420p,scale=out_color_matrix=bt709:out_range=tv`` after the same
    tag fill reproduces the canonical frames bit-exactly (untagged / BT.2020 / BT.709 clips). Without
    it the gap reads up to +0.25 high on untagged Wikimedia clips. ``reference_meta`` (VideoMeta of
    the original source) supplies the tags when ``reference`` is a cache of it."""
    import time as _time

    ffmpeg = ffmpeg or default_ffmpeg_with_vmaf()
    with tempfile.TemporaryDirectory(prefix="qgap-") as tmp:
        log_path = os.path.join(tmp, "vmaf.json")
        models = "version=vmaf_v0.6.1\\:name=vmaf|version=vmaf_v0.6.1neg\\:name=neg"
        from streamenc.probe import probe_video

        pair_box = ["setpts=N/FRAME_RATE/TB"]  # replaced by _quick_pairing(rmeta) before the graph is built

        def _chain(meta) -> str:
            fill = untagged_color_fill(getattr(meta, "pix_fmt", None), getattr(meta, "color_space", None) or "unknown",
                                       getattr(meta, "color_range", None) or "unknown") if meta is not None else {}
            f = []
            if fill:
                kv = []
                if fill.get("fill_color_space"):
                    kv.append(f"colorspace={CANONICAL_COLOR}")
                if fill.get("fill_color_range"):
                    kv.append(f"range={CANONICAL_RANGE}")
                f.append("setparams=" + ":".join(kv))
            elif meta is not None:  # restore the source's own tags (a raw cache keeps them; be explicit)
                kv = []
                if getattr(meta, "color_space", None):
                    kv.append(f"colorspace={meta.color_space}")
                if getattr(meta, "color_range", None):
                    kv.append(f"range={meta.color_range}")
                if kv:
                    f.append("setparams=" + ":".join(kv))
            f += [f"format={CANONICAL_PIX_FMT}", f"scale=out_color_matrix={CANONICAL_COLOR}:out_range={CANONICAL_RANGE}",
                  pair_box[0]]
            return ",".join(f)

        try:
            cmeta = probe_video(candidate, count_frames=False)
        except Exception:  # noqa: BLE001
            cmeta = None
        rmeta = reference_meta
        if rmeta is None:
            try:
                rmeta = probe_video(reference, count_frames=False)
            except Exception:  # noqa: BLE001
                rmeta = None
        dec_t, vmaf_t = threads or _quick_threads()
        pair_box[0] = _quick_pairing(rmeta, force=exact_pairing)
        # 2026-11-12: compare a candidate that holds only frames [a, b) of the reference (segment probes)
        _ref_trim = (f"trim=start_frame={int(ref_trim[0])}:end_frame={int(ref_trim[1])},setpts=PTS-STARTPTS,"
                     if ref_trim else "")
        graph = (
            f"[0:v:0]{_chain(cmeta)}[d];"
            f"[1:v:0]{_ref_trim}{_chain(rmeta)}[r];"
            f"[d][r]libvmaf=model='{models}':n_threads={vmaf_t}:n_subsample={max(1, int(subsample))}"
            f":log_fmt=json:log_path={_escape_filter_option(log_path)}"
        )
        argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", str(dec_t), "-i", candidate,
                "-threads", str(dec_t), "-i", reference, "-filter_complex", graph, "-f", "null", "-"]
        t0 = _time.perf_counter()
        result = run_command(argv, timeout=timeout)
        wall = _time.perf_counter() - t0
        if not result.ok:
            raise RuntimeError(f"libvmaf failed: {result.stderr.strip() or result.stdout.strip()}")
        doc = json.loads(Path(log_path).read_text())
        v = float(doc["pooled_metrics"]["vmaf"]["mean"])
        n = float(doc["pooled_metrics"]["neg"]["mean"])
        return QuickGap(vmaf=v, vmaf_neg=n, gap=abs(v - n), frames=len(doc.get("frames") or []),
                        cpu_seconds=float(result.cpu_seconds), wall_seconds=wall)
