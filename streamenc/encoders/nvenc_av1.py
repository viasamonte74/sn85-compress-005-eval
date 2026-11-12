"""NVIDIA AV1 NVENC configuration builders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NvencAv1Config:
    """Conservative VBR baseline for streaming rules."""

    preset: str = "p6"
    tune: str = "hq"
    rc: str = "vbr"
    multipass: str = "disabled"
    rc_lookahead: int = 8
    spatial_aq: bool = True
    temporal_aq: bool = True
    aq_strength: int = 8
    strict_gop: bool = True
    no_scenecut: bool = True
    forced_idr: bool = True
    # Aim under budget; peak VBV cap under peak_ratio.
    target_fraction: float = 0.975
    maxrate_fraction: float = 1.30
    bufsize_fraction: float = 2.00
    # Bounded corrective aims if measured packets exceed limits.
    retry_fractions: tuple[float, ...] = (0.975, 0.94, 0.90, 0.85, 0.78)


def build_ffmpeg_argv(
    *,
    input_path: str,
    output_path: str,
    kbps: float,
    gop: int,
    cfg: NvencAv1Config,
    pix_fmt: str = "yuv420p",
    color_space: str | None = "bt709",
    color_primaries: str | None = "bt709",
    color_transfer: str | None = "bt709",
    color_range: str | None = "tv",
) -> list[str]:
    maxrate = kbps * cfg.maxrate_fraction
    bufsize = kbps * cfg.bufsize_fraction
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        # Inputs are sha256 names with no extension (FFV1/Matroska).
        "-f",
        "matroska",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "av1_nvenc",
        "-preset",
        cfg.preset,
        "-tune",
        cfg.tune,
        "-rc",
        cfg.rc,
        "-b:v",
        f"{kbps:.3f}k",
        "-maxrate",
        f"{maxrate:.3f}k",
        "-bufsize",
        f"{bufsize:.3f}k",
        "-g",
        str(gop),
        "-forced-idr",
        "1" if cfg.forced_idr else "0",
        "-strict_gop",
        "1" if cfg.strict_gop else "0",
        "-rc-lookahead",
        str(cfg.rc_lookahead),
        "-no-scenecut",
        "1" if cfg.no_scenecut else "0",
        "-spatial-aq",
        "1" if cfg.spatial_aq else "0",
        "-temporal-aq",
        "1" if cfg.temporal_aq else "0",
        "-aq-strength",
        str(cfg.aq_strength),
        "-multipass",
        cfg.multipass,
        "-pix_fmt",
        pix_fmt,
    ]
    if color_space:
        argv.extend(["-colorspace", color_space])
    if color_primaries:
        argv.extend(["-color_primaries", color_primaries])
    if color_transfer:
        argv.extend(["-color_trc", color_transfer])
    if color_range:
        argv.extend(["-color_range", color_range])
    argv.extend(
        [
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            output_path,
        ]
    )
    return argv
