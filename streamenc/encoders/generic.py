"""Generic FFmpeg encoder argv builders for benchmark challengers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EncoderConfig:
    config_id: str
    codec: str  # av1_nvenc | hevc_nvenc | libsvtav1
    stage: int  # 1 = screen, 2 = quality
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def label(self) -> str:
        return self.config_id


def build_encode_argv(
    *,
    cfg: EncoderConfig,
    input_path: str,
    output_path: str,
    kbps: float,
    gop: int,
    pix_fmt: str = "yuv420p",
    color_space: str | None = "bt709",
    color_primaries: str | None = "bt709",
    color_transfer: str | None = "bt709",
    color_range: str | None = "tv",
) -> list[str]:
    p = cfg.params
    maxrate_fraction = float(p.get("maxrate_fraction", 1.30))
    bufsize_fraction = float(p.get("bufsize_fraction", 2.00))
    maxrate = kbps * maxrate_fraction
    bufsize = kbps * bufsize_fraction

    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
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
    ]
    # Optional lightweight prefilter selected by the decision engine.
    vf = p.get("vf") or p.get("prefilter_vf")
    if vf:
        argv.extend(["-vf", str(vf)])
    argv.extend(["-c:v", cfg.codec])

    if cfg.codec in {"av1_nvenc", "hevc_nvenc"}:
        argv.extend(
            [
                "-preset",
                str(p.get("preset", "p6")),
                "-tune",
                str(p.get("tune", "hq")),
                "-rc",
                str(p.get("rc", "vbr")),
                "-b:v",
                f"{kbps:.3f}k",
                "-maxrate",
                f"{maxrate:.3f}k",
                "-bufsize",
                f"{bufsize:.3f}k",
                "-g",
                str(gop),
                "-forced-idr",
                "1",
                "-strict_gop",
                "1" if p.get("strict_gop", True) else "0",
                "-rc-lookahead",
                str(int(p.get("rc_lookahead", 8))),
                "-no-scenecut",
                "1" if p.get("no_scenecut", True) else "0",
                "-spatial-aq",
                "1" if p.get("spatial_aq", True) else "0",
                "-temporal-aq",
                "1" if p.get("temporal_aq", True) else "0",
                "-aq-strength",
                str(int(p.get("aq_strength", 8))),
                "-multipass",
                str(p.get("multipass", "disabled")),
            ]
        )
        if p.get("highbitdepth"):
            argv.extend(["-highbitdepth", "true"])
            if cfg.codec == "av1_nvenc":
                pix_fmt = "p010le"
            elif cfg.codec == "hevc_nvenc":
                pix_fmt = "p010le"
    elif cfg.codec == "libsvtav1":
        argv.extend(
            [
                "-preset",
                str(int(p.get("preset", 6))),
                "-b:v",
                f"{kbps:.3f}k",
                "-g",
                str(gop),
            ]
        )
        svt = dict(p.get("svtav1_params") or {})
        # Keyint alignment via svt params when useful.
        svt.setdefault("keyint", gop)
        if svt:
            joined = ":".join(f"{k}={v}" for k, v in svt.items())
            argv.extend(["-svtav1-params", joined])
        if p.get("pix_fmt"):
            pix_fmt = str(p["pix_fmt"])
    else:
        raise ValueError(f"unsupported codec {cfg.codec}")

    argv.extend(["-pix_fmt", pix_fmt])
    if color_space:
        argv.extend(["-colorspace", color_space])
    if color_primaries:
        argv.extend(["-color_primaries", color_primaries])
    if color_transfer:
        argv.extend(["-color_trc", color_transfer])
    if color_range:
        argv.extend(["-color_range", color_range])
    argv.extend(["-movflags", "+faststart", "-f", "mp4", output_path])
    return argv


def staged_configs() -> list[EncoderConfig]:
    """Prioritized, non-exhaustive config catalogue."""
    cfgs: list[EncoderConfig] = []
    # Stage 1: core NVENC AV1 presets
    for preset in ("p5", "p6", "p7"):
        cfgs.append(
            EncoderConfig(
                config_id=f"av1_nvenc_{preset}_vbr",
                codec="av1_nvenc",
                stage=1,
                params={"preset": preset, "rc": "vbr"},
                notes="baseline VBR family",
            )
        )
    # RC / AQ / lookahead variants around p6
    cfgs.append(
        EncoderConfig(
            "av1_nvenc_p6_noaq",
            "av1_nvenc",
            1,
            {"preset": "p6", "spatial_aq": False, "temporal_aq": False},
            "disable AQ",
        )
    )
    cfgs.append(
        EncoderConfig(
            "av1_nvenc_p6_la32",
            "av1_nvenc",
            1,
            {"preset": "p6", "rc_lookahead": 32},
            "longer lookahead",
        )
    )
    cfgs.append(
        EncoderConfig(
            "av1_nvenc_p6_multipass_qres",
            "av1_nvenc",
            1,
            {"preset": "p6", "multipass": "qres"},
            "quarter-res multipass",
        )
    )
    cfgs.append(
        EncoderConfig(
            "av1_nvenc_p6_10bit",
            "av1_nvenc",
            1,
            {"preset": "p6", "highbitdepth": True},
            "10-bit encode from 8-bit input",
        )
    )
    # HEVC challenger
    cfgs.append(
        EncoderConfig(
            "hevc_nvenc_p6_vbr",
            "hevc_nvenc",
            1,
            {"preset": "p6", "rc": "vbr"},
            "HEVC NVENC challenger",
        )
    )
    # SVT-AV1 presets + tune
    for preset in (5, 6, 7):
        cfgs.append(
            EncoderConfig(
                f"svt_av1_p{preset}_psnr",
                "libsvtav1",
                1,
                {"preset": preset, "svtav1_params": {"tune": 1}},
                "SVT-AV1 PSNR tune",
            )
        )
    cfgs.append(
        EncoderConfig(
            "svt_av1_p6_vq",
            "libsvtav1",
            1,
            {"preset": 6, "svtav1_params": {"tune": 0}},
            "SVT-AV1 VQ/VMAF-oriented tune",
        )
    )
    cfgs.append(
        EncoderConfig(
            "svt_av1_p6_10bit",
            "libsvtav1",
            1,
            {"preset": 6, "pix_fmt": "yuv420p10le", "svtav1_params": {"tune": 1}},
            "SVT 10-bit",
        )
    )
    # Hard-path catalogue: slower / fuller-budget oriented.
    cfgs.append(
        EncoderConfig(
            "hard_nvenc_p7_fill",
            "av1_nvenc",
            2,
            {
                "preset": "p7",
                "rc": "vbr",
                "rc_lookahead": 32,
                "spatial_aq": True,
                "temporal_aq": True,
                "aq_strength": 10,
                "multipass": "qres",
                "maxrate_fraction": 1.45,
                "bufsize_fraction": 2.2,
            },
            "hard: NVENC p7 + LA32 + high AQ + near-peak maxrate",
        )
    )
    cfgs.append(
        EncoderConfig(
            "hard_nvenc_p6_noaq_fill",
            "av1_nvenc",
            2,
            {
                "preset": "p6",
                "spatial_aq": False,
                "temporal_aq": False,
                "rc_lookahead": 32,
                "maxrate_fraction": 1.45,
                "bufsize_fraction": 2.0,
            },
            "hard: noaq + long LA + aggressive maxrate",
        )
    )
    for preset, tune, tag in ((5, 1, "psnr"), (4, 1, "psnr"), (5, 0, "vq"), (4, 0, "vq")):
        cfgs.append(
            EncoderConfig(
                f"hard_svt_p{preset}_{tag}",
                "libsvtav1",
                2,
                {
                    "preset": preset,
                    "svtav1_params": {"tune": tune},
                    "maxrate_fraction": 1.35,
                    "bufsize_fraction": 2.0,
                },
                f"hard: SVT-AV1 p{preset} tune={tune}",
            )
        )
    return cfgs
