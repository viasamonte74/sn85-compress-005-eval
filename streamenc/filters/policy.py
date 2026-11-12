"""Safe optional pre-filter policy.

Aggressive sharpening / contrast / saturation are forbidden.
Mild denoise is default-off; hard-path may use stronger denoise when
bitrate is otherwise infeasible at 0.05 bpp.
"""

from __future__ import annotations

from streamenc.features.extract import FeatureVector

FILTER_CATALOG = {
    "none": None,
    # Mild spatial+temporal denoise (hqdn3d). Values kept conservative.
    "hqdn3d_light": "hqdn3d=1.0:1.0:2.0:2.0",
    # Spatial-only mild blur substitute when temporal denoise risks ghosting.
    "hqdn3d_spatial": "hqdn3d=1.5:1.5:0:0",
    # Hard-path: stronger denoise for bitrate-incompressible / salvage encodes.
    "hqdn3d_hard": "hqdn3d=4.0:3.0:6.0:4.5",
    "atadenoise_hard": "atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04",
    "hard_denoise": "hqdn3d=3.0:2.0:4.0:3.0,atadenoise=0a=0.015:0b=0.03:1a=0.015:1b=0.03",
    # Mild lowpass via 85% bicubic downscale + lanczos upscale (same output size).
    # {w}/{h} are filled with the source size so the output dimensions match exactly.
    "ds85": "scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # Stronger lowpass for hard-clip rescue (2026-11-10): 2/3 and 1/2 bicubic downscale + lanczos upscale.
    "ds67": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "ds50": "scale=trunc(iw/4)*2:trunc(ih/4)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # Round 2 (2026-11-10, rnd_hard_03: low-pass and denoise barely cut bits, superres did): temporal levers.
    # hold2 / hold3: keep every 2nd / 3rd frame and repeat it (repeated frames cost almost no bits).
    "hold2": "framestep=2,fps={fps}",
    "hold3": "framestep=3,fps={fps}",
    # tmix3: 1-2-1 temporal blend (synthetic motion blur, makes chaotic motion predictable)
    "tmix3": "tmix=frames=3:weights='1 2 1'",
    # Round 3 (reference: sr16 + hold3 = first valid encode of rnd_hard_03, VMAF 56.8; sr16 + hold2 was 1.08x budget)
    "hold2_ds85": "framestep=2,fps={fps},scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "hold2_ds67": "framestep=2,fps={fps},scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # avgNhold: average each group of N frames, then repeat it (hold with motion blur instead of a sampled frame)
    "avg2hold": "tmix=frames=2,framestep=2,fps={fps}",
    "avg3hold": "tmix=frames=3,framestep=3,fps={fps}",
    "hold2_ds50": "framestep=2,fps={fps},scale=trunc(iw/4)*2:trunc(ih/4)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # Wide search 2 (2026-11-11, ds67 was +3.1 VMAF on ai_hard_20_1080): more low-pass shapes
    "ds75": "scale=trunc(iw*3/8)*2:trunc(ih*3/8)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "ds60": "scale=trunc(iw*3/10)*2:trunc(ih*3/10)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "ds67l": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=lanczos,scale={w}:{h}:flags=lanczos",
    "ds67a": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=area,scale={w}:{h}:flags=bicubic",
    "ds67_usm": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.5:5:5:0",
    "hq_ds67": "hqdn3d=1.0:1.0:2.0:2.0,scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "dsw67": "scale=trunc(iw/3)*2:ih:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "dsh67": "scale=iw:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "nlm": "nlmeans=s=2.5:p=5:r=9",
    "cblur": "boxblur=luma_radius=0:luma_power=0:chroma_radius=2:chroma_power=1",
    # Sharpening sweep (2026-11-11): ds67 + unsharp 0.5 was +3.91 VMAF on ai_hard_12_1080 but NEG gap 3.18 (> 3.0).
    # Luma-only unsharp at weaker strengths, alone and after the 67 % / 85 % low-pass; cas = contrast-adaptive.
    "usm10": "unsharp=5:5:0.1:5:5:0",
    "ds67_usm10": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.1:5:5:0",
    "ds85_usm10": "scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.1:5:5:0",
    "usm20": "unsharp=5:5:0.2:5:5:0",
    "ds67_usm20": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.2:5:5:0",
    "ds85_usm20": "scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.2:5:5:0",
    "usm30": "unsharp=5:5:0.3:5:5:0",
    "ds67_usm30": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.3:5:5:0",
    "ds85_usm30": "scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.3:5:5:0",
    "usm40": "unsharp=5:5:0.4:5:5:0",
    "ds67_usm40": "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.4:5:5:0",
    "ds85_usm40": "scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos,unsharp=5:5:0.4:5:5:0",
    "usm3_30": "unsharp=3:3:0.3:3:3:0",
    "usm7_20": "unsharp=7:7:0.2:7:7:0",
    "cas20": "cas=strength=0.2:planes=1",
    "cas40": "cas=strength=0.4:planes=1",
    # q07/q08 study (2026-11-11, user's preprocessing list): milder temporal / chroma-only / bm3d variants
    "atad_light": "atadenoise=0a=0.01:0b=0.02:1a=0.01:1b=0.02",
    "hqdn3d_temporal": "hqdn3d=0:0:3:3",
    "hqdn3d_chroma": "hqdn3d=0:3:0:4",
    "cblur4": "boxblur=luma_radius=0:luma_power=0:chroma_radius=4:chroma_power=1",
    "bm3d_s3": "bm3d=sigma=3:block=8:bstep=4:group=1:estim=basic",
    "nlm_light": "nlmeans=s=1.5:p=5:r=7",
    # q07/q08 round 2: chroma-only low-pass variants and combinations (VMAF is luma-only; chroma gates still apply)
    "cblur8": "boxblur=luma_radius=0:luma_power=0:chroma_radius=8:chroma_power=1",
    "cblur4x2": "boxblur=luma_radius=0:luma_power=0:chroma_radius=4:chroma_power=2",
    "c410": "format=yuv410p,format=yuv420p",
    "cblur4_ds85": "boxblur=luma_radius=0:luma_power=0:chroma_radius=4:chroma_power=1,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # 2026-11-11 round 5 (research): stronger chroma blur + the same 85 % luma low-pass
    "cblur8_ds85": "boxblur=luma_radius=0:luma_power=0:chroma_radius=8:chroma_power=1,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # 2026-11-12 round 6 (research, q08): luma low-pass strength / shape under cblur8, and cblur16
    "cblur8_ds80": "boxblur=luma_radius=0:luma_power=0:chroma_radius=8:chroma_power=1,scale=trunc(iw*2/5)*2:trunc(ih*2/5)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "cblur8_ds90": "boxblur=luma_radius=0:luma_power=0:chroma_radius=8:chroma_power=1,scale=trunc(iw*9/20)*2:trunc(ih*9/20)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "cblur8_ds85l": "boxblur=luma_radius=0:luma_power=0:chroma_radius=8:chroma_power=1,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=lanczos,scale={w}:{h}:flags=lanczos",
    "cblur8_gb6": "boxblur=luma_radius=0:luma_power=0:chroma_radius=8:chroma_power=1,gblur=sigma=0.6:planes=1",
    "cblur16_ds85": "boxblur=luma_radius=0:luma_power=0:chroma_radius=16:chroma_power=1,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    # 2026-11-12 round 7 (research, q08): edge-preserving chroma smoothing (luma untouched: psnr_y = inf) + ds85.
    # Hypothesis: cblur16 smears chroma across sharp edges (anim_hard_007 -0.78 / -0.39, hfr_rnd_hard_03 NEG gap +0.66).
    "csm5_8_ds85": "smartblur=lr=0.1:ls=0:lt=0:cr=5:cs=1:ct=8,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "csm5_20_ds85": "smartblur=lr=0.1:ls=0:lt=0:cr=5:cs=1:ct=20,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "cbil8_ds85": "bilateral=sigmaS=8:sigmaR=0.12:planes=6,scale=trunc(iw*17/40)*2:trunc(ih*17/40)*2:flags=bicubic,scale={w}:{h}:flags=lanczos",
    "gblur05": "gblur=sigma=0.5",
    "gblur08": "gblur=sigma=0.8",
}


def filter_vf(name: str) -> str | None:
    if name not in FILTER_CATALOG:
        raise KeyError(name)
    return FILTER_CATALOG[name]


def choose_prefilter(features: FeatureVector, *, allow_filter: bool = True) -> str:
    """Return a catalog key. Default none; light denoise only for unstable motion."""
    if not allow_filter:
        return "none"
    motion = float(features.motion_complexity)
    instability = float(features.temporal_instability)
    # Screen-like static content: never filter (edges/text).
    if float(features.chroma_complexity) < 0.05 and float(features.spatial_complexity) >= 5.0:
        return "none"
    if instability >= 0.35 and motion >= 0.04:
        return "hqdn3d_light"
    return "none"
