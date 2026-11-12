"""Multi-encoder arm registry (SVT-AV1, libaom, x265, VP9, x264, NVENC).

Every arm is a plain dict so it can be serialised into label catalogs. ``build_commands``
turns an arm into one ffmpeg argv (single pass) or two (pass 1 to null, pass 2 to file).
Arms run on the bundled ffmpeg n8.1 (SVT-AV1 v4.3) unless ``bin == "system"``
(ffmpeg 6.1 / SVT-AV1 v1.7).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
FFBIN = _ROOT / "third_party" / "ffmpeg-n8.1-linux64-gpl" / "bin"
FFMPEG = str(FFBIN / "ffmpeg") if (FFBIN / "ffmpeg").is_file() else os.environ.get("STREAMENC_FFMPEG", "ffmpeg")
FFPROBE = str(FFBIN / "ffprobe") if (FFBIN / "ffprobe").is_file() else os.environ.get("STREAMENC_FFPROBE", "ffprobe")
SYSTEM_FFMPEG = "ffmpeg"
THREADS = int(os.environ.get("STREAMENC_ARM_THREADS", "8"))

ARMS: dict[str, dict[str, Any]] = {
    # SVT-AV1 v4.3
    "svt_p6_t1": {"codec": "libsvtav1", "preset": 6, "svt": {"tune": 1}},
    # ---- CRF (constant quality) family, 2026-11-08: deterministic for lp>1 (VBR is not) and
    # +2..+5.5 VMAF over the best VBR arm at equal bitrate on hard clips (CRF sweep, 4 clips).
    "svt_crf_p2_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 2, "svt": {"tune": 1}},
    "svt_crf_p3_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 3, "svt": {"tune": 1}},
    "svt_crf_p4_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 4, "svt": {"tune": 1}},
    "svt_crf_p5_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 5, "svt": {"tune": 1}},
    "svt_crf_p6_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 6, "svt": {"tune": 1}},
    "svt_crf_p4_t0": {"codec": "libsvtav1", "rc": "crf", "preset": 4, "svt": {"tune": 0}},
    "svt_crf_p4_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    "svt_crf_p4_t1_sharp": {"codec": "libsvtav1", "rc": "crf", "preset": 4, "svt": {"tune": 1, "sharpness": 2}},
    "svt_crf_p4_t1_lqb": {"codec": "libsvtav1", "rc": "crf", "preset": 4, "svt": {"tune": 1, "luminance-qp-bias": 5}},
    "svt_crf_p4_t1_cap": {"codec": "libsvtav1", "rc": "crf", "preset": 4, "cap": 1.0, "svt": {"tune": 1}},
    # ---- v7 (2026-11-08): qm+lookahead120 is a consistent +0.19 VMAF at equal bits (v6, 35 clips);
    # luminance-qp-bias 5 is a per-clip gamble (-10..+8, median +0.03); p2 ~= p3 on average but wins
    # big outliers; p8 is the hopeless-clip (max VMAF < 60) speed arm; t0 is the NEG-gap-safe alternative.
    "svt_crf_p2_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 2,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    "svt_crf_p3_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 3,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    # 2026-11-12 q08 lab (research arm, used only via STREAMENC_HARD_ARM): preset 2 + luminance bias + 3 hierarchy
    # levels (short reference chains suit chaotic motion): q08 whole clip 60.97 @ 98.8 % vs p3_lqb 60.15 @ 100.7 %.
    "svt_crf_p2_qm_la_lqb_h3": {"codec": "libsvtav1", "rc": "crf", "preset": 2,
                                "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120, "luminance-qp-bias": 5,
                                        "hierarchical-levels": 3}},
    "svt_crf_p3_qm_la_lqb": {"codec": "libsvtav1", "rc": "crf", "preset": 3,
                             "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120, "luminance-qp-bias": 5}},
    "svt_crf_p4_qm_la_lqb": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                             "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120, "luminance-qp-bias": 5}},
    "svt_crf_p5_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 5,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    "svt_crf_p6_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 6,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    "svt_crf_p8_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 8, "svt": {"tune": 1}},
    # deep-hopeless clips (score 0 whatever is shipped): one fast encode, no rate search
    "svt_crf_p12_t1": {"codec": "libsvtav1", "rc": "crf", "preset": 12, "svt": {"tune": 1}},
    "svt_crf_p3_t0_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 3,
                            "svt": {"tune": 0, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    # ---- hard-clip rescue candidates (2026-11-10): clips where no v8 arm reaches VMAF 60 or none fits the budget
    # even at CRF 63. Base = svt_crf_p4_qm_la + one tool each. Superres = AV1 normative horizontal downscale inside
    # the loop (decoder upscales to full size; TPL is disabled with it).
    "svt_crf_p4_qm_sr12": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                           "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                   "superres-mode": 1, "superres-denom": 12, "superres-kf-denom": 12}},
    "svt_crf_p4_qm_sr16": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                           "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                   "superres-mode": 1, "superres-denom": 16, "superres-kf-denom": 16}},
    "svt_crf_p4_qm_la_vb": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                            "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                    "enable-variance-boost": 1, "variance-boost-strength": 2}},
    "svt_crf_p4_qm_la_tf3": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                             "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120, "tf-strength": 3}},
    "svt_crf_p4_qm_la_fg": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                            "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                    "film-grain": 10, "film-grain-denoise": 1}},
    "svt_crf_p2_qm_sr16": {"codec": "libsvtav1", "rc": "crf", "preset": 2,
                           "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                   "superres-mode": 1, "superres-denom": 16, "superres-kf-denom": 16}},
    "svt_crf_p1_qm_sr16": {"codec": "libsvtav1", "rc": "crf", "preset": 1,
                           "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                   "superres-mode": 1, "superres-denom": 16, "superres-kf-denom": 16}},
    "svt_crf_p0_qm_sr16": {"codec": "libsvtav1", "rc": "crf", "preset": 0,
                           "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                   "superres-mode": 1, "superres-denom": 16, "superres-kf-denom": 16}},
    # p1 / p0 versions of the label base (2026-11-11 sharpening sweep stage 2: p0 was the only SVT tool to gain
    # consistently in the wide searches, +0.8..+3.1 VMAF over p4)
    "svt_crf_p1_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 1,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    "svt_crf_p0_qm_la": {"codec": "libsvtav1", "rc": "crf", "preset": 0,
                         "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120}},
    "svt_crf_p4_qm_sr16_qpc": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                               "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                       "superres-mode": 1, "superres-denom": 16, "superres-kf-denom": 16,
                                       "qp-scale-compress-strength": 3}},
    "svt_crf_p4_qm_la_qpc": {"codec": "libsvtav1", "rc": "crf", "preset": 4,
                             "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "lookahead": 120,
                                     "qp-scale-compress-strength": 3}},
    # "tuned" family = the official-baseline settings of the 2026-11-05 session (beat all 15 v3 arms on 20/26 hard clips)
    "svt_p2_tuned": {"codec": "libsvtav1", "preset": 2, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "luminance-qp-bias": 5}},
    "svt_p3_tuned": {"codec": "libsvtav1", "preset": 3, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "luminance-qp-bias": 5}},
    "svt_p4_tuned": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "luminance-qp-bias": 5}},
    "svt_p6_tuned": {"codec": "libsvtav1", "preset": 6, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0, "luminance-qp-bias": 5}},
    "svt_p4_vq_tuned": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 0, "enable-qm": 1, "qm-min": 0, "luminance-qp-bias": 5}},
    "svt_p5_t1": {"codec": "libsvtav1", "preset": 5, "svt": {"tune": 1}},
    "svt_p4_t1": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1}},
    "svt_p3_t1": {"codec": "libsvtav1", "preset": 3, "svt": {"tune": 1}},
    "svt_p2_t1": {"codec": "libsvtav1", "preset": 2, "svt": {"tune": 1}},
    "svt_p4_t0": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 0}},
    "svt_p4_t1_la120": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120}},
    "svt_p4_t1_vboost": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-variance-boost": 1}},
    "svt_p4_t1_fg8": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "film-grain": 8, "film-grain-denoise": 1}},
    "svt_p4_t1_fg8nd": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "film-grain": 8, "film-grain-denoise": 0}},
    "svt_p4_t1_sr12": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "superres-mode": 1, "superres-denom": 12}},
    "svt_p4_t1_sr_auto": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "superres-mode": 4}},
    "svt_p4_t1_tf0": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-tf": 0}},
    "svt_p4_t1_qm": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "enable-qm": 1, "qm-min": 0}},
    "svt_p3_combo": {"codec": "libsvtav1", "preset": 3,
                     "svt": {"tune": 1, "lookahead": 120, "superres-mode": 4, "enable-qm": 1, "qm-min": 0}},
    "svt_p4_combo": {"codec": "libsvtav1", "preset": 4,
                     "svt": {"tune": 1, "lookahead": 120, "superres-mode": 4, "enable-qm": 1, "qm-min": 0}},
    "svt_p4_la120_qm": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "enable-qm": 1, "qm-min": 0}},
    "svt_p4_la120_tfs3": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "tf-strength": 3}},
    "svt_p4_la120_open": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 1, "lookahead": 120, "irefresh-type": 1}},
    "svt_p4_la120_10b": {"codec": "libsvtav1", "preset": 4, "pix_fmt": "yuv420p10le", "svt": {"tune": 1, "lookahead": 120}},
    "svt_p4_la120_t0": {"codec": "libsvtav1", "preset": 4, "svt": {"tune": 0, "lookahead": 120}},
    # SVT-AV1 v1.7 (system ffmpeg 6.1)
    "svt17_p5_t1": {"codec": "libsvtav1", "bin": "system", "preset": 5, "svt": {"tune": 1}},
    "svt17_p5_t0": {"codec": "libsvtav1", "bin": "system", "preset": 5, "svt": {"tune": 0}},
    "svt17_p4_t0": {"codec": "libsvtav1", "bin": "system", "preset": 4, "svt": {"tune": 0}},
    # libaom (2-pass VBR)
    "aom_c5_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 5, "tune": "psnr"},
    "aom_c4_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 4, "tune": "psnr"},
    "aom_c3_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 3, "tune": "psnr"},
    "aom_c4_ssim": {"codec": "libaom-av1", "two_pass": True, "cpu": 4, "tune": "ssim"},
    "aom_c4_psnr_dn": {"codec": "libaom-av1", "two_pass": True, "cpu": 4, "tune": "psnr", "aom": {"denoise-noise-level": 10}},
    # x265 (2-pass)
    "x265_medium_psnr": {"codec": "libx265", "two_pass": True, "preset": "medium", "tune": "psnr"},
    "x265_slow_psnr": {"codec": "libx265", "two_pass": True, "preset": "slow", "tune": "psnr"},
    "x265_slower_psnr": {"codec": "libx265", "two_pass": True, "preset": "slower", "tune": "psnr"},
    "x265_slow_aq3": {"codec": "libx265", "two_pass": True, "preset": "slow", "x265": {"aq-mode": 3}},
    # VP9 (2-pass)
    "vp9_c3": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 3},
    "vp9_c2": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 2},
    "vp9_c1": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 1},
    # x264 (2-pass)
    "x264_veryslow_psnr": {"codec": "libx264", "two_pass": True, "preset": "veryslow", "tune": "psnr"},
    # NVENC (bundled SDK: uhq tune, full-res multipass, temporal filter)
    "nv_av1_p7_uhq": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32},
    "nv_av1_p7_hq_full": {"codec": "av1_nvenc", "preset": "p7", "tune": "hq", "multipass": "fullres", "la": 32},
    "nv_av1_p7_hq_full_noaq": {"codec": "av1_nvenc", "preset": "p7", "tune": "hq", "multipass": "fullres", "la": 32, "aq": False},
    "nv_av1_p7_uhq_tf": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32, "tf_level": 4},
    "nv_av1_p5_hq": {"codec": "av1_nvenc", "preset": "p5", "tune": "hq", "multipass": "qres", "la": 16},
    "nv_hevc_p7_uhq": {"codec": "hevc_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32},
    "nv_hevc_p7_hq_full": {"codec": "hevc_nvenc", "preset": "p7", "tune": "hq", "multipass": "fullres", "la": 32},
    # ---- wide search 2 (2026-11-11): NVENC variants and more CPU-encoder references
    "nv_av1_p7_uhq_10b": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32,
                          "extra": ["-highbitdepth", "1"]},
    "nv_av1_p7_uhq_la3": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32,
                          "extra": ["-lookahead_level", "3"]},
    "nv_av1_p7_uhq_la60": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 60},
    "nv_av1_p7_uhq_aq15": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32,
                           "extra": ["-aq-strength", "15"]},
    "nv_av1_p7_uhq_noaq": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32, "aq": False},
    "nv_av1_p7_uhq_10b_tf": {"codec": "av1_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32,
                             "tf_level": 4, "extra": ["-highbitdepth", "1"]},
    "nv_hevc_p7_uhq_tf": {"codec": "hevc_nvenc", "preset": "p7", "tune": "uhq", "multipass": "fullres", "la": 32, "tf_level": 4},
    "aom_c2_psnr": {"codec": "libaom-av1", "two_pass": True, "cpu": 2, "tune": "psnr"},
    "aom_c3_ssim": {"codec": "libaom-av1", "two_pass": True, "cpu": 3, "tune": "ssim"},
    "x265_veryslow_psnr": {"codec": "libx265", "two_pass": True, "preset": "veryslow", "tune": "psnr"},
    "vp9_c0": {"codec": "libvpx-vp9", "two_pass": True, "cpu": 0},
}


def get_arm(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise KeyError(f"unknown arm {arm_id}")
    return ARMS[arm_id]


def _head(inp: str, vf: str | None, binary: str) -> list[str]:
    a = [binary, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", inp,
         "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough"]
    if vf:
        a += ["-vf", vf]
    return a


def _tail(out: str, pix_fmt: str = "yuv420p") -> list[str]:
    return ["-pix_fmt", pix_fmt, "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv", "-f", "mp4", out]


def build_commands(arm: dict[str, Any], inp: str, out: str, kbps: float, gop: int, passlog: str,
                   vf: str | None = None, threads: int = THREADS, crf: float | None = None) -> list[list[str]]:
    """``kbps`` is the VBR target (or the budget for a capped-CRF ``cap`` multiplier); ``crf`` is
    required when ``arm["rc"] == "crf"`` (constant quality, deterministic for lp > 1 unless capped)."""
    c = arm["codec"]
    binary = SYSTEM_FFMPEG if arm.get("bin") == "system" else FFMPEG
    maxrate, bufsize = kbps * 1.45, kbps * 2.0
    rate = ["-b:v", f"{kbps:.2f}k"]
    vbv = ["-maxrate", f"{maxrate:.2f}k", "-bufsize", f"{bufsize:.2f}k"]
    pix = arm.get("pix_fmt", "yuv420p")

    if c == "libsvtav1":
        svt = {"keyint": gop, "scd": 0, "lp": threads, **arm.get("svt", {})}
        if arm.get("rc") == "crf":
            if crf is None:
                raise ValueError("crf arm needs a crf value")
            svt["crf"] = f"{float(crf):.2f}"  # fractional (quarter-step) CRF, SVT-AV1 >= 3.0
            rc = []
            if arm.get("cap"):  # capped CRF: SVT max_bit_rate = cap x budget (NOT deterministic for lp > 1)
                rc += ["-maxrate", f"{kbps * float(arm['cap']):.2f}k", "-bufsize", f"{kbps * 2.0:.2f}k"]
        else:
            rc = rate
        enc = ["-c:v", c, "-preset", str(arm["preset"]), *rc, "-g", str(gop),
               "-svtav1-params", ":".join(f"{k}={v}" for k, v in svt.items())]
        return [_head(inp, vf, binary) + enc + _tail(out, pix)]
    if c in ("av1_nvenc", "hevc_nvenc"):
        aq = arm.get("aq", True)
        enc = ["-c:v", c, "-preset", arm["preset"], "-tune", arm["tune"], "-rc", "vbr", *rate, *vbv,
               "-g", str(gop), "-forced-idr", "1", "-strict_gop", "1", "-no-scenecut", "1",
               "-rc-lookahead", str(arm.get("la", 32)), "-multipass", arm.get("multipass", "fullres"),
               "-spatial-aq", "1" if aq else "0", "-temporal-aq", "1" if aq else "0"]
        if "tf_level" in arm:
            enc += ["-tf_level", str(arm["tf_level"])]
        enc += [str(x) for x in arm.get("extra", [])]  # extra NVENC options (wide search, 2026-11-11)
        return [_head(inp, vf, binary) + enc + _tail(out, pix)]

    if c == "libaom-av1":
        aomp = {"enable-keyframe-filtering": 0, **arm.get("aom", {})}
        common = ["-c:v", c, "-cpu-used", str(arm["cpu"]), "-row-mt", "1", "-threads", str(threads),
                  "-tiles", "2x1", "-lag-in-frames", "35", "-auto-alt-ref", "1", "-tune", arm["tune"],
                  "-g", str(gop), "-keyint_min", str(gop),
                  "-aom-params", ":".join(f"{k}={v}" for k, v in aomp.items()),
                  *rate, *vbv, "-undershoot-pct", "5", "-overshoot-pct", "5"]
    elif c == "libx265":
        xp = {"keyint": gop, "min-keyint": gop, "scenecut": 0, "open-gop": 0,
              "vbv-maxrate": int(maxrate), "vbv-bufsize": int(bufsize), "pools": threads,
              "stats": passlog + ".x265", **arm.get("x265", {})}
        common = ["-c:v", c, "-preset", arm["preset"], *rate]
        if arm.get("tune"):
            common += ["-tune", arm["tune"]]
        params = ":".join(f"{k}={v}" for k, v in xp.items())
        p1 = _head(inp, vf, binary) + common + ["-x265-params", params + ":pass=1", "-f", "null", "-"]
        p2 = _head(inp, vf, binary) + common + ["-x265-params", params + ":pass=2"] + _tail(out, pix)
        return [p1, p2]
    elif c == "libvpx-vp9":
        common = ["-c:v", c, "-deadline", "good", "-cpu-used", str(arm["cpu"]), "-row-mt", "1",
                  "-threads", str(threads), "-tile-columns", "2", "-lag-in-frames", "25",
                  "-auto-alt-ref", "1", "-arnr-maxframes", "7", "-arnr-strength", "4",
                  "-g", str(gop), "-keyint_min", str(gop), *rate, *vbv]
    elif c == "libx264":
        common = ["-c:v", c, "-preset", arm["preset"], "-tune", arm.get("tune", "psnr"), *rate, *vbv,
                  "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0", "-threads", str(threads)]
    else:
        raise ValueError(f"unsupported codec {c}")

    p1 = _head(inp, vf, binary) + common + ["-pass", "1", "-passlogfile", passlog, "-f", "null", "-"]
    p2 = _head(inp, vf, binary) + common + ["-pass", "2", "-passlogfile", passlog] + _tail(out, pix)
    return [p1, p2]
