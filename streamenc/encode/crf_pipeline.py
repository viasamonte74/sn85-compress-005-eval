"""CRF contender pipeline: fast-preset pilot search, then one final encode with the chosen arm.

Why: SVT-AV1 constant-quality (CRF) beats 1-pass VBR by +1.4 VMAF on average (+11 on outliers) at
equal bits and is deterministic, but CRF has to be searched per clip. The CRF that lands on the
budget transfers between presets (sd ~0.3 for p3-p5), so the search runs on a cheap preset and the
expensive preset is encoded once (plus at most two corrective re-encodes).

Budget semantics follow the validator: average packet bitrate <= 1.01 x budget, 2 s peak window
<= 1.5 x budget, keyframe every <= 2 s, first frame key. We aim at AIM_UTIL of the budget and accept
[ACCEPT_LOW, ACCEPT_HIGH].
"""

from __future__ import annotations

import math
import re
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from streamenc.encode.ratefit import (CRF_MAX, CRF_MIN, CRF_QUANT, RATE_SLACK, _quant, budget_use, default_slope, local_slope,
                                      solve_crf)
from streamenc.encoders.arms import FFMPEG as FFMPEG_BIN, FFPROBE, build_commands, get_arm
from streamenc.probe import probe_video
from streamenc.probe.ffprobe_meta import gop_for_timestamps, packet_pts
from streamenc.util.logging import get_logger
from streamenc.util.subprocess_run import run_command
from streamenc.validate.vmaf import quick_vmaf_gap, relabel_vf
from streamenc.batch import state as _bstate
from streamenc.util.subprocess_run import CancelledCommand
from streamenc.validate.official_streaming import (
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)

log = get_logger("streamenc.crf_pipeline")

FINAL_ARM_DEFAULT = "svt_crf_p3_qm_la"
PILOT_ARM_DEFAULT = "svt_crf_p8_t1"
SPEED_ARM = "svt_crf_p8_t1"  # hopeless clips (no arm reaches 60 VMAF): only the speed term counts
AIM_UTIL = 0.985      # pilot search target (pilot points define the curve)
# 2026-11-11: STREAMENC_* env overrides for the budget-fill experiment only (defaults = shipped values)
FINAL_AIM = float(os.environ.get("STREAMENC_FINAL_AIM", "0.965"))  # final encode aims lower: an overshoot costs a full re-encode
ACCEPT_LOW = float(os.environ.get("STREAMENC_ACCEPT_LOW", "0.93"))  # re-encode (fill-up) only below this
ACCEPT_HIGH = 1.0 + RATE_SLACK  # avg / target cap; validator allows 1.01 (budget_use is relative to this cap)
PILOT_TOL_LOG = float(os.environ.get("STREAMENC_PILOT_TOL", "0.04"))  # pilot converged when |ln(util/pilot_aim)| < 4 %
PILOT_BRACKET_LOG = float(os.environ.get("STREAMENC_PILOT_BRACKET", "0.12"))  # ... or when points within 12 % bracket the aim (interpolated estimate)
PILOT_MAX = int(os.environ.get("STREAMENC_PILOT_MAX", "2"))  # e2e 2026-11-11: 5 -> 2 pilots, practice 118.4 -> 91.0 s, quality equal
FINAL_MAX_FIX = 2
# ---- VMAF-NEG gate (validator: |VMAF - VMAF-NEG| > 3.0 -> clip score 0). Fallback order NEG_SAFE_ARMS, derived
# 2026-11-12 on the compliant label set only (656 AI / render / screen clips, no natural footage; MINING.md training-
# data rule): gate fail rate 0.15 % for p3_t0 / p4_lqb / p8_t1 vs 0.31-0.46 % for the other arms; in the 16 primary-
# arm failures (5 clips) tune-0 passes with the widest margin (gaps 2.35-2.93 vs p4_lqb 2.88-3.00) and t0 -> p8_t1
# rescues 4 of the 5 clips (mean clip score 0 -> 0.275).
# ---- decode-once source cache: every encode attempt re-decoded the FFV1 source (1080p ~65 ms/frame,
# ~17 % of runtime CPU on e2e_test16 at 4.5 attempts/clip). CFR sources are decoded once to raw y4m
# (rawvideo in Matroska) on local disk (page cache, reclaimable: no tmpfs/RAM pinning in the 8 GB sandbox).
RAW_CACHE = os.environ.get("STREAMENC_RAW_CACHE", "1") not in ("0", "false", "no")
RAW_CACHE_MAX_BYTES = 3.0e9
STAGE2_UTIL = 0.985     # the stage-2 models' pilot-CRF feature = pilot CRF at this budget share
# 2026-11-12 prediction fix (research; default OFF until approved). Stage 2 over-predicts hard clips (held-out AI-hard
# MAE 6.6, label 50-60 -> +17: regression to the mean). Stage 3 MEASURES the two pilots' VMAF exactly (validator frame
# pairing, every frame), interpolates the pilot arm's VMAF to budget use STAGE2_UTIL and predicts each arm from it
# (models with vmaf3_models): held-out MAE 0.73 AI-hard / 0.64 render / 0.20 screen (scripts/predfix_exp2.py).
# The pilots are then encoded WITHOUT the hard prefilter (labels are unfiltered); the prefilter is decided from the
# stage-3 prediction and applied to the final encodes only.
STAGE3 = os.environ.get("STREAMENC_STAGE3", "0") not in ("0", "", "false", "no")
STAGE3_THREADS = (8, 8)  # decoder / libvmaf threads of the pilot measurement (every frame: exact)
# Only measure when stage 2 predicts a best VMAF below this: ungated, practice wall 107.0 -> 156.9 s (easy 1080p clips:
# ~90 CPU-s of measurement each, 8-11 s waits, and +-0.3 VMAF arm swaps) for +0.0008 quality. Held-out: below 94 covers
# every clip whose true best VMAF is < 90 (28 of 54 measured); stage 2 errs little above that.
STAGE3_MAX_PRED = float(os.environ.get("STREAMENC_STAGE3_MAX_PRED", "94"))
# "full": stage 3 replaces stage 2 (pilots unfiltered). "hopeless" (2026-11-12 weak-spot try): stage 3 only decides
# hopeless clips (fast p8 path, they score 0 anyway); every other decision stays stage 2 and the pilots keep the
# stage-1 prefilter, so non-hopeless clips are encoded exactly as without stage 3.
STAGE3_MODE = os.environ.get("STREAMENC_STAGE3_MODE", "full")
# 2026-11-12 early bump (default ON since 1.0.0rc5, user yes): when a join is further over the cap than the probe
# (shortest segment) can still absorb, move the other segments up at once (1-2 quarter steps sized to the overshoot)
# instead of after two futile probe trims. Held-out 4-clip batches: 172.5 -> 164.2 s/3428, final 0.6236 -> 0.6259 (the
# two slowest batches -13.6 / -15.5 s, anim_hard_007_i60_1080 same output); practice outputs identical.
SEGFINAL_EARLY_BUMP = os.environ.get("STREAMENC_SEGFINAL_EARLY_BUMP", "1") not in ("0", "", "false", "no")
# 2026-11-12 weak-spot try (research, default OFF): decide the hard prefilter per clip by measuring it. The fixed rule
# (predicted best < 80 -> cblur4_ds85) helps realistic AI clips (+1.0..+3.1 VMAF) but hurts animation-style ones
# (anim_hard_002_1080 -5.9, anim_hard_017_1080 -2.6). A p8 pair at the pilot CRF (with / without the filter, exact
# VMAF, compared at equal bits) got the sign right on 13 of 14 clips with a known real gain; deciding by it: +7.97 VMAF
# summed vs -1.02 for always filtering (scripts/filter_gain_proxy.py). Candidates: stage-2 predicted best < this.
FILTER_PROBE = os.environ.get("STREAMENC_FILTER_PROBE", "0") not in ("0", "", "false", "no")
FILTER_PROBE_MAX_PRED = float(os.environ.get("STREAMENC_FILTER_PROBE_MAX_PRED", "88"))
FILTER_PROBE_MARGIN = 0.1   # VMAF the filter must win by
# "segment" (2026-11-12, user: "start single-segment prefilter"): the pair is one 2 s keyframe segment (middle one if
# full length), encoded with / without the filter at the FIRST pilot's CRF on a background thread while the second
# pilot encodes (clips whose stage-1 predicted best < FILTER_PROBE_EARLY_MAX), compared against the same source frames.
# Offline (14 clips with a known real gain): sign 13/14, +7.97 VMAF summed, 3.8 s per probe alone (whole clip: 8.7 s in
# the pipeline, on the critical path). Used where the fixed rule would filter (stage-2 best < FILTER_PROBE_MAX_PRED).
FILTER_PROBE_MODE = os.environ.get("STREAMENC_FILTER_PROBE_MODE", "segment")
# 2026-11-12 (user: "Go ahead"): zero-cost version - a model predicts the prefilter gain from the features + pilot CRF
# (trained on offline single-segment probes of every labelled clip, scripts/train_filter_gain.py; model files with
# .fgain_model). Where the fixed rule would filter, the filter is kept only if the predicted gain > the margin.
FILTER_MODEL = os.environ.get("STREAMENC_FILTER_MODEL", "0") not in ("0", "", "false", "no")
# 2026-11-12 free-time refill (research, default OFF): easy clips aim at FINAL_AIM_EASY and the fill-up is weighed
# against lam x CPU, so they often ship at 70-89 % of the budget (test16: q04 0.742 / VMAF 96.83). A batch lasts as
# long as its slowest clip: while another clip is still running, one refill aiming at FREE_FILL_AIM costs no wall
# time. It is cancelled the moment this clip is the last one running (streamenc.batch.state), kept only if valid.
FREE_FILL = os.environ.get("STREAMENC_FREE_FILL", "0") not in ("0", "", "false", "no")
FREE_FILL_BELOW = 0.97
FREE_FILL_AIM = 0.985
# 2026-11-12 (research, default OFF): the VMAF-NEG check samples every 5th frame, which read the gap 0.28 low on a
# 60 fps clip (exact on 24/30 fps within 0.03); at or above this frame rate measure every frame (0 = never).
NEG_FULL_FPS = float(os.environ.get("STREAMENC_NEG_FULL_FPS", "0"))
FILTER_MODEL_MARGIN = float(os.environ.get("STREAMENC_FILTER_MODEL_MARGIN", "0.1"))
FILTER_PROBE_EARLY_MAX = float(os.environ.get("STREAMENC_FILTER_PROBE_EARLY_MAX", "85"))
if FILTER_PROBE_MODE == "segment" and "STREAMENC_FILTER_PROBE_MAX_PRED" not in os.environ:
    FILTER_PROBE_MAX_PRED = float(os.environ.get("STREAMENC_HARD_VMAF", "80"))  # = HARD_VMAF (defined below)


# 2026-11-13 research only (default off): "perfect prediction" oracle. JSON {input basename: VMAF shift} added to every
# arm's stage-2 prediction, so the run routes as if the model had predicted the real VMAF. Never set at run time.
PRED_SHIFT_FILE = os.environ.get("STREAMENC_PRED_SHIFT_FILE", "")
# 2026-11-13 colour fix: relabel untagged sources as BT.709 before encoding (see validate.vmaf.relabel_vf); "0" = old behaviour
COLOUR_RELABEL = os.environ.get("STREAMENC_COLOUR_RELABEL", "1") not in ("0", "", "false", "no")


def _pred_shift(input_path: str) -> float:
    if not PRED_SHIFT_FILE:
        return 0.0
    import json as _json
    try:
        return float(_json.loads(Path(PRED_SHIFT_FILE).read_text()).get(os.path.basename(input_path), 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


# 2026-11-13 clean-content tools (scripts/clean_config_search.py, results/clean_search): extra SVT-AV1 options added to
# every final / fallback encode (never to the p8 pilots, whose CRF response is a model input, nor to p8 / p12 speed
# encodes). STREAMENC_SVT_EXTRA = comma list of tool names below ("" = off). SVT 4.3 ignores the chroma-u/v-*-qindex
# options; use-fixed-qindex-offsets=2 applies only the chroma offsets (luma rate control unchanged). VMAF is luma-only.
SVT_TOOLS: dict[str, dict[str, Any]] = {
    "chq63": {"use-fixed-qindex-offsets": 2, "key-frame-chroma-qindex-offset": 63, "chroma-qindex-offsets": "63,63,63,63,63,63"},
    "chq32": {"use-fixed-qindex-offsets": 2, "key-frame-chroma-qindex-offset": 32, "chroma-qindex-offsets": "32,32,32,32,32,32"},
    "sqo8": {"startup-qp-offset": -8},
    "scm1": {"scm": 1},
    "acb1": {"ac-bias": 1.0},
    "vb2o2": {"enable-variance-boost": 1, "variance-boost-strength": 2, "variance-octile": 2},
}
# 1.0.0rc9 default "chq63": label search on render / screen / AI training clips +0.30 / +0.47 VMAF at presets 4 / 6
# (CPU x0.8-1.0, VMAF - VMAF-NEG gap unchanged); e2e practice 0.7673, test16 0.8481, hard 0.6543, held-out 0.6485 q.
SVT_EXTRA = [t for t in os.environ.get("STREAMENC_SVT_EXTRA", "chq63").split(",") if t]
# Tools for clips the content classifier (predictor .kind_model, features only) calls render / screen: on AI clips
# they gained nothing and screen-content mode costs up to 1.7x encode CPU there.
# OFF by default (rc9 tests: the classifier calls the practice renders AI, start-up QP -8 cannot fit hard 60 fps renders
# at CRF 63 and blew the rate model up at low CRF; screen-content mode cost 3-4x CPU on near-lossless screens).
SVT_EXTRA_CLEAN = [t for t in os.environ.get("STREAMENC_SVT_EXTRA_CLEAN", "").split(",") if t]
CLEAN_P_MIN = float(os.environ.get("STREAMENC_CLEAN_P_MIN", "0.5"))
SVT_EXTRA_SKIP = {"svt_crf_p8_t1", "svt_crf_p12_t1"}


def _arm(arm_id: str, extra: list[str] | None = None) -> dict[str, Any]:
    """get_arm plus the clean-content SVT tools (``extra`` overrides STREAMENC_SVT_EXTRA for this clip)."""
    arm = get_arm(arm_id)
    tools = SVT_EXTRA if extra is None else extra
    if not tools or arm_id in SVT_EXTRA_SKIP or arm.get("codec") != "libsvtav1":
        return arm
    svt = dict(arm.get("svt", {}))
    for t in tools:
        svt.update(SVT_TOOLS[t])
    return {**arm, "svt": svt}


def _pilot_vmaf_at_budget(pts: list[tuple[float, float]]) -> float:
    """Pilot-arm VMAF at budget use STAGE2_UTIL from measured (use, vmaf) pilots: linear in ln(use) through the two
    extreme pilots (one pilot, or a non-positive slope: the vmaf_per_log_bits rule), from the pilot nearest budget."""
    pts = sorted(pts)
    u, v = min(pts, key=lambda q: abs(math.log(max(q[0], 1e-6) / STAGE2_UTIL)))
    s = None
    if len(pts) >= 2 and abs(math.log(pts[-1][0] / pts[0][0])) > 0.02:
        s = (pts[-1][1] - pts[0][1]) / math.log(pts[-1][0] / pts[0][0])
    if s is None or s <= 0.0:
        s = vmaf_per_log_bits(v)
    return min(100.0, max(0.0, v + s * math.log(STAGE2_UTIL / max(u, 1e-6))))
STAGE2_SHIP_PILOT_USE = 0.85  # stage-2 hopeless: ship the valid pilot if it uses >= this share
NEG_LIMIT = 3.0
NEG_SAFE = 2.85          # subsampled estimate error <= 0.07 (vmaf.quick_vmaf_gap); margin for it
NEG_CHECK_P = 0.005      # measure when the model gives P(gap > NEG_SAFE) >= this (cost ~2-14 CPU-s)
# 1.0.0rc13: also measure the gap when the gap model cannot vouch for the shipped encode: no prediction for its arm
# (the q08-route hard arm: val16b v16bai_medium_001 shipped gap 3.25 unmeasured -> gate fail, 0 points) or a
# prefilter on it (the gap models were trained on unfiltered encodes). Unseen val16d / val16e: outputs unchanged.
NEG_CHECK_UNKNOWN = os.environ.get("STREAMENC_NEG_CHECK_UNKNOWN", "1") not in ("0", "", "false", "no")
# 1.0.0rc16 (from rc15): when no fallback reaches NEG_SAFE on the 1-in-5-frame estimate, the shipped encode is chosen by
# every-frame measurements (val16f v16fai_medium_001 shipped at a true 2.887 from an estimate of 2.852).
NEG_EXACT_PICK = os.environ.get("STREAMENC_NEG_EXACT_PICK", "1") not in ("0", "", "false", "no")
NEG_SUBSAMPLE = 5
NEG_SAFE_ARMS = ("svt_crf_p3_t0_qm_la", "svt_crf_p8_t1")
# median CRF(target arm) - CRF(source arm) at equal budget use (CRF normalised to 98.5 % use), quarter steps.
# 2026-11-11: recomputed on the compliant label set only (756 AI / render / screen clips, no natural footage;
# MINING.md training-data rule); the earlier values came from 669 clips incl. natural video.
NEG_CRF_SHIFT = {
    "svt_crf_p3_t0_qm_la": {"svt_crf_p2_qm_la": 0.0, "svt_crf_p3_qm_la": 0.25, "svt_crf_p3_qm_la_lqb": 0.0,
                            "svt_crf_p4_qm_la": 0.25, "svt_crf_p4_qm_la_lqb": -0.25, "svt_crf_p5_qm_la": 0.25,
                            "svt_crf_p6_qm_la": -0.25, "svt_crf_p8_t1": -1.75},
    "svt_crf_p8_t1": {"svt_crf_p2_qm_la": 2.0, "svt_crf_p3_qm_la": 2.75, "svt_crf_p3_qm_la_lqb": 2.0,
                      "svt_crf_p3_t0_qm_la": 1.75, "svt_crf_p4_qm_la": 3.0, "svt_crf_p4_qm_la_lqb": 2.25,
                      "svt_crf_p5_qm_la": 2.75, "svt_crf_p6_qm_la": 2.25},
}
CRF_START_DEFAULT = 45.0
OVER_CODES = ("STREAMING_BITRATE_EXCEEDED", "STREAMING_PEAK_EXCEEDED")
# Easy clips (predicted VMAF >= EASY_VMAF) gain ~1-2 VMAF per doubling of bits: aim lower so the slow
# final rarely overshoots (an overshoot costs a mandatory re-encode).
EASY_VMAF = 90.0
FINAL_AIM_EASY = float(os.environ.get("STREAMENC_FINAL_AIM_EASY", "0.92"))
# Predicted best VMAF below this: the clip scores 0 whatever we ship (gate fail == VMAF < 60 == 0),
# so encode once with the speed arm and ship it without a rate search.
DEEP_HOPELESS_VMAF = 50.0
# VMAF per CPU-second (speed term), used when the plan carries none (the shipped model carries the same 0.012).
# Re-checked 2026-11-12 on compliant data only (CC0 practice pack + 65 held-out AI / render / screen clips, 4-clip
# batches): final practice / held-out = 0.7652 / 0.6215 at 0.008, 0.7672 / 0.6260 at 0.012, 0.7614 / 0.6274 at 0.016.
DEFAULT_LAM = 0.012
# ---- hard-AI path + segment fill (2026-11-11, candidate; switches default OFF until approved for shipping)
# Hard clips (predicted best VMAF < HARD_VMAF): chroma blur + 85 % low-pass before every encode. Practice q07/q08
# (p2, corrected to full budget): +0.66 / +0.68 VMAF, NEG gap +0.07, chroma / tone gates pass.
HARD_VMAF = float(os.environ.get("STREAMENC_HARD_VMAF", "80"))
HARD_PREFILTER = os.environ.get("STREAMENC_HARD_PREFILTER", "cblur4_ds85")
# 2026-11-11 test option: on the hard path also force at least this (slower) preset, e.g. svt_crf_p2_qm_la. Practice
# q08 p2 + cblur4_ds85 60.58 / p1 61.04 vs p3_lqb 59.56 (results/user_tests/q0708_study2.jsonl). "" = off.
# Hard-path arm upgrade, since 1.0.0rc6 (user yes 2026-11-12) ON for q08-like clips only (limits below); "" = off.
HARD_ARM = os.environ.get("STREAMENC_HARD_ARM", "svt_crf_p2_qm_la_lqb_h3")
# 2026-11-12 q08 follow-up (research, default off): apply HARD_ARM only to clips whose (stage 2) predicted best VMAF
# is below this, and use this prefilter with it ("" = keep the hard prefilter). 2026-11-11: forcing p2/p1 on every
# hard-path clip raised VMAF but cost more time than it gained.
HARD_ARM_MAX_PRED = float(os.environ.get("STREAMENC_HARD_ARM_MAX_PRED", "70"))
HARD_ARM_PREFILTER = os.environ.get("STREAMENC_HARD_ARM_PREFILTER", "cblur16_ds85")
# q08-like clips only (user, 2026-11-12): the HARD_ARM gain is about constant (+0.6..+1.1 VMAF on p2) but its extra
# time scales with how expensive the clip is to encode, and it lands on the batch wall whenever the clip is the batch's
# slowest (aix_hard_019 +20 s, ai_hard_21_1080 +93 s at 1080p). Cap the cost: frame height <= this (0 = any) and mean
# pilot CPU-seconds per megapixel-frame <= this (0 = any): q08 0.0741 vs aix_hard_019 0.0881.
HARD_ARM_MAX_HEIGHT = int(os.environ.get("STREAMENC_HARD_ARM_MAX_HEIGHT", "720"))
HARD_ARM_MAX_PILOT_CPU = float(os.environ.get("STREAMENC_HARD_ARM_MAX_PILOT_CPU", "0.08"))
# e2e 4-clip batches (H2c): practice q08 60.13 -> 61.23 VMAF, quality 0.7637 -> 0.7672 (110.4 s); held-out 65 clips
# byte-identical (no clip qualifies). The 0.08 pilot-CPU limit is fitted on few clips and depends on the CPU type.
# Segment fill: CRF moves in quarter steps (~3-5 % bitrate each on hard clips), so the final encode often lands at
# 94-97.5 % of the budget. Re-encode a subset of keyframe segments (GOPs) one quarter step lower and splice them in
# (q08: whole 55.5 = 97.46 % / VMAF 59.52; segments 55.25/55.5/55.5 = 99.20 % / 60.01; segment encodes are
# bit-for-bit as efficient as the whole-clip encode).
SEGFILL = os.environ.get("STREAMENC_SEGFILL", "1") not in ("0", "", "false", "no")
SEGFILL_AIM = 0.998        # predicted budget use to stay under (ACCEPT_HIGH = 1.0, validator limit 1.01)
SEGFILL_MIN_GAIN = 0.004   # skip when the predicted use gain is below 0.4 %
SEGFILL_K = 0.048          # default ln(bits) increase per quarter CRF step (q08: ln(1.0220/0.9746))
# Segmented final (candidate, default OFF): the first whole-clip final encode lands anywhere from 0.5x to 2x of the
# budget (e2e 2026-11-11: 1/3 over budget, 1/4-1/2 under 93 %; 1.3-1.6 final encodes per clip = 51-67 % of all CPU).
# Encode the final per keyframe segment instead: the smallest segment first as a probe (measures this clip's real
# final/pilot bit ratio), the others at the corrected CRF, then the probe again as a fine trim -> about one final
# encode in total and ~99 % of the budget. Segment encodes are as efficient as a whole-clip encode.
SEGFINAL = os.environ.get("STREAMENC_SEGFINAL", "1") not in ("0", "", "false", "no")
SEGFINAL_AIM = float(os.environ.get("STREAMENC_SEGFINAL_AIM", "0.992"))
# 2026-11-13: most quarter steps one early bump may move the other segments (was fixed at 2; anim_hard_007_i60_1080
# with chroma QP needed 4 -> two extra probe encodes, +150 CPU-s). The step count is the computed need, so a larger cap
# only matters when the need is larger.
EARLY_BUMP_MAX_STEPS = int(os.environ.get("STREAMENC_EARLY_BUMP_MAX_STEPS", "4"))  # rc9: 4 (held-out 168.0 -> 159.8 s)
SEGFINAL_SPREAD = 0.75     # max CRF difference between segments of one clip
SEGFINAL_MIN_CRF = 15.0    # below this (near-lossless, mostly screen) keep the whole-clip fit
SEGFINAL_PAR = int(os.environ.get("STREAMENC_SEGFINAL_PAR", "2"))
SEGFINAL_BUMP = os.environ.get("STREAMENC_SEGFINAL_BUMP", "1") not in ("0", "", "false", "no")  # SEGFINAL only
SEGFINAL_TRIES = 6 if SEGFINAL_BUMP else 4
# 2026-11-12: when the NEG guard rejects a hard-path encode, first re-encode the same arm and CRF with this milder
# hard prefilter before switching arms (e2e: cblur16_ds85 raised hfr_rnd_hard_03 gap 2.40 -> 3.06; the arm ladder
# kept cblur16 and ended on p8, VMAF 77.4 -> 66.7). "" = off (old behaviour).
NEG_PREFILTER_FALLBACK = os.environ.get("STREAMENC_NEG_PREFILTER_FALLBACK", "")
# 2026-11-12 probe fill-down (q08 oracle, results/alloc_oracle/q08_cb4.json): segfinal only moves the probe segment
# UP when over budget and accepts the first join under the cap (q08: 56.75/56.75/57.5 = 0.9922, VMAF 59.70), while
# 56.75 x 3 = 1.0067 <= cap gives 60.13. When the accepted join is more than FILLDOWN_MIN_GAP under the cap,
# re-encode the probe (shortest segment) at the lowest quarter step predicted to stay under the cap, measure the
# join exactly, keep it only if valid. Worth-it rule: lam x the probe encode's own CPU. On by default (2026-11-12).
SEGFINAL_FILLDOWN = os.environ.get("STREAMENC_SEGFINAL_FILLDOWN", "1") not in ("0", "", "false", "no")
FILLDOWN_MIN_GAP = 0.004   # skip when the join is already within 0.4 % of the cap
FILLDOWN_MARGIN = 0.001    # predicted use must stay 0.1 % under the cap (k error; the join is measured anyway)
HOPELESS_ARM = "svt_crf_p12_t1"


def vmaf_per_log_bits(vmaf: float) -> float:
    """dVMAF / d ln(bits) near the budget, measured on paired e2e encodes (same clip+arm, two rates):
    19-23 at VMAF 63-66, 11.7 at 77, 4.6 at 90, 1-2 at 93-96, ~0 at 98-100 -> 0.55 x (100 - VMAF)."""
    return min(25.0, max(0.0, 0.55 * (100.0 - vmaf)))


@dataclass
class Plan:
    """What to encode: produced by a predictor (model or rules) from features + geometry."""

    final_arm: str = FINAL_ARM_DEFAULT
    pilot_arm: str = PILOT_ARM_DEFAULT
    crf_start: float = CRF_START_DEFAULT
    # ln(bits(final_arm) / bits(pilot_arm)) at equal CRF (learned from labels; negative = final arm
    # spends fewer bits at the same CRF, so it can run at a lower CRF). Constant across CRF levels,
    # unlike a CRF shift (measured -0.22..-0.25 for p8_t1 -> p3_qm_la on two hard clips).
    rate_ratio_log: float = 0.0
    crf_offset: float = 0.0  # extra additive CRF shift (normally 0)
    hopeless: bool = False
    fill_up: bool = True  # allow a fill-up re-encode when its expected VMAF gain beats its CPU cost
    pred_vmaf: float | None = None   # predicted VMAF of the final arm at budget
    pred_best: float | None = None   # predicted best VMAF over arms
    pred_cpu: float | None = None    # predicted CPU seconds of one final encode
    lam: float = DEFAULT_LAM
    notes: str = ""


@dataclass
class CrfOutcome:
    ok: bool
    config_id: str
    crf: float | None
    util: float | None
    avg_kbps: float | None
    peak_kbps: float | None
    attempts: int
    pilot_attempts: int
    encode_seconds: float
    cpu_seconds: float
    frames: int
    target_kbps: float
    reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _measure(out: str, *, fps: float, frames: int, target: float, max_keyframe_interval: float,
             peak_ratio: float, peak_window: float) -> tuple[float, list[str], Any]:
    m = measure_streaming(read_packets_ffprobe(out, ffprobe=FFPROBE), fps=fps, frames=frames,
                          peak_window_seconds=peak_window)
    vs = streaming_violations(m, target_kbps=target, peak_ratio=peak_ratio, peak_window_seconds=peak_window,
                              max_keyframe_interval_seconds=max_keyframe_interval, fps=fps)
    return m.avg_kbps / target, sorted({v.code for v in vs}), m


def _preset(arm_id: str) -> int:
    m = re.search(r"_p(\d+)", arm_id)
    return int(m.group(1)) if m else 99


def _bracket_done(points: list[tuple[float, float, bool]]) -> bool:
    """True when an over-budget CRF and a valid CRF are adjacent quarter steps (nothing left between)."""
    hi_over = max((c for c, _, o in points if o), default=None)
    lo_ok = min((c for c, _, o in points if not o), default=None)
    return hi_over is not None and lo_ok is not None and 0 < lo_ok - hi_over <= CRF_QUANT + 1e-9


_RAW_LOCK = threading.Lock()
_RAW_RESERVED = [0.0]   # bytes of raw caches being written right now (all clips of the batch)


def _raw_source(input_path: str, meta: Any, pts: list[float]) -> tuple[str | None, str | None, float]:
    """Decode ``input_path`` once to rawvideo-in-Matroska in a local temp dir: keeps the source's
    timestamps and colour tags, so every encode from it is byte-identical to encoding the source
    (verified on untagged / BT.2020 / BT.709 clips) at ~0 decode cost. (A y4m drops the colour tags,
    and ffmpeg 8 then auto-converts unknown -> bt709 at encode time.) Returns (path, dir, cpu_s)."""
    if not RAW_CACHE or (pts and len(pts) != meta.nb_frames):
        return None, None, 0.0
    bpp = 3.0 if any(b in (meta.pix_fmt or "") for b in ("10", "12", "16")) else 1.5
    if "444" in (meta.pix_fmt or ""):
        bpp *= 2.0
    elif "422" in (meta.pix_fmt or ""):
        bpp *= 4.0 / 3.0
    raw_bytes = meta.width * meta.height * bpp * meta.nb_frames
    base = tempfile.gettempdir()
    # 1.0.0rc16 (from rc15): the batch's clips run as threads and each checked the free space on its own, so on a small
    # /tmp they could all pass and then fill it together; caches still being written are now reserved under a lock.
    with _RAW_LOCK:
        try:
            free = shutil.disk_usage(base).free - _RAW_RESERVED[0]
        except OSError:
            return None, None, 0.0
        if raw_bytes > RAW_CACHE_MAX_BYTES or free < 2.0 * raw_bytes + 1e9:
            return None, None, 0.0
        _RAW_RESERVED[0] += raw_bytes
    try:
        d = tempfile.mkdtemp(prefix="streamenc-raw-", dir=base)
        raw = os.path.join(d, "src.mkv")
        r = run_command([FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", input_path,
                         "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "rawvideo", "-f", "matroska", raw], timeout=1800)
    finally:
        with _RAW_LOCK:
            _RAW_RESERVED[0] -= raw_bytes
    ok = r.ok and os.path.exists(raw) and os.path.getsize(raw) >= 0.98 * raw_bytes
    if not ok:
        shutil.rmtree(d, ignore_errors=True)
        return None, None, float(r.cpu_seconds)
    return raw, d, float(r.cpu_seconds)


# 1.0.0rc16 raw-cache prefetch: probe, packet timestamps and the raw cache of the source (the first steps of
# encode_item_crf, same calls and same results) start in a background thread when the clip starts, so they overlap the
# feature extraction and model wait instead of following them (sequential: 1.5-2.7 s + 0.4-2.8 s per clip).
RAW_PREFETCH = os.environ.get("STREAMENC_RAW_PREFETCH", "1") not in ("0", "", "false", "no")


def _prefetch_source(input_path: str) -> tuple[Any, list[float], tuple[str | None, str | None, float], float, float]:
    t0 = time.perf_counter()
    meta = probe_video(input_path, count_frames=True)
    try:
        pts = packet_pts(input_path)
    except Exception:  # noqa: BLE001
        pts = []
    t1 = time.perf_counter()
    raw = _raw_source(input_path, meta, pts)
    return meta, pts, raw, t1 - t0, time.perf_counter() - t1


def encode_item_crf(
    input_path: str,
    output_path: str,
    plan: Plan,
    *,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    peak_window: float = 2.0,
    peak_ratio: float = 1.5,
    time_budget_seconds: float | None = None,
    threads: int | None = None,
    prefetch: Any = None,
) -> CrfOutcome:
    t_start = time.perf_counter()
    tm: dict[str, float] = {}   # wall seconds of the stages no encode entry covers (logging only)
    pre = None
    if prefetch is not None:
        try:
            pre = prefetch.result()
        except Exception:  # noqa: BLE001 - prefetch failed: do the same steps here, as without it
            pre = None
    if pre is not None:
        tm["prefetch_wait"] = time.perf_counter() - t_start
    meta = pre[0] if pre is not None else probe_video(input_path, count_frames=True)
    # untagged source: relabel (as the validator does to the reference), never let ffmpeg convert it
    relabel = relabel_vf(meta.pix_fmt, meta.color_space, meta.color_range) if COLOUR_RELABEL else None
    clean_p = getattr(plan, "clean_p", None)
    clip_extra: list[str] = list(SVT_EXTRA) + [t for t in SVT_EXTRA_CLEAN if clean_p is not None and clean_p >= CLEAN_P_MIN
                                               and t not in SVT_EXTRA]
    target = streaming_target_kbps(width=meta.width, height=meta.height, fps=meta.fps, bits_per_pixel=bits_per_pixel)
    if pre is not None:
        pts = pre[1]
    else:
        try:
            pts = packet_pts(input_path)
        except Exception:  # noqa: BLE001
            pts = []
    gop = gop_for_timestamps(pts, meta.fps, max_keyframe_interval) if pts else \
        max(1, int(max_keyframe_interval * meta.fps + 1e-9))
    tm["probe"] = time.perf_counter() - t_start
    out_dir = Path(output_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".crf-", dir=out_dir))
    hist: list[dict[str, Any]] = []
    if SVT_EXTRA or SVT_EXTRA_CLEAN:
        hist.append({"tag": "svt_extra", "tools": clip_extra, "clean_p": None if clean_p is None else round(clean_p, 3)})
    bg_jobs: list = []    # background futures (segment filter probe): waited for before the work dir is removed
    cpu_total = 0.0
    src = input_path      # encoder/VMAF source: the raw cache once made
    raw_dir: str | None = pre[2][1] if pre is not None else None   # removed in the finally below whatever happens
    enc_seconds = 0.0
    attempts = 0
    kw = dict(fps=meta.fps, frames=meta.nb_frames, target=target, max_keyframe_interval=max_keyframe_interval,
              peak_ratio=peak_ratio, peak_window=peak_window)

    def _remaining() -> float:
        if time_budget_seconds is None:
            return float("inf")
        return time_budget_seconds - (time.perf_counter() - t_start)

    def _encode(arm_id: str, crf: float, tag: str, cancel=None) -> tuple[str | None, float | None, list[str], Any]:
        nonlocal cpu_total, enc_seconds, attempts
        attempts += 1
        arm = _arm(arm_id, clip_extra)
        out = str(work / f"{tag}{attempts}.mp4")
        t0 = time.perf_counter()
        cpu = 0.0
        err = ""
        for cmd in build_commands(arm, src, out, target, gop, str(work / "pass"), hard_vf,
                                  **({"threads": threads} if threads else {}), crf=crf, pre_vf=relabel):
            try:
                r = run_command(cmd, timeout=3600, cancel=cancel)
            except CancelledCommand:  # free-time work stopped: this clip is now the batch's last
                hist.append({"tag": tag, "arm": arm_id, "crf": crf, "cancelled": True,
                             "seconds": round(time.perf_counter() - t0, 3)})
                return None, None, ["CANCELLED"], None
            cpu += r.cpu_seconds
            if not r.ok:
                err = (r.stderr or r.stdout or "")[-300:].replace("\n", " ")
                break
        dt = time.perf_counter() - t0
        enc_seconds += dt
        cpu_total += cpu
        if err or not os.path.exists(out):
            hist.append({"tag": tag, "arm": arm_id, "crf": crf, "error": err or "no output", "seconds": round(dt, 3)})
            return None, None, ["ENCODE_FAILED"], None
        t1 = time.perf_counter()
        util, codes, m = _measure(out, **kw)
        tm["measure"] = tm.get("measure", 0.0) + time.perf_counter() - t1
        use = budget_use(m.avg_kbps, m.peak_kbps, target, peak_ratio)
        hist.append({"tag": tag, "arm": arm_id, "crf": crf, "util": round(util, 4), "use": round(use, 4),
                     "codes": codes, "seconds": round(dt, 3), "cpu_seconds": round(cpu, 3)})
        return out, util, codes, m

    def _hard_vf(stage: str) -> str | None:
        """Hard-AI prefilter when the (stage 1 or stage 2) predicted best VMAF is below HARD_VMAF."""
        if not HARD_PREFILTER or plan.hopeless or plan.pred_best is None or plan.pred_best >= HARD_VMAF:
            return None
        try:
            from streamenc.filters.policy import filter_vf
            v = filter_vf(HARD_PREFILTER)
            hist.append({"tag": "hard_prefilter", "filter": HARD_PREFILTER, "stage": stage,
                         "pred_best": round(plan.pred_best, 2)})
            return v.format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction)) if v else None
        except Exception as exc:  # noqa: BLE001
            hist.append({"tag": "hard_prefilter", "error": str(exc)[-120:]})
            return None

    hard_vf: str | None = None if (STAGE3 and STAGE3_MODE == "full") else _hard_vf("stage1")

    def _seg_filter_probe(crf_: float) -> dict:
        """One keyframe segment with / without HARD_PREFILTER on the pilot arm at ``crf_``, exact VMAF against the same
        source frames, compared at equal bits (runs on a background thread during the second pilot)."""
        from concurrent.futures import ThreadPoolExecutor
        from streamenc.filters.policy import filter_vf as _fvf
        t0_ = time.perf_counter()
        bounds_ = [(a_, min(a_ + gop, meta.nb_frames)) for a_ in range(0, meta.nb_frames, gop)]
        sa, sb = bounds_[len(bounds_) // 2]
        if sb - sa != gop:
            sa, sb = bounds_[0]
        fvf = _fvf(HARD_PREFILTER).format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction))
        arm_ = get_arm(plan.pilot_arm)

        def _one(filt: bool):
            vf_ = f"trim=start_frame={sa}:end_frame={sb},setpts=PTS-STARTPTS" + (f",{fvf}" if filt else "")
            o_ = str(work / f"fseg{int(filt)}.mp4")
            cpu_ = 0.0
            for cmd in build_commands(arm_, src, o_, target, gop, str(work / f"fsegpass{int(filt)}"), vf_,
                                      **({"threads": threads} if threads else {}), crf=crf_, pre_vf=relabel):
                r_ = run_command(cmd, timeout=900)
                cpu_ += r_.cpu_seconds
                if not r_.ok:
                    raise RuntimeError((r_.stderr or "")[-200:])
            by_ = sum(p_.size_bytes for p_ in read_packets_ffprobe(o_, ffprobe=FFPROBE))
            q_ = quick_vmaf_gap(src, o_, subsample=1, reference_meta=meta, threads=STAGE3_THREADS, exact_pairing=True,
                                ref_trim=(sa, sb))
            return by_, q_.vmaf, cpu_ + q_.cpu_seconds
        with ThreadPoolExecutor(2) as ex_:
            (bu_, vu_, cu_), (bf_, vf2_, cf_) = list(ex_.map(_one, [False, True]))
        gain_ = vf2_ + vmaf_per_log_bits(vf2_) * math.log(max(bu_, 1) / max(bf_, 1)) - vu_
        return {"gain": gain_, "vf": fvf, "crf": crf_, "seg": (sa, sb), "cpu": cu_ + cf_,
                "seconds": round(time.perf_counter() - t0_, 3)}
    final_arm = SPEED_ARM if plan.hopeless else plan.final_arm
    easy = plan.pred_vmaf is not None and plan.pred_vmaf >= EASY_VMAF and not plan.hopeless
    final_aim = FINAL_AIM_EASY if easy else FINAL_AIM
    ratio_fn = getattr(plan, "ratio_fn", None)

    def _ratio(crf: float) -> float:
        if ratio_fn is None:
            return plan.rate_ratio_log
        try:
            return float(ratio_fn(crf))
        except Exception:  # noqa: BLE001
            return plan.rate_ratio_log

    sf_calls = {"n": 0}  # separate files per call (the NEG prefilter fallback runs it a second time)
    sf_last: dict[str, Any] = {}  # last successful segmented final: segment files / CRFs (+ segfill splices)

    def _segmented_final(arm_id: str, crf_start: float) -> tuple | None:
        """Final encode per keyframe segment with closed-loop rate control. Returns (out, util, m, crf) or None
        (caller falls back to the whole-clip fit)."""
        nonlocal cpu_total, enc_seconds, attempts
        from fractions import Fraction
        n = meta.nb_frames
        bounds = [(a, min(a + gop, n)) for a in range(0, n, gop)]
        arm = _arm(arm_id, clip_extra)
        if len(bounds) < 2 or arm.get("rc") != "crf" or not pts or len(pts) != n:
            return None
        # e2e 2026-11-11: all 6 failures were near-lossless clips (CRF 2-12, screen content) whose limit is the 2 s
        # peak, which a per-segment trim cannot fix -> keep the whole-clip fit there.
        if crf_start < SEGFINAL_MIN_CRF:
            return None
        if best_pilot is not None and best_pilot[3].peak_kbps / peak_ratio > 1.02 * best_pilot[3].avg_kbps:
            return None
        step = [b - a for a, b in zip(pts, pts[1:])]
        if step and (max(step) - min(step)) > 0.25 / meta.fps:
            return None  # variable frame rate: keep the whole-clip path
        sf_calls["n"] += 1
        segdir = work / ("sf" if sf_calls["n"] == 1 else f"sf{sf_calls['n']}")
        segdir.mkdir(exist_ok=True)
        ns = len(bounds)
        # per-segment complexity weights from the last pilot encode (split at its keyframes), else frame counts
        w = [float(b - a) for a, b in bounds]
        pil = sorted(work.glob("pilot*.mp4"), key=lambda q: int("".join(ch for ch in q.stem if ch.isdigit()) or 0))
        if pil:
            r = run_command([FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", str(pil[-1]), "-map", "0:v:0",
                             "-c", "copy", "-f", "segment", "-segment_format", "ivf", "-segment_frames",
                             ",".join(str(a) for a, _ in bounds[1:]), "-reset_timestamps", "1", str(segdir / "p%03d.ivf")], timeout=300)
            cpu_total += r.cpu_seconds
            pp = [segdir / f"p{i:03d}.ivf" for i in range(ns)]
            if r.ok and all(q.exists() for q in pp):
                w = [float(q.stat().st_size) for q in pp]
        wsum = sum(w)
        budget = SEGFINAL_AIM * (1.0 + RATE_SLACK) * target * 1000.0 / 8.0 * (n / meta.fps)  # bytes of video packets
        k = pilot_slope if pilot_slope else default_slope(crf_start)             # ln(bits) per CRF unit
        k = min(0.6, max(0.03, k))
        t0 = time.perf_counter()
        seg_cpu = 0.0
        enc: dict[int, tuple[Path, float, int]] = {}                            # segment -> (file, crf, bytes)
        enc_cpu: dict[int, float] = {}                                          # segment -> CPU-s of its last encode

        def _enc(i: int, crf_: float) -> int | None:
            nonlocal seg_cpu
            a, b = bounds[i]
            vf = f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS" + (f",{hard_vf}" if hard_vf else "")
            o = segdir / f"s{i:03d}_{int(round(crf_ * 100))}.ivf"
            for cmd in build_commands(arm, src, str(o), target, gop, str(segdir / f"pass{i}"), vf,
                                      **({"threads": threads} if threads else {}), crf=crf_, pre_vf=relabel):
                cmd[len(cmd) - 1 - cmd[::-1].index("-f") + 1] = "ivf"
                rr = run_command(cmd, timeout=3600)
                seg_cpu += rr.cpu_seconds
                enc_cpu[i] = rr.cpu_seconds
                if not rr.ok or not o.exists():
                    hist.append({"tag": "segfinal", "error": (rr.stderr or "")[-160:]})
                    return None
            enc[i] = (o, crf_, o.stat().st_size)
            return enc[i][2]

        def _cq(x: float, up: bool) -> float:
            q = math.ceil(x / CRF_QUANT - 1e-9) * CRF_QUANT if up else math.floor(x / CRF_QUANT + 1e-9) * CRF_QUANT
            return max(CRF_MIN, min(CRF_MAX, q))

        def _finish(ok: bool):
            nonlocal cpu_total, enc_seconds
            cpu_total += seg_cpu
            enc_seconds += time.perf_counter() - t0

        probe = min(range(ns), key=lambda i: (bounds[i][1] - bounds[i][0], -i))
        c0 = _quant(max(CRF_MIN, min(CRF_MAX, crf_start)))
        b = _enc(probe, c0)
        if b is None:
            _finish(False)
            return None
        # clip scale: bytes per unit weight at CRF c0, from the probe; common CRF that fills the budget
        scale = b / w[probe]
        x = c0 + math.log(max(scale * wsum / budget, 1e-6)) / k
        xq = _cq(x, True)
        order = sorted((i for i in range(ns) if i != probe), key=lambda i: -w[i])
        # encode the other segments SEGFINAL_PAR at a time (independent encodes; sequential short segments left
        # cores idle: e2e wall +4-10 % although CPU-seconds fell), re-planning between waves
        from concurrent.futures import ThreadPoolExecutor
        waves = [order[q:q + SEGFINAL_PAR] for q in range(0, len(order), SEGFINAL_PAR)]
        pos = -1
        for wave in waves:
            with ThreadPoolExecutor(len(wave)) as ex:
                got = list(ex.map(lambda i_: _enc(i_, xq), wave))
            if any(g is None for g in got):
                _finish(False)
                return None
            pos += len(wave)
            i = wave[-1]
            # re-plan the remaining segments from what the encoded ones actually cost at xq
            done = [j for j in order[:pos + 1]]
            sc = sum(enc[j][2] for j in done) / sum(w[j] for j in done)          # bytes per weight at xq (measured)
            rest = [j for j in order[pos + 1:]]
            if rest:
                left = budget - sum(enc[j][2] for j in done) - sc * w[probe]
                need = sc * sum(w[j] for j in rest)
                xr = xq + math.log(max(need / max(left, 1.0), 1e-6)) / k
                xq_new = _cq(xr, True)
                xq = max(xq - SEGFINAL_SPREAD, min(xq + SEGFINAL_SPREAD, xq_new))
        # probe segment last, as the fine trim: lowest CRF that keeps the total inside the budget
        others = sum(enc[j][2] for j in order)
        base_crf = sorted(enc[j][1] for j in order)[len(order) // 2]
        sc = others / max(sum(w[j] for j in order), 1e-9)
        left = budget - others
        xp = enc[order[-1]][1] + math.log(max(sc * w[probe] / max(left, 1.0), 1e-6)) / k
        xp = max(base_crf - SEGFINAL_SPREAD, min(base_crf + SEGFINAL_SPREAD, _cq(xp, True)))

        def _join(tag: str):
            nonlocal cpu_total
            lst = segdir / f"{tag}.txt"
            lst.write_text("".join(f"file '{enc[i][0]}'\n" for i in range(ns)))
            o = str(work / (f"segfinal_{tag}.mp4" if sf_calls["n"] == 1 else f"segfinal{sf_calls['n']}_{tag}.mp4"))
            fr = Fraction(str(meta.fps_fraction)) if getattr(meta, "fps_fraction", None) else Fraction(meta.fps).limit_denominator(1001)
            rc = run_command([FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                              "-map", "0:v:0", "-c", "copy", "-video_track_timescale", str(fr.numerator), "-f", "mp4", o], timeout=600)
            cpu_total += rc.cpu_seconds
            if not rc.ok or not os.path.exists(o):
                return None
            util, codes, m = _measure(o, **kw)
            if len(read_packets_ffprobe(o, ffprobe=FFPROBE)) != n:
                codes = sorted(set(codes) | {"FRAME_COUNT"})
            return o, util, codes, m

        result = None
        tries = []
        bumps = 0
        for attempt in range(SEGFINAL_TRIES):
            if abs(enc[probe][1] - xp) > 1e-6 and _enc(probe, xp) is None:
                break
            j = _join(f"a{attempt}")
            if j is None:
                break
            o, util, codes, m = j
            use = budget_use(m.avg_kbps, m.peak_kbps, target, peak_ratio)
            tries.append((round(xp, 2), round(use, 4)))
            if not codes and util <= ACCEPT_HIGH:
                result = (o, util, m, base_crf)
                break
            if "STREAMING_KEYFRAME_INTERVAL" in codes or "FRAME_COUNT" in codes:
                break
            stp = CRF_QUANT * (2 if use > 1.03 else 1)
            xp_next = _cq(xp + stp, True)                                        # over budget: probe one step up
            # 2026-11-11: the probe is the shortest segment, so trimming it moves the total by only ~0.3-0.7 % per
            # step; after two failed joins move the other segments up one step instead (not at the spread limit:
            # e2e 2026-11-11 that fired after one try on practice q07 and dropped its budget use 0.997 -> 0.979)
            # (practice q07/q08 on p2: 4 tries ending 1.0035 / 1.0097 -> whole-clip refit, 2x wall, budget 0.92)
            early = False
            if SEGFINAL_EARLY_BUMP and SEGFINAL_BUMP and bumps < 2 and use < 1.10 and len(tries) >= 1 + 2 * bumps:
                tot_b = sum(enc[i_][2] for i_ in range(ns))
                p_sh = enc[probe][2] / max(tot_b, 1.0)                          # probe's byte share
                room = max(0.0, base_crf + SEGFINAL_SPREAD - xp)                 # CRF the probe can still rise
                cap = p_sh * (1.0 - math.exp(-k * room))                         # its max relative cut
                need = 1.0 - 1.0 / max(use, 1e-9)                               # relative cut to reach the caps
                if need > 0.7 * cap:
                    early = True
                    per_step = max((1.0 - p_sh) * (1.0 - math.exp(-k * CRF_QUANT)), 1e-6)  # cut per step of the others
                    nstep = max(1, min(EARLY_BUMP_MAX_STEPS, math.ceil(max(need - 0.5 * cap, 0.0) / per_step)))
                    stp = max(stp, CRF_QUANT * nstep)
                    hist.append({"tag": "early_bump", "use": round(use, 4), "need": round(need, 4),
                                 "probe_cap": round(cap, 4), "step": stp})
            if early or (SEGFINAL_BUMP and bumps < 2 and use < 1.10 and len(tries) >= 2 + 2 * bumps):
                bumps += 1
                for wave in [order[q:q + SEGFINAL_PAR] for q in range(0, len(order), SEGFINAL_PAR)]:
                    with ThreadPoolExecutor(len(wave)) as ex:
                        got = list(ex.map(lambda i_: _enc(i_, _cq(enc[i_][1] + stp, True)), wave))
                    if any(g is None for g in got):
                        break
                else:
                    others = sum(enc[j_][2] for j_ in order)
                    base_crf = sorted(enc[j_][1] for j_ in order)[len(order) // 2]
                    sc = others / max(sum(w[j_] for j_ in order), 1e-9)
                    left = budget - others
                    xp = enc[order[-1]][1] + math.log(max(sc * w[probe] / max(left, 1.0), 1e-6)) / k
                    xp = max(base_crf - SEGFINAL_SPREAD, min(base_crf + SEGFINAL_SPREAD, _cq(xp, True)))
                    continue
                break
            xp = xp_next
            if xp > base_crf + SEGFINAL_SPREAD + 1.0:
                break
        if result is not None:
            # state for _probe_fill_down (runs after segment fill, so it only uses budget segfill left unused)
            sf_last.clear()
            sf_last.update({"arm": arm_id, "enc": enc, "enc_cpu": enc_cpu, "probe": probe, "order": order,
                            "base_crf": base_crf, "k": k, "enc_fn": _enc, "join_fn": _join, "out": result[0]})
        _finish(result is not None)
        attempts += 1
        hist.append({"tag": "segfinal", "arm": arm_id, "ok": result is not None, "crf": base_crf, "crf_start": c0,
                     "crf_probe": round(enc[probe][1], 2), "segments": ns, "tries": tries, "k": round(k, 3),
                     "use": tries[-1][1] if tries else None, "bumps": bumps, "cpu_seconds": round(seg_cpu, 3),
                     "seconds": round(time.perf_counter() - t0, 3)})
        return result

    def _segment_fill(best: tuple, arm_id: str, *, neg_checked: bool) -> tuple:
        """Splice lower-CRF keyframe segments into the final encode so it lands nearer the budget."""
        nonlocal cpu_total, enc_seconds
        from fractions import Fraction
        from itertools import combinations
        out0, util0, m0, crf0 = best
        n = meta.nb_frames
        use0 = budget_use(m0.avg_kbps, m0.peak_kbps, target, peak_ratio)
        bounds = [(a, min(a + gop, n)) for a in range(0, n, gop)]
        if len(bounds) < 2 or use0 >= SEGFILL_AIM - SEGFILL_MIN_GAIN or crf0 - CRF_QUANT < CRF_MIN:
            return best
        if pts and len(pts) == n:  # constant frame rate only: segments are re-timed by frame count
            step = [b - a for a, b in zip(pts, pts[1:])]
            if step and (max(step) - min(step)) > 0.25 / meta.fps:
                hist.append({"tag": "segfill", "skip": "vfr"})
                return best
        # quarter-step rate ratio: measured when the fit tried crf0 - 0.25, else the default
        k = SEGFILL_K
        for h in hist:
            if h.get("arm") == arm_id and h.get("crf") is not None and abs(h["crf"] - (crf0 - CRF_QUANT)) < 1e-6 \
                    and h.get("use") and use0 > 0:
                k = max(0.01, math.log(h["use"] / use0))
        # split the final encode at its keyframes (packet copy), measure each segment's bytes
        segdir = work / "seg"
        segdir.mkdir(exist_ok=True)
        cuts = ",".join(str(a) for a, _ in bounds[1:])
        r = run_command([FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", out0, "-map", "0:v:0",
                         "-c", "copy", "-f", "segment", "-segment_format", "ivf", "-segment_frames", cuts,
                         "-reset_timestamps", "1", str(segdir / "hi%03d.ivf")], timeout=600)
        cpu_total += r.cpu_seconds
        hi = [segdir / f"hi{i:03d}.ivf" for i in range(len(bounds))]
        if not r.ok or not all(p.exists() for p in hi):
            hist.append({"tag": "segfill", "skip": "split failed"})
            return best
        size = [p.stat().st_size for p in hi]
        tot = float(sum(size))
        # choose the subset of segments to lower: predicted use closest to SEGFILL_AIM from below
        cand = []
        for r_ in range(1, len(bounds) + 1):
            for S in combinations(range(len(bounds)), r_):
                add = sum(size[i] for i in S) * (math.exp(k) - 1.0)
                cand.append((use0 * (tot + add) / tot, S))
        ok = [c for c in cand if c[0] <= SEGFILL_AIM]
        if not ok:
            hist.append({"tag": "segfill", "skip": "no subset fits", "k": round(k, 4)})
            return best
        pred_use, S = max(ok, key=lambda c: (c[0], -len(c[1])))
        if pred_use - use0 < SEGFILL_MIN_GAIN:
            return best
        last_cpu = next((h.get("cpu_seconds", 0.0) for h in reversed(hist) if h.get("arm") == arm_id and "crf" in h), 0.0)
        frac = sum(bounds[i][1] - bounds[i][0] for i in S) / n
        v = plan.pred_vmaf if plan.pred_vmaf is not None else 70.0
        gain = vmaf_per_log_bits(v) * math.log(pred_use / use0)
        cost = (plan.lam or DEFAULT_LAM) * last_cpu * frac
        if gain <= cost:
            hist.append({"tag": "segfill", "skip": "not worth", "gain": round(gain, 3), "cost": round(cost, 3)})
            return best
        if _remaining() < 1.5 * last_cpu * frac / max(1, threads or 8) + 5.0:
            hist.append({"tag": "segfill", "skip": "time"})
            return best
        # encode the chosen segments one quarter step lower (same arm, same prefilter, keyframe first)
        arm = _arm(arm_id, clip_extra)
        lo: dict[int, Path] = {}
        t0 = time.perf_counter()
        seg_cpu = 0.0
        for i in S:
            a, b = bounds[i]
            vf = f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS" + (f",{hard_vf}" if hard_vf else "")
            o = segdir / f"lo{i:03d}.ivf"
            cmds = build_commands(arm, src, str(o), target, gop, str(segdir / f"pass{i}"), vf,
                                  **({"threads": threads} if threads else {}), crf=crf0 - CRF_QUANT, pre_vf=relabel)
            for cmd in cmds:
                j = len(cmd) - 1 - cmd[::-1].index("-f")
                cmd[j + 1] = "ivf"
                rr = run_command(cmd, timeout=3600)
                seg_cpu += rr.cpu_seconds
                if not rr.ok:
                    hist.append({"tag": "segfill", "error": (rr.stderr or "")[-200:]})
                    cpu_total += seg_cpu
                    return best
            lo[i] = o
        enc_seconds += time.perf_counter() - t0
        cpu_total += seg_cpu
        # exact sizes now known for both versions: pick the best valid combination without further encodes
        fr = Fraction(str(meta.fps_fraction)) if getattr(meta, "fps_fraction", None) else Fraction(meta.fps).limit_denominator(1001)
        best_new = None
        for r_ in range(len(S), 0, -1):
            for T in combinations(S, r_):
                est = use0 * (tot + sum(lo[i].stat().st_size - size[i] for i in T)) / tot
                if est > SEGFILL_AIM + 0.002:
                    continue
                lst = segdir / f"list_{'_'.join(map(str, T))}.txt"
                lst.write_text("".join(f"file '{(lo[i] if i in T else hi[i])}'\n" for i in range(len(bounds))))
                o = str(segdir / f"filled_{'_'.join(map(str, T))}.mp4")
                rc = run_command([FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                                  "-i", str(lst), "-map", "0:v:0", "-c", "copy", "-video_track_timescale", str(fr.numerator),
                                  "-f", "mp4", o], timeout=600)
                cpu_total += rc.cpu_seconds
                if not rc.ok or not os.path.exists(o):
                    continue
                util, codes, m = _measure(o, **kw)
                nfr = len(read_packets_ffprobe(o, ffprobe=FFPROBE))
                use = budget_use(m.avg_kbps, m.peak_kbps, target, peak_ratio)
                if codes or nfr != n or util > ACCEPT_HIGH:
                    continue
                if best_new is None or use > best_new[0]:
                    best_new = (use, o, util, m, T)
            if best_new is not None:
                break
        if best_new is None:
            hist.append({"tag": "segfill", "skip": "no valid splice", "segments": list(S), "cpu_seconds": round(seg_cpu, 3)})
            return best
        use, o, util, m, T = best_new
        if neg_checked:
            g = _gap(o, arm_id)
            if g is None or g > NEG_SAFE:
                hist.append({"tag": "segfill", "skip": "neg gap", "gap": g})
                return best
        hist.append({"tag": "segfill", "crf_lo": crf0 - CRF_QUANT, "segments": list(T), "chosen": list(S), "use_before": round(use0, 4),
                     "use": round(use, 4), "pred_use": round(pred_use, 4), "k": round(k, 4),
                     "cpu_seconds": round(seg_cpu, 3), "seconds": round(time.perf_counter() - t0, 3)})
        if sf_last and sf_last.get("out") == out0 and len(sf_last["enc"]) == len(bounds):
            for i in T:
                sf_last["enc"][i] = (lo[i], crf0 - CRF_QUANT, lo[i].stat().st_size)
            sf_last["out"] = o
        return (o, util, m, crf0)

    def _probe_fill_down(best: tuple, arm_id: str, *, neg_checked: bool) -> tuple:
        """2026-11-12 (q08 oracle): re-encode the probe (shortest) segment of the segmented final at the lowest
        quarter step predicted to stay under the cap, after segment fill, so it only spends budget left unused."""
        nonlocal cpu_total, enc_seconds
        st = sf_last
        if not (SEGFINAL_FILLDOWN and st and st["arm"] == arm_id and best[0] == st["out"]):
            return best
        enc, probe, order, k = st["enc"], st["probe"], st["order"], st["k"]
        util_ = best[1]
        rec: dict[str, Any] = {"tag": "filldown", "from_crf": round(enc[probe][1], 2), "from_util": round(util_, 4)}
        if util_ >= ACCEPT_HIGH - FILLDOWN_MIN_GAP:
            hist.append({**rec, "skip": "near cap"})
            return best
        others_b = sum(enc[j][2] for j in order)
        pb, pc = enc[probe][2], enc[probe][1]
        lo_lim = max(CRF_MIN, st["base_crf"] - SEGFINAL_SPREAD)
        cands: list[tuple[float, float]] = []
        x_ = pc
        while True:
            x2 = _quant(x_ - CRF_QUANT)
            if x2 < lo_lim - 1e-9:
                break
            pred = util_ * (others_b + pb * math.exp(k * (pc - x2))) / (others_b + pb)
            if pred > ACCEPT_HIGH - FILLDOWN_MARGIN:
                break
            cands.append((x2, pred))
            x_ = x2
        if not cands:
            hist.append({**rec, "skip": "no step fits"})
            return best
        x_t, pred_t = cands[-1]
        v_ = plan.pred_vmaf if plan.pred_vmaf is not None else 70.0
        gain = vmaf_per_log_bits(v_) * math.log(pred_t / util_)
        pcpu = st["enc_cpu"].get(probe, 0.0)
        cost = (plan.lam or DEFAULT_LAM) * pcpu
        rec.update({"to_crf": x_t, "pred_util": round(pred_t, 4), "gain": round(gain, 3), "cost": round(cost, 3)})
        if gain <= cost:
            hist.append({**rec, "skip": "not worth"})
            return best
        if _remaining() < 3.0 * max(pcpu / max(1, threads or 8), 1.0) + 5.0:
            hist.append({**rec, "skip": "time"})
            return best
        prev = enc[probe]
        t0 = time.perf_counter()
        for x_try in (x_t, _quant(x_t + CRF_QUANT)):
            if x_try >= pc - 1e-9:
                break
            if st["enc_fn"](probe, x_try) is None:
                enc[probe] = prev
                break
            cpu_total += st["enc_cpu"].get(probe, 0.0)
            jj = st["join_fn"](f"fd{x_try:.2f}")
            if jj is None:
                enc[probe] = prev
                break
            o2, util2, codes2, m2 = jj
            if not codes2 and util2 <= ACCEPT_HIGH and util2 > util_:
                if neg_checked:
                    g = _gap(o2, arm_id)
                    if g is None or g > NEG_SAFE:
                        enc[probe] = prev
                        hist.append({**rec, "skip": "neg gap", "gap": g})
                        enc_seconds += time.perf_counter() - t0
                        return best
                st["out"] = o2
                enc_seconds += time.perf_counter() - t0
                hist.append({**rec, "ok": True, "crf": x_try, "util": round(util2, 4)})
                return (o2, util2, m2, best[3])
            enc[probe] = prev
        enc_seconds += time.perf_counter() - t0
        hist.append({**rec, "skip": "no valid join"})
        return best

    def _fill_worth(use: float, last_cpu: float) -> bool:
        """Expected VMAF gain of one fill-up re-encode vs its CPU cost (lam VMAF per CPU-second)."""
        if not plan.fill_up:
            return False
        if plan.pred_vmaf is None:
            return use < ACCEPT_LOW
        v = plan.pred_vmaf
        if v >= 99.5 or use >= final_aim:
            return False
        gain = vmaf_per_log_bits(v) * math.log(final_aim / max(use, 1e-3))
        return gain > (plan.lam or DEFAULT_LAM) * last_cpu

    try:
        if plan.hopeless and plan.pred_best is not None and plan.pred_best < DEEP_HOPELESS_VMAF:
            crf = _quant(max(CRF_MIN, min(CRF_MAX, max(plan.crf_start, 40.0))))
            out, util, codes, m = _encode(HOPELESS_ARM, crf, "hopeless")
            if out is not None:
                os.replace(out, output_path)
                os.chmod(output_path, 0o644)
                return CrfOutcome(True, HOPELESS_ARM, crf, util, m.avg_kbps, m.peak_kbps, attempts, 0, enc_seconds,
                                  cpu_total, meta.nb_frames, target,
                                  reason="deep_hopeless" + (f" {','.join(codes)}" if codes else ""), history=hist)
        # ---- decode the source once (>= 2 encodes follow from here)
        t1 = time.perf_counter()
        if pre is not None:
            y4m, raw_dir, raw_cpu = pre[2]
            tm["raw_cache_bg"] = round(pre[4], 3)
        else:
            y4m, raw_dir, raw_cpu = _raw_source(input_path, meta, pts)
        tm["raw_cache"] = time.perf_counter() - t1
        cpu_total += raw_cpu
        if y4m is not None:
            src = y4m
            hist.append({"tag": "raw_cache", "cpu_seconds": round(raw_cpu, 3)})
        # ---- pilot search (skipped when the pilot arm IS the final arm)
        points: list[tuple[float, float]] = []
        pilot_attempts = 0
        best_pilot: tuple[str, float, list[str], Any, float] | None = None  # out, util, codes, m, crf
        crf = _quant(plan.crf_start)
        if plan.pilot_arm != final_arm:
            # The final arm spends exp(rate_ratio_log) x the pilot's bits at equal CRF, so search for
            # the CRF where the *pilot* uses AIM / ratio of the budget (may exceed 1: fine, the pilot
            # output is not shipped unless every final attempt fails).
            pilot_aim = final_aim * math.exp(-_ratio(crf))
            seg_probe_fut = None                       # background single-segment filter probe (FILTER_PROBE_MODE)
            seg_probe_pool = None
            pilot_outs: list[tuple[float, str]] = []  # (budget use, file) of each valid pilot, for stage 3
            pilot_crf_of: dict[str, float] = {}
            pilots_filtered = hard_vf is not None
            vq_cache: dict[str, Any] = {}

            def _vq(path_: str):
                """Exact validator-equivalent VMAF of an encode (cached per file)."""
                if path_ not in vq_cache:
                    vq_cache[path_] = quick_vmaf_gap(src, path_, subsample=1, reference_meta=meta,
                                                     threads=STAGE3_THREADS, exact_pairing=True)
                return vq_cache[path_]
            for _ in range(PILOT_MAX):
                pilot_aim = final_aim * math.exp(-_ratio(crf))
                out, util, codes, m = _encode(plan.pilot_arm, crf, "pilot")
                pilot_attempts += 1
                if out is None:
                    break
                # Solve on the binding budget share (avg or 2 s peak): peak-limited clips converge.
                use = budget_use(m.avg_kbps, m.peak_kbps, target, peak_ratio)
                points.append((crf, use))
                pilot_outs.append((use, out))
                pilot_crf_of[out] = crf
                early_best = plan.pred_best
                if seg_probe_fut is None and FILTER_PROBE and FILTER_PROBE_MODE == "segment" and getattr(plan, "stage2_fn", None):
                    try:  # stage 2 on this first pilot alone: stage 1 is far too optimistic on hard clips to gate on
                        early_best = plan.stage2_fn(solve_crf(points, STAGE2_UTIL))[3]
                    except Exception:  # noqa: BLE001
                        pass
                if (seg_probe_fut is None and FILTER_PROBE and FILTER_PROBE_MODE == "segment" and HARD_PREFILTER
                        and not plan.hopeless and early_best is not None and early_best < FILTER_PROBE_EARLY_MAX):
                    from concurrent.futures import ThreadPoolExecutor
                    seg_probe_pool = ThreadPoolExecutor(1)
                    seg_probe_fut = seg_probe_pool.submit(_seg_filter_probe, crf)
                    bg_jobs.append(seg_probe_fut)
                if not codes and (best_pilot is None or crf < best_pilot[4]):
                    best_pilot = (out, util, codes, m, crf)
                if abs(math.log(max(use, 1e-3) / pilot_aim)) < PILOT_TOL_LOG:
                    break
                above = [u for _, u in points if u > pilot_aim]
                below = [u for _, u in points if u <= pilot_aim]
                if above and below and math.log(min(above) / pilot_aim) < PILOT_BRACKET_LOG \
                        and math.log(pilot_aim / max(below)) < PILOT_BRACKET_LOG:
                    break
                nxt = _quant(solve_crf(points, pilot_aim))
                if use > pilot_aim and nxt <= crf:
                    nxt = _quant(crf + CRF_QUANT)
                if use < pilot_aim and nxt >= crf:
                    nxt = _quant(crf - CRF_QUANT)
                if nxt in {c for c, _ in points} or not (CRF_MIN <= nxt <= CRF_MAX):
                    break
                crf = nxt
            # ---- stage 2: re-predict from the measured pilot CRF at budget (VMAF MAE 7.1 -> 3.8)
            s2 = None
            if points and getattr(plan, "stage2_fn", None) is not None:
                t1 = time.perf_counter()
                try:
                    _sh = _pred_shift(input_path)
                    s2 = plan.stage2_fn(solve_crf(points, STAGE2_UTIL), **({"shift": _sh} if _sh else {}))
                    if _sh:
                        hist.append({"tag": "pred_shift", "shift": round(_sh, 3)})
                except Exception:  # noqa: BLE001
                    s2 = None
                tm["stage2"] = time.perf_counter() - t1
            # ---- stage 3 (STREAMENC_STAGE3): the pilots' measured VMAF -> per-arm prediction (hard-looking clips only)
            s3 = None
            if (STAGE3 and pilot_outs and getattr(plan, "stage3_fn", None) is not None
                    and (s2 is None or s2[3] < STAGE3_MAX_PRED)):
                t1 = time.perf_counter()
                try:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(len(pilot_outs)) as ex_:
                        qs = list(ex_.map(lambda uo: (uo[0], _vq(uo[1])), pilot_outs))
                    s3_cpu = sum(q_.cpu_seconds for _, q_ in qs)
                    cpu_total += s3_cpu
                    v8 = _pilot_vmaf_at_budget([(u_, q_.vmaf) for u_, q_ in qs])
                    s3 = plan.stage3_fn(solve_crf(points, STAGE2_UTIL), v8)
                    hist.append({"tag": "stage3", "v8": round(v8, 3), "pilots": [(round(u_, 4), round(q_.vmaf, 3)) for u_, q_ in qs],
                                 "pred_best": round(s3[3], 2) if s3 else None, "hopeless": s3[2] if s3 else None,
                                 "seconds": round(time.perf_counter() - t1, 3), "cpu_seconds": round(s3_cpu, 3),
                                 "arm": s3[0] if s3 else None})
                except Exception as exc:  # noqa: BLE001 — fall back to stage 2
                    hist.append({"tag": "stage3", "error": str(exc)[-200:]})
                    s3 = None
            if s3 is not None and STAGE3_MODE == "hopeless" and not s3[2]:
                hist.append({"tag": "stage3_unused", "reason": "not hopeless", "pred_best": round(s3[3], 2)})
                s3 = None
            if s3 is not None:
                if s2 is not None:
                    hist.append({"tag": "stage2_unused", "arm": s2[0], "pred_vmaf": round(s2[1].get(s2[0], 0.0), 2),
                                 "pred_best": round(s2[3], 2)})
                s2 = s3
            if s2 is not None:
                arm2, vm2, hop2, best2 = s2
                plan.vmaf_by_arm, plan.pred_best = vm2, best2
                hist.append({"tag": "stage2", "arm": arm2, "pred_vmaf": round(vm2.get(arm2, 0.0), 2),
                             "pred_best": round(best2, 2), "hopeless": hop2, "was": final_arm})
                if hop2:
                    # Predicted < 57 VMAF on every arm: only speed counts. The pilot arm is the speed arm;
                    # ship the valid pilot if it is near budget, else fit the pilot arm itself (cheap).
                    plan.hopeless = True
                    if best_pilot is not None:
                        out, util, codes, m, pcrf = best_pilot
                        if budget_use(m.avg_kbps, m.peak_kbps, target, peak_ratio) >= STAGE2_SHIP_PILOT_USE:
                            os.replace(out, output_path)
                            os.chmod(output_path, 0o644)
                            return CrfOutcome(True, plan.pilot_arm, pcrf, util, m.avg_kbps, m.peak_kbps, attempts,
                                              pilot_attempts, enc_seconds, cpu_total, meta.nb_frames, target,
                                              reason="stage2_hopeless", history=hist)
                    final_arm = plan.pilot_arm
                    ratio_fn, plan.rate_ratio_log = None, 0.0
                elif arm2 != final_arm:
                    final_arm = arm2
                    ratio_fn = plan.ratio_fn_for(final_arm) if hasattr(plan, "ratio_fn_for") else ratio_fn
                plan.pred_vmaf = vm2.get(final_arm, plan.pred_vmaf)
                fp_done = False
                if seg_probe_fut is not None:
                    t1 = time.perf_counter()
                    try:
                        sp = seg_probe_fut.result()
                        cpu_total += sp["cpu"]
                        cand = not plan.hopeless and plan.pred_best is not None and plan.pred_best < FILTER_PROBE_MAX_PRED
                        if cand:
                            on = sp["gain"] > FILTER_PROBE_MARGIN
                            hard_vf = sp["vf"] if on else None
                            fp_done = True
                        hist.append({"tag": "filter_probe", "mode": "segment", "used": cand, "on": cand and sp["gain"] > FILTER_PROBE_MARGIN,
                                     "gain": round(sp["gain"], 3), "crf": sp["crf"], "seg": sp["seg"], "probe_seconds": sp["seconds"],
                                     "wait": round(time.perf_counter() - t1, 3), "pred_best": round(plan.pred_best or 0.0, 2)})
                    except Exception as exc:  # noqa: BLE001 — fall back to the fixed rule
                        hist.append({"tag": "filter_probe", "mode": "segment", "error": str(exc)[-200:]})
                    finally:
                        seg_probe_pool.shutdown(wait=False)
                if (not fp_done and seg_probe_fut is None and FILTER_PROBE and FILTER_PROBE_MODE != "segment"
                        and HARD_PREFILTER and not plan.hopeless and pilot_outs
                        and plan.pred_best is not None and plan.pred_best < FILTER_PROBE_MAX_PRED):
                    t1 = time.perf_counter()
                    try:
                        from concurrent.futures import ThreadPoolExecutor
                        from streamenc.filters.policy import filter_vf as _fvf
                        fvf = _fvf(HARD_PREFILTER).format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction))
                        u_p, o_p = min(pilot_outs, key=lambda uo: abs(math.log(max(uo[0], 1e-6) / STAGE2_UTIL)))
                        c_p = pilot_crf_of[o_p]
                        keep_vf = hard_vf
                        hard_vf = None if pilots_filtered else fvf      # the counterpart of the pilot
                        o_c, _u_c, _codes_c, m_c = _encode(plan.pilot_arm, c_p, "fprobe")
                        hard_vf = keep_vf
                        if o_c is not None:
                            u_c = budget_use(m_c.avg_kbps, m_c.peak_kbps, target, peak_ratio)
                            with ThreadPoolExecutor(2) as ex_:
                                q_p, q_c = list(ex_.map(_vq, [o_p, o_c]))
                            (uu, vu), (uf, vf_) = ((u_c, q_c.vmaf), (u_p, q_p.vmaf)) if pilots_filtered else ((u_p, q_p.vmaf), (u_c, q_c.vmaf))
                            gain = vf_ + vmaf_per_log_bits(vf_) * math.log(max(uu, 1e-6) / max(uf, 1e-6)) - vu
                            on = gain > FILTER_PROBE_MARGIN
                            cpu_total += q_p.cpu_seconds + q_c.cpu_seconds
                            hard_vf = fvf if on else None
                            fp_done = True
                            hist.append({"tag": "filter_probe", "filter": HARD_PREFILTER, "on": on, "gain": round(gain, 3),
                                         "crf": c_p, "unfiltered": (round(uu, 4), round(vu, 3)), "filtered": (round(uf, 4), round(vf_, 3)),
                                         "pred_best": round(plan.pred_best, 2), "seconds": round(time.perf_counter() - t1, 3)})
                    except Exception as exc:  # noqa: BLE001 — fall back to the fixed rule
                        hist.append({"tag": "filter_probe", "error": str(exc)[-200:]})
                if hard_vf is None and not plan.hopeless and not fp_done:
                    hard_vf = _hard_vf("stage2")  # final encodes only; the final rate fit absorbs the rate change
                if (FILTER_MODEL and hard_vf and not fp_done and not plan.hopeless and points
                        and getattr(plan, "fgain_fn", None) is not None):
                    try:
                        g_ = plan.fgain_fn(solve_crf(points, STAGE2_UTIL))
                        keep_ = g_ > FILTER_MODEL_MARGIN
                        hist.append({"tag": "filter_model", "gain": round(g_, 3), "keep": keep_,
                                     "pred_best": round(plan.pred_best or 0.0, 2)})
                        if not keep_:
                            hard_vf = None  # final encodes without the prefilter
                    except Exception as exc:  # noqa: BLE001 — keep the fixed rule
                        hist.append({"tag": "filter_model", "error": str(exc)[-200:]})
                easy = plan.pred_vmaf is not None and plan.pred_vmaf >= EASY_VMAF and not plan.hopeless
                final_aim = FINAL_AIM_EASY if easy else FINAL_AIM
                pilot_aim = final_aim * math.exp(-_ratio(crf))
            pil_cpu = [h_["cpu_seconds"] for h_ in hist if h_.get("tag") == "pilot" and "cpu_seconds" in h_]
            pil_cpu_mpf = (sum(pil_cpu) / len(pil_cpu) / max(meta.width * meta.height * meta.nb_frames / 1e6, 1e-6)
                           if pil_cpu else None)
            cheap = ((not HARD_ARM_MAX_HEIGHT or meta.height <= HARD_ARM_MAX_HEIGHT)
                     and (not HARD_ARM_MAX_PILOT_CPU or (pil_cpu_mpf is not None and pil_cpu_mpf <= HARD_ARM_MAX_PILOT_CPU)))
            if (hard_vf and HARD_ARM and not plan.hopeless and _preset(final_arm) > _preset(HARD_ARM)
                    and (plan.pred_best if plan.pred_best is not None else 0.0) < HARD_ARM_MAX_PRED and cheap):
                hist.append({"tag": "hard_arm", "was": final_arm, "arm": HARD_ARM, "pred_best": plan.pred_best,
                             "prefilter": HARD_ARM_PREFILTER or None,
                             "pilot_cpu_mpf": None if pil_cpu_mpf is None else round(pil_cpu_mpf, 4)})
                final_arm = HARD_ARM
                if HARD_ARM_PREFILTER:
                    from streamenc.filters.policy import filter_vf as _fvf2
                    hard_vf = _fvf2(HARD_ARM_PREFILTER).format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction))
                if hasattr(plan, "ratio_fn_for"):  # p1 has no rate model: p2's bits are the closest proxy
                    ratio_fn = plan.ratio_fn_for(HARD_ARM if HARD_ARM in (plan.vmaf_by_arm or {}) else "svt_crf_p2_qm_la")
                pilot_aim = final_aim * math.exp(-_ratio(crf))
            crf_est = solve_crf(points, pilot_aim) if points else plan.crf_start
            if points:  # re-evaluate the per-clip ratio at the estimate once
                pilot_aim = final_aim * math.exp(-_ratio(crf_est))
                crf_est = solve_crf(points, pilot_aim)
            pilot_slope = local_slope(points, pilot_aim) if points else None
        else:
            crf_est = plan.crf_start
            pilot_slope = None

        # ---- final encode(s)
        def _fit(arm_id: str, start_crf: float, tag: str) -> tuple[str, float, Any, float] | None:
            """Encode ``arm_id`` from ``start_crf``, correct towards final_aim; best valid (lowest CRF)."""
            f_crf = _quant(max(CRF_MIN, min(CRF_MAX, start_crf)))
            fpoints: list[tuple[float, float]] = []
            fflags: list[tuple[float, float, bool]] = []
            fbest: tuple[str, float, Any, float] | None = None  # out, util, m, crf
            fixes = 0
            free_done = [False]
            while True:
                out, util, codes, m = _encode(arm_id, f_crf, tag)
                if out is None:
                    break
                use = budget_use(m.avg_kbps, m.peak_kbps, target, peak_ratio)
                fpoints.append((f_crf, use))
                over = bool(codes) or util > ACCEPT_HIGH
                fflags.append((f_crf, use, over))
                if not over and (fbest is None or f_crf < fbest[3]):
                    fbest = (out, util, m, f_crf)
                if not over and not _fill_worth(use, hist[-1].get("cpu_seconds", 0.0)):
                    ev = _bstate.last_one_event(input_path) if FREE_FILL else None
                    if (ev is not None and not ev.is_set() and use < FREE_FILL_BELOW and not free_done[0]
                            and _remaining() >= 1.5 * hist[-1].get("seconds", 0.0) + 5.0):
                        free_done[0] = True
                        hi_over = max((c for c, _, o in fflags if o), default=None)
                        x_ = _quant(solve_crf(fpoints, FREE_FILL_AIM, slope_hint=pilot_slope))
                        x_ = min(x_, _quant(f_crf - CRF_QUANT))
                        if hi_over is not None:
                            x_ = max(x_, _quant(hi_over + CRF_QUANT))
                        if CRF_MIN <= x_ < f_crf and x_ not in {c for c, _ in fpoints}:
                            o2, u2, c2, m2 = _encode(arm_id, x_, tag + "free", cancel=ev)
                            rec_ = {"tag": "free_fill", "from_crf": f_crf, "from_use": round(use, 4), "crf": x_}
                            if o2 is not None:
                                use2 = budget_use(m2.avg_kbps, m2.peak_kbps, target, peak_ratio)
                                ok2 = not c2 and u2 <= ACCEPT_HIGH
                                rec_.update({"use": round(use2, 4), "kept": ok2})
                                if ok2 and (fbest is None or x_ < fbest[3]):
                                    fbest = (o2, u2, m2, x_)
                            else:
                                rec_["kept"] = False
                            hist.append(rec_)
                    break
                if fixes >= FINAL_MAX_FIX or _bracket_done(fflags):
                    break
                # Expect the final encode of the heaviest preset to take about as long again.
                last = hist[-1].get("seconds", 0.0)
                if _remaining() < 1.5 * last + 2.0:
                    break
                nxt = _quant(solve_crf(fpoints, final_aim, slope_hint=pilot_slope))
                if over:
                    f_crf = _quant(min(CRF_MAX, max(nxt, f_crf + 2 * CRF_QUANT)))
                else:
                    f_crf = _quant(max(CRF_MIN, min(nxt, f_crf - CRF_QUANT)))
                # Stay strictly inside the (over, valid) bracket measured so far.
                hi_over = max((c for c, _, o in fflags if o), default=None)
                lo_ok = min((c for c, _, o in fflags if not o), default=None)
                if lo_ok is not None:
                    f_crf = min(f_crf, _quant(lo_ok - CRF_QUANT))
                if hi_over is not None:
                    f_crf = max(f_crf, _quant(hi_over + CRF_QUANT))
                if f_crf in {c for c, _ in fpoints} or not (CRF_MIN <= f_crf <= CRF_MAX):
                    break
                fixes += 1
            return fbest

        best = None
        if SEGFINAL and not plan.hopeless:
            try:
                best = _segmented_final(final_arm, crf_est + plan.crf_offset)
            except Exception as exc:  # noqa: BLE001 — fall back to the whole-clip fit
                hist.append({"tag": "segfinal", "error": str(exc)[-200:]})
                best = None
        if best is None:
            best = _fit(final_arm, crf_est + plan.crf_offset, "final")
        chosen_id = final_arm
        if best is None and best_pilot is not None:
            # Every final attempt overshot/failed: ship the valid pilot encode rather than nothing.
            out, util, codes, m, pcrf = best_pilot
            best = (out, util, m, pcrf)
            chosen_id = plan.pilot_arm
        if best is None:
            return CrfOutcome(False, final_arm, None, None, None, None, attempts, pilot_attempts, enc_seconds,
                              cpu_total, meta.nb_frames, target, reason="no valid encode", history=hist)

        # ---- VMAF-NEG guard: measure the gap when the model says it may fail; re-fit a NEG-safe arm.
        def _p_neg(arm_id: str, crf_: float) -> float | None:
            fn = getattr(plan, "gap_fn", None)
            if fn is None:
                return None
            try:
                r = fn(arm_id, crf_, (getattr(plan, "vmaf_by_arm", None) or {}).get(arm_id))
            except Exception:  # noqa: BLE001
                return None
            if r is None:
                return None
            g, sd = r
            return 0.5 * (1.0 - math.erf((NEG_SAFE - g) / (max(sd, 0.05) * math.sqrt(2.0))))

        gap_of: dict[str, tuple[float, bool]] = {}   # file -> (gap, every frame) of each measurement

        def _gap(out_path: str, arm_id: str, exact: bool = False) -> float | None:
            nonlocal cpu_total
            try:
                full_ = exact or (bool(NEG_FULL_FPS) and float(meta.fps) >= NEG_FULL_FPS)
                q = quick_vmaf_gap(src, out_path, subsample=1 if full_ else NEG_SUBSAMPLE, reference_meta=meta,
                                   threads=(8, 8) if full_ else None)
            except Exception as exc:  # noqa: BLE001
                hist.append({"tag": "neg_check", "arm": arm_id, "error": str(exc)[-200:]})
                return None
            cpu_total += q.cpu_seconds
            hist.append({"tag": "neg_check", "arm": arm_id, "gap": round(q.gap, 3), "vmaf": round(q.vmaf, 3),
                         "seconds": round(q.wall_seconds, 3), "cpu_seconds": round(q.cpu_seconds, 3)}
                        | ({"exact": True} if exact else {}))
            gap_of[out_path] = (q.gap, bool(full_))
            return q.gap

        neg_reason = ""
        if not plan.hopeless:
            p_fail = _p_neg(chosen_id, best[3])
            unknown_ = NEG_CHECK_UNKNOWN and (p_fail is None or bool(hard_vf))
            if unknown_ and (p_fail is None or p_fail < NEG_CHECK_P):
                hist.append({"tag": "neg_unknown", "arm": chosen_id, "p_fail": None if p_fail is None else round(p_fail, 4),
                             "filtered": bool(hard_vf)})
            if unknown_ or (p_fail is not None and p_fail >= NEG_CHECK_P):
                g0 = _gap(best[0], chosen_id)
                if g0 is not None and g0 > NEG_SAFE:
                  tried = [(g0, best, chosen_id, hard_vf)]
                  try:  # 2026-11-12 (rc7): a failure in the fallback must never lose the valid encode
                    src_arm, src_crf = chosen_id, best[3]
                    pf_ok = False
                    if (hard_vf and NEG_PREFILTER_FALLBACK and NEG_PREFILTER_FALLBACK != HARD_PREFILTER
                            and _remaining() >= 2.0 * max((h.get("seconds", 0.0) for h in hist), default=0.0) + 5.0):
                        from streamenc.filters.policy import filter_vf
                        fv = filter_vf(NEG_PREFILTER_FALLBACK)
                        hard_vf = fv.format(w=meta.width, h=meta.height, fps=str(meta.fps_fraction)) if fv else None
                        hist.append({"tag": "neg_prefilter", "filter": NEG_PREFILTER_FALLBACK, "arm": chosen_id,
                                     "crf": best[3]})
                        pf_best = None
                        if SEGFINAL and get_arm(chosen_id).get("rc") == "crf":
                            try:
                                pf_best = _segmented_final(chosen_id, best[3])
                            except Exception as exc:  # noqa: BLE001
                                hist.append({"tag": "segfinal", "error": str(exc)[-200:]})
                        if pf_best is None:
                            pf_best = _fit(chosen_id, best[3], "negpf")
                        if pf_best is not None:
                            gp = _gap(pf_best[0], chosen_id)
                            tried.append((gp if gp is not None else NEG_LIMIT + 1.0, pf_best, chosen_id, hard_vf))
                            pf_ok = gp is not None and gp <= NEG_SAFE
                    for alt in (() if pf_ok else NEG_SAFE_ARMS):
                        if alt == chosen_id or alt in {a for _, _, a, _ in tried}:  # rc7 fix: 4-tuples (crashed rc1-rc6)
                            continue
                        if _remaining() < 2.0 * max((h.get("seconds", 0.0) for h in hist), default=0.0) + 5.0:
                            break
                        shift = NEG_CRF_SHIFT.get(alt, {}).get(src_arm, 0.0)
                        alt_best = _fit(alt, src_crf + shift, "neg")
                        if alt_best is None:
                            continue
                        ga = _gap(alt_best[0], alt)
                        tried.append((ga if ga is not None else NEG_LIMIT + 1.0, alt_best, alt, hard_vf))
                        if ga is not None and ga <= NEG_SAFE:
                            break
                        src_arm, src_crf = alt, alt_best[3]
                  except Exception as exc:  # noqa: BLE001 — ship the best encode tried so far
                    hist.append({"tag": "neg_fallback", "error": str(exc)[-200:]})
                  if NEG_EXACT_PICK and len(tried) > 1 and min(r[0] for r in tried) > NEG_SAFE:
                      # 1.0.0rc16 (from rc15): no candidate reached NEG_SAFE on the 1-in-5 estimate: measure every frame of
                      # each and ship the lowest true gap (an estimate can be off by up to 0.07 either way)
                      ex_ = []
                      for g_, b_, a_, f_ in tried:
                          k_ = gap_of.get(b_[0])
                          ge = k_[0] if k_ and k_[1] else None
                          if ge is None and g_ <= NEG_LIMIT and _remaining() > 15.0:
                              ge = _gap(b_[0], a_, exact=True)
                          ex_.append((ge if ge is not None else g_, b_, a_, f_))
                      hist.append({"tag": "neg_exact_pick", "estimates": [round(r[0], 3) for r in tried],
                                   "exact": [round(r[0], 3) for r in ex_]})
                      tried = ex_
                  g_ship, best, chosen_id, hard_vf = min(tried, key=lambda r: r[0])  # segfill uses the shipped filter
                  neg_reason = f"neg_fallback gap {g0:.2f}->{g_ship:.2f}"
        if SEGFILL and not plan.hopeless and get_arm(chosen_id).get("rc") == "crf":
            try:
                best = _segment_fill(best, chosen_id, neg_checked=bool(neg_reason) or any(
                    h.get("tag") == "neg_check" for h in hist))
            except Exception as exc:  # noqa: BLE001 — never lose the valid encode
                hist.append({"tag": "segfill", "error": str(exc)[-200:]})
        if SEGFINAL_FILLDOWN and not plan.hopeless:
            try:
                best = _probe_fill_down(best, chosen_id, neg_checked=any(h.get("tag") == "neg_check" for h in hist))
            except Exception as exc:  # noqa: BLE001 — never lose the valid encode
                hist.append({"tag": "filldown", "error": str(exc)[-200:]})
        out, util, m, crf_used = best
        os.replace(out, output_path)
        os.chmod(output_path, 0o644)
        return CrfOutcome(True, chosen_id, crf_used, util, m.avg_kbps, m.peak_kbps, attempts, pilot_attempts,
                          enc_seconds, cpu_total, meta.nb_frames, target, reason=neg_reason, history=hist)
    finally:
        t1 = time.perf_counter()
        for f_ in bg_jobs:  # a background probe still writing into work / reading the raw cache
            try:
                f_.exception()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(work, ignore_errors=True)
        if raw_dir:
            shutil.rmtree(raw_dir, ignore_errors=True)
        tm["cleanup"] = time.perf_counter() - t1
        tm["item"] = time.perf_counter() - t_start
        hist.append({"tag": "timing"} | {f"t_{k}": round(v, 3) for k, v in tm.items()})


Predictor = Callable[[Any, Any], Plan]  # (features, meta) -> Plan


def rule_predictor(features: Any, meta: Any) -> Plan:
    """Fallback plan when no trained model is available."""
    return Plan()


def process_item_crf(
    input_path: str,
    output_path: str,
    *,
    predictor: Predictor | None = None,
    bits_per_pixel: float = 0.05,
    max_keyframe_interval: float = 2.0,
    peak_window: float = 2.0,
    peak_ratio: float = 1.5,
    time_budget_seconds: float | None = None,
    max_samples: int = 24,
    **_ignored: Any,
):
    """Batch-executor item function (same ItemResult contract as pipeline.item.process_item)."""
    from streamenc.features.analyze import analyze_video
    from streamenc.pipeline.item import ItemResult

    t0 = time.perf_counter()
    feature_seconds = 0.0
    plan_seconds = 0.0
    plan = Plan()
    rule_id = "CRF_DEFAULT"
    pre = None
    if RAW_PREFETCH:
        try:
            from concurrent.futures import ThreadPoolExecutor
            pool_ = ThreadPoolExecutor(1, thread_name_prefix="prefetch")
            pre = pool_.submit(_prefetch_source, input_path)
            pool_.shutdown(wait=False)
        except Exception:  # noqa: BLE001 - no prefetch: encode_item_crf does these steps itself
            pre = None
    try:
        analysis = analyze_video(input_path, mode="sampled", max_samples=max_samples, max_width=640)
        feature_seconds = time.perf_counter() - t0
        pred = predictor or rule_predictor
        plan = pred(analysis.features, probe_video(input_path, count_frames=False))
        plan_seconds = time.perf_counter() - t0 - feature_seconds
        rule_id = "CRF_MODEL" if predictor else "CRF_RULES"
    except Exception as exc:  # noqa: BLE001 — never lose an item to the predictor
        log.warning("predictor failed for %s: %s", input_path, exc)
    remaining = None if time_budget_seconds is None else max(5.0, time_budget_seconds - (time.perf_counter() - t0))
    try:
        outcome = encode_item_crf(
            input_path, output_path, plan,
            bits_per_pixel=bits_per_pixel, max_keyframe_interval=max_keyframe_interval,
            peak_window=peak_window, peak_ratio=peak_ratio, time_budget_seconds=remaining, prefetch=pre,
        )
    finally:
        if pre is not None:  # the raw cache dir (encode_item_crf removes it too; this covers an early exception)
            try:
                rd = pre.result()[2][1]
                if rd:
                    shutil.rmtree(rd, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass
    log.info(
        "item %s: %s crf=%s util=%s attempts=%d (pilot %d) cpu=%.1fs wall=%.1fs plan=%s%s",
        Path(input_path).name[:12], outcome.config_id, outcome.crf,
        None if outcome.util is None else round(outcome.util, 4), outcome.attempts, outcome.pilot_attempts,
        outcome.cpu_seconds, time.perf_counter() - t0, plan.final_arm,
        (f" ({plan.notes})" if plan.notes else "") + (f" [{outcome.reason}]" if outcome.reason else ""),
    )
    log.info("  %s", {"tag": "timing0", "t_features": round(feature_seconds, 3), "t_plan": round(plan_seconds, 3)})
    for h in outcome.history:
        log.info("  %s", {k: v for k, v in h.items() if k != "arm"} | {"arm": h.get("arm")})
    return ItemResult(
        input=input_path,
        output=output_path if outcome.ok else "",
        ok=outcome.ok,
        clip_id=Path(input_path).name,
        config_id=outcome.config_id,
        rule_id=rule_id,
        prefilter="none",
        feature_seconds=feature_seconds,
        encode_seconds=outcome.encode_seconds,
        total_seconds=time.perf_counter() - t0,
        attempts=outcome.attempts,
        avg_kbps=outcome.avg_kbps,
        peak_kbps=outcome.peak_kbps,
        frames=outcome.frames,
        reason=outcome.reason,
        fallback_used=outcome.config_id != (SPEED_ARM if plan.hopeless else plan.final_arm),
        extras={"crf": outcome.crf, "util": outcome.util, "cpu_seconds": outcome.cpu_seconds,
                "pilot_attempts": outcome.pilot_attempts, "plan": asdict(plan), "history": outcome.history},
    )
