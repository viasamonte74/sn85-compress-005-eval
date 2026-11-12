"""Frozen encoder catalog for supervised encoder-rank training (label policy v3).

Label ids are ``arm`` or ``arm+prefilter``; arms live in ``streamenc.encoders.arms``
(all encoders of the bundled ffmpeg n8.1 plus SVT-AV1 v1.7 from system ffmpeg) and
prefilters in ``streamenc.filters.policy.FILTER_CATALOG``. The arm set was chosen from
the hard-clip exploration (results/hard_explore/trials.csv): every arm here either won
or was within ~0.5 VMAF of the best on some hard clip, or is a fast/other-family arm
kept so the model can trade speed for quality on easier content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamenc.encoders.arms import get_arm
from streamenc.filters.policy import FILTER_CATALOG

LABEL_CATALOG_IDS: tuple[str, ...] = (
    # v7 (2026-11-08): SVT-AV1 4.3, fractional-CRF only (deterministic; VBR dropped), all with
    # enable-qm/qm-min 0/lookahead 120 except the speed arm. The pilot arm is searched first and its
    # CRF seeds the others (PILOT_LABEL_ID).
    "svt_crf_p5_qm_la",      # pilot (fast, CRF transfers to p2-p4 within ~0.3)
    "svt_crf_p2_qm_la",
    "svt_crf_p3_qm_la",
    "svt_crf_p3_qm_la_lqb",
    "svt_crf_p4_qm_la",
    "svt_crf_p4_qm_la_lqb",
    "svt_crf_p6_qm_la",
    "svt_crf_p8_t1",         # hopeless-clip speed arm
    "svt_crf_p3_t0_qm_la",   # VQ tune: lower VMAF-NEG gap
)

PILOT_LABEL_ID = "svt_crf_p5_qm_la"

LABEL_POLICY_VERSION = 8  # 8: peak-aware CRF search, bracket stop + bisection, 7 attempts, low-CRF slope


def resolve_label(label_id: str) -> tuple[dict[str, Any], str]:
    """label id -> (arm dict, prefilter name)."""
    arm_id, _, prefilter = label_id.partition("+")
    prefilter = prefilter or "none"
    if prefilter not in FILTER_CATALOG:
        raise KeyError(f"unknown prefilter {prefilter}")
    return get_arm(arm_id), prefilter


def build_catalog() -> dict[str, Any]:
    entries = []
    for i, lid in enumerate(LABEL_CATALOG_IDS):
        arm, pf = resolve_label(lid)
        entries.append({"config_id": lid, "class_index": i, "arm": arm, "prefilter": pf,
                        "prefilter_vf": FILTER_CATALOG[pf]})
    return {
        "version": LABEL_POLICY_VERSION,
        "description": "Discrete encoder action space for video-transformer ranking",
        "config_ids": list(LABEL_CATALOG_IDS),
        "entries": entries,
        "encode": "streamenc.encode.ratefit.encode_rate_fit (closed-loop to ~99% of the 0.05 bpp budget)",
        "ranking": {
            "primary": ["gate_passed", "clip_score", "vmaf", "util_score"],
            "requires": ["passed_streaming", "passed_perceptual_vmaf_gates"],
            "trainable_only_if": "winner gate-valid; CE supervision additionally requires clip_score > 0",
            "efficient_winner": "fastest gate-valid trial within 0.5 VMAF of the winner",
        },
        "label_policy_version": LABEL_POLICY_VERSION,
    }


def write_catalog(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_catalog(), indent=2))
    return path


def load_catalog(path: Path | str | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "data" / "encoder_rank" / "catalog.json"
    path = Path(path)
    if not path.is_file():
        return build_catalog()
    return json.loads(path.read_text())
