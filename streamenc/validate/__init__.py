"""Validation engine (Phase 3)."""

from .official_streaming import (
    StreamingMeasurement,
    StreamingPacket,
    ValidityViolation,
    measure_streaming,
    read_packets_ffprobe,
    streaming_target_kbps,
    streaming_violations,
)
from .report import GateResult, ValidationReport
from .validate_clip import validate_clip
from .vmaf import VMAF_MODEL_DELTA_MAX, compute_vmaf_pair

__all__ = [
    "StreamingMeasurement",
    "StreamingPacket",
    "ValidityViolation",
    "measure_streaming",
    "read_packets_ffprobe",
    "streaming_target_kbps",
    "streaming_violations",
    "GateResult",
    "ValidationReport",
    "validate_clip",
    "VMAF_MODEL_DELTA_MAX",
    "compute_vmaf_pair",
]
