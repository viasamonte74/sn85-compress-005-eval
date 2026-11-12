"""Encoding execution helpers."""

from .packets import PacketStats, measure_packet_kbps

__all__ = ["PacketStats", "measure_packet_kbps", "EncodeResult", "encode_baseline"]


def __getattr__(name: str):
    if name in {"EncodeResult", "encode_baseline"}:
        from .baseline import EncodeResult, encode_baseline

        return {"EncodeResult": EncodeResult, "encode_baseline": encode_baseline}[name]
    raise AttributeError(name)
