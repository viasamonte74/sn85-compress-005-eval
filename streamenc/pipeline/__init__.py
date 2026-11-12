"""End-to-end encode pipeline (decision → filter → correct)."""

from streamenc.pipeline.item import ItemResult, process_item

PHASE = 10

__all__ = ["PHASE", "ItemResult", "process_item"]
