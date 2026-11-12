"""Inference helper: video path → predicted config_id."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from streamenc.ml.encoder_rank.catalog import LABEL_CATALOG_IDS
from streamenc.ml.encoder_rank.model import VideoEncoderRanker
from streamenc.ml.encoder_rank.video_io import sample_video_tensor

DEFAULT_CKPT = Path(__file__).resolve().parents[3] / "results" / "encoder_rank_train" / "best.pt"


@lru_cache(maxsize=1)
def _load_model(ckpt_path: str, device: str) -> tuple[VideoEncoderRanker, list[str], dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    catalog_ids = list(ckpt.get("catalog_ids") or LABEL_CATALOG_IDS)
    args = ckpt.get("args") or {}
    model = VideoEncoderRanker(
        len(catalog_ids),
        img_size=int(args.get("img_size", 128)),
        num_frames=int(args.get("num_frames", 16)),
        dim=int(args.get("dim", 256)),
        depth=int(args.get("depth", 6)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, catalog_ids, args


def predict_config(
    video_path: str | Path,
    *,
    ckpt: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Return {config_id, confidence, probs}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(ckpt or DEFAULT_CKPT)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(ckpt)
    model, catalog_ids, args = _load_model(ckpt, device)
    video = sample_video_tensor(
        video_path,
        num_frames=int(args.get("num_frames", 16)),
        img_size=int(args.get("img_size", 128)),
    )
    with torch.no_grad():
        out = model(video.unsqueeze(0).to(device))
        probs = torch.softmax(out["logits"], dim=-1)[0].cpu()
    idx = int(probs.argmax().item())
    return {
        "config_id": catalog_ids[idx],
        "config_index": idx,
        "confidence": float(probs[idx].item()),
        "probs": {catalog_ids[i]: float(probs[i].item()) for i in range(len(catalog_ids))},
    }
