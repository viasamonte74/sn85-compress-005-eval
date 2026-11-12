"""PyTorch dataset for encoder-rank training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from streamenc.ml.encoder_rank.catalog import LABEL_CATALOG_IDS, load_catalog
from streamenc.ml.encoder_rank.video_io import sample_video_tensor


class EncoderRankDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        num_frames: int = 16,
        img_size: int = 128,
        catalog_ids: tuple[str, ...] | None = None,
        min_winner_score: float = 0.0,
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.catalog_ids = list(catalog_ids or LABEL_CATALOG_IDS)
        self.id_to_idx = {c: i for i, c in enumerate(self.catalog_ids)}

        labels_path = self.root / "labels.jsonl"
        self.labels: list[dict[str, Any]] = []
        with labels_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Only high-quality, gate-valid winners for CE supervision.
                if row.get("split") != split:
                    continue
                if row.get("winner_config_id") not in self.id_to_idx:
                    continue
                if row.get("trainable") is False:
                    continue
                if not row.get("winner_passed"):
                    continue
                # Winners with clip_score 0 are only "best VMAF", not high-score settings.
                if float(row.get("winner_clip_score") or 0.0) <= min_winner_score:
                    continue
                self.labels.append(row)

        trials_path = self.root / "trials.csv"
        self.trials = pd.read_csv(trials_path) if trials_path.is_file() else pd.DataFrame()

    def __len__(self) -> int:
        return len(self.labels)

    def _target_score_vector(self, clip_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (C,) soft ranking targets from trials; mask missing configs."""
        C = len(self.catalog_ids)
        scores = torch.full((C,), -1e4)
        mask = torch.zeros(C, dtype=torch.bool)
        if self.trials.empty:
            return scores, mask
        g = self.trials[self.trials["clip_id"] == clip_id]
        for _, r in g.iterrows():
            cid = r["config_id"]
            if cid not in self.id_to_idx:
                continue
            i = self.id_to_idx[cid]
            mask[i] = True
            # Soft ranking targets: gate-valid trials only.
            if not bool(r.get("passed_streaming")) or not bool(r.get("passed")):
                continue
            if not pd.notna(r.get("vmaf")):
                continue
            vmaf = float(r["vmaf"])
            cscore = float(r["clip_score"]) if pd.notna(r.get("clip_score")) else 0.0
            scores[i] = 10.0 * cscore + 0.05 * vmaf
        return scores, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.labels[idx]
        video = sample_video_tensor(row["path"], num_frames=self.num_frames, img_size=self.img_size)
        y = self.id_to_idx[row["winner_config_id"]]
        target_scores, mask = self._target_score_vector(row["clip_id"])
        return {
            "video": video,
            "label": y,
            "target_scores": target_scores,
            "score_mask": mask,
            "clip_id": row["clip_id"],
            "path": row["path"],
            "winner_config_id": row["winner_config_id"],
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video": torch.stack([b["video"] for b in batch], dim=0),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "target_scores": torch.stack([b["target_scores"] for b in batch], dim=0),
        "score_mask": torch.stack([b["score_mask"] for b in batch], dim=0),
        "clip_id": [b["clip_id"] for b in batch],
        "path": [b["path"] for b in batch],
        "winner_config_id": [b["winner_config_id"] for b in batch],
    }
