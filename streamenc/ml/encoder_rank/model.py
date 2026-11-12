"""Lightweight video transformer for encoder-config classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """Tubelet embedding: (B,C,T,H,W) → (B, N, D)."""

    def __init__(self, img_size: int = 128, patch: int = 16, tubelet: int = 2, in_ch: int = 3, dim: int = 256):
        super().__init__()
        self.patch = patch
        self.tubelet = tubelet
        self.grid = img_size // patch
        self.proj = nn.Conv3d(
            in_ch,
            dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,T,H,W
        x = self.proj(x)  # B,D,T',H',W'
        b, d, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, d)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class VideoEncoderRanker(nn.Module):
    """Video transformer → config class logits (+ optional score vector over catalog)."""

    def __init__(
        self,
        n_classes: int,
        *,
        img_size: int = 128,
        num_frames: int = 16,
        patch: int = 16,
        tubelet: int = 2,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        drop: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.n_classes = n_classes
        self.embed = PatchEmbed3D(img_size=img_size, patch=patch, tubelet=tubelet, dim=dim)
        n_tokens = (num_frames // tubelet) * (img_size // patch) * (img_size // patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_tokens, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        # Soft score head: predicts relative quality per config (for ranking loss).
        self.score_head = nn.Linear(dim, n_classes)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """video: (B, T, C, H, W) in [0,1] or normalized."""
        x = video.permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,H,W
        tok = self.embed(x)
        b = tok.size(0)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, tok], dim=1)
        # Interpolate pos if token count mismatches (variable T).
        if x.size(1) != self.pos.size(1):
            pos = F.interpolate(
                self.pos.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return {
            "logits": self.head(cls_out),
            "score_logits": self.score_head(cls_out),
            "embedding": cls_out,
        }


def ranking_loss(score_logits: torch.Tensor, target_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Listwise CE over softmax of predicted scores vs soft targets from normalized metrics.

    target_scores: (B, C) raw metric (e.g. clip_score*10 + vmaf/100); masked invalid trials → -inf
    """
    # Soft targets
    t = target_scores.masked_fill(~mask, -1e4)
    t = F.softmax(t, dim=-1)
    logp = F.log_softmax(score_logits, dim=-1)
    return -(t * logp).sum(dim=-1).mean()
