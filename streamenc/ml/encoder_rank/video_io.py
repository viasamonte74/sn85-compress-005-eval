"""Sample fixed-length RGB frame tensors from FFV1/MKV clips."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def sample_video_tensor(
    path: str | Path,
    *,
    num_frames: int = 16,
    img_size: int = 128,
) -> torch.Tensor:
    """Return float tensor (T, C, H, W) in [0, 1]."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    if total > 0:
        for i in _uniform_indices(total, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(fr)
    if len(frames) < num_frames:
        # Sequential fallback / fill
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        seq: list[np.ndarray] = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            seq.append(fr)
        if seq:
            frames = [seq[i] for i in _uniform_indices(len(seq), num_frames)]
    cap.release()

    if not frames:
        return torch.zeros(num_frames, 3, img_size, img_size)

    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]

    out = []
    for fr in frames:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        out.append(rgb.astype(np.float32) / 255.0)
    arr = np.stack(out, axis=0)  # T,H,W,C
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _uniform_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return [0] * n
    if total == 1:
        return [0] * n
    return [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
