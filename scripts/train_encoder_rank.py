#!/usr/bin/env python3
"""Train video transformer for encoder-config ranking/classification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from streamenc.ml.encoder_rank.catalog import LABEL_CATALOG_IDS, load_catalog
from streamenc.ml.encoder_rank.dataset import EncoderRankDataset, collate_batch
from streamenc.ml.encoder_rank.model import VideoEncoderRanker, ranking_loss
from streamenc.util.logging import get_logger

log = get_logger("train_encoder_rank")


def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = top3 = n = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            label = batch["label"].to(device)
            out = model(video)
            logits = out["logits"]
            loss = F.cross_entropy(logits, label)
            loss_sum += float(loss.item()) * label.size(0)
            pred = logits.argmax(dim=-1)
            correct += int((pred == label).sum().item())
            topk = logits.topk(min(3, logits.size(-1)), dim=-1).indices
            top3 += int((topk == label.unsqueeze(-1)).any(dim=-1).sum().item())
            n += label.size(0)
    return {
        "loss": loss_sum / max(n, 1),
        "acc1": correct / max(n, 1),
        "acc3": top3 / max(n, 1),
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "encoder_rank"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "encoder_rank_train"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(data_root / "catalog.json")
    n_classes = len(catalog["config_ids"])
    assert n_classes == len(LABEL_CATALOG_IDS)

    train_ds = EncoderRankDataset(data_root, "train", num_frames=args.num_frames, img_size=args.img_size)
    val_ds = EncoderRankDataset(data_root, "val", num_frames=args.num_frames, img_size=args.img_size)
    # If val empty (tiny set), fall back to train holdout indices via test split as val.
    test_ds = EncoderRankDataset(data_root, "test", num_frames=args.num_frames, img_size=args.img_size)
    if len(val_ds) == 0:
        val_ds = test_ds
    log.info("train=%d val=%d test=%d classes=%d device=%s", len(train_ds), len(val_ds), len(test_ds), n_classes, device)
    if len(train_ds) == 0:
        log.error("No train labels yet — run build_encoder_rank_dataset.py first")
        return 1

    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, max(len(val_ds), 1)),
        shuffle=False,
        num_workers=1,
        collate_fn=collate_batch,
    ) if len(val_ds) else None

    model = VideoEncoderRanker(
        n_classes,
        img_size=args.img_size,
        num_frames=args.num_frames,
        dim=args.dim,
        depth=args.depth,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_acc = -1.0
    best_path = out_dir / "best.pt"
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            video = batch["video"].to(device)
            label = batch["label"].to(device)
            target_scores = batch["target_scores"].to(device)
            score_mask = batch["score_mask"].to(device)
            out = model(video)
            ce = F.cross_entropy(out["logits"], label)
            rk = ranking_loss(out["score_logits"], target_scores, score_mask)
            loss = ce + args.rank_weight * rk
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * label.size(0)
            n += label.size(0)
        sched.step()
        train_loss = running / max(n, 1)
        val_metrics = evaluate(model, val_loader, device) if val_loader and len(val_ds) else {"acc1": 0, "acc3": 0, "loss": 0, "n": 0}
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        log.info("epoch %d train_loss=%.4f val_acc1=%.3f val_acc3=%.3f", epoch, train_loss, val_metrics["acc1"], val_metrics["acc3"])
        if val_metrics["acc1"] >= best_acc:
            best_acc = val_metrics["acc1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "catalog_ids": list(LABEL_CATALOG_IDS),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_acc1": best_acc,
                },
                best_path,
            )

    # Always save last
    last_path = out_dir / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "catalog_ids": list(LABEL_CATALOG_IDS),
            "args": vars(args),
            "epoch": args.epochs,
        },
        last_path,
    )
    if not best_path.is_file():
        best_path.write_bytes(last_path.read_bytes())

    # Test eval on best
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loader = DataLoader(
        test_ds if len(test_ds) else train_ds,
        batch_size=min(args.batch_size, max(len(test_ds) or len(train_ds), 1)),
        shuffle=False,
        num_workers=1,
        collate_fn=collate_batch,
    )
    test_metrics = evaluate(model, test_loader, device)
    summary = {
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "best_val_acc1": best_acc,
        "test": test_metrics,
        "epochs": args.epochs,
        "seconds": time.perf_counter() - t0,
        "best_checkpoint": str(best_path),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
