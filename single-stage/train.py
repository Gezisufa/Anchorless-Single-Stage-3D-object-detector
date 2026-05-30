# train.py
import os
import time
import random
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VoxelDetDataset

from production.model import VoxelOneStage3D

from training.trainer import OneStageTrainer
from training.assigner import VoxelAssigner
from training.losses import VoxelOneStageLoss


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("one_stage_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@torch.no_grad()
def box_iou_3d_xyzxyz(a, b):
    ix1 = torch.maximum(a[:, 0], b[:, 0])
    iy1 = torch.maximum(a[:, 1], b[:, 1])
    iz1 = torch.maximum(a[:, 2], b[:, 2])

    ix2 = torch.minimum(a[:, 3], b[:, 3])
    iy2 = torch.minimum(a[:, 4], b[:, 4])
    iz2 = torch.minimum(a[:, 5], b[:, 5])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0) * (iz2 - iz1).clamp(min=0)

    va = (a[:, 3] - a[:, 0]).clamp(min=0) * (a[:, 4] - a[:, 1]).clamp(min=0) * (a[:, 5] - a[:, 2]).clamp(min=0)
    vb = (b[:, 3] - b[:, 0]).clamp(min=0) * (b[:, 4] - b[:, 1]).clamp(min=0) * (b[:, 5] - b[:, 2]).clamp(min=0)

    union = (va + vb - inter).clamp(min=1e-6)
    return inter / union


@torch.no_grad()
def validation_mean_iou(model, val_loader, assigner, device, score_thresh=0.2):
    model.eval()
    all_ious = []
    total_used = 0

    for batch in val_loader:
        x = batch["x"].to(device, non_blocking=True)
        raw_list = batch["targets"]["raw"]

        out = model(x)
        obj_prob = out["obj_logit"].sigmoid()
        dxyz = out["dxyz"]
        size = out["size"]

        B = x.shape[0]
        Gz, Gx, Gy = assigner.grid_size
        stride = assigner.grid_stride

        for b in range(B):
            t = raw_list[b].to(device)
            if t.numel() == 0:
                continue

            gt = t[:, 1:]
            centers = gt[:, 0:3]
            sizes_gt = gt[:, 3:6]

            g = torch.floor(centers / stride).long()
            gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]

            valid = (
                (gz >= 0) & (gz < Gz) &
                (gx >= 0) & (gx < Gx) &
                (gy >= 0) & (gy < Gy)
            )
            if not valid.any():
                continue

            gx, gy, gz = gx[valid], gy[valid], gz[valid]
            centers = centers[valid]
            sizes_gt = sizes_gt[valid]

            cell_cx = (gx.float() + 0.5) * stride
            cell_cy = (gy.float() + 0.5) * stride
            cell_cz = (gz.float() + 0.5) * stride

            p_obj = obj_prob[b, 0, gz, gx, gy]
            keep = p_obj > score_thresh
            if not keep.any():
                continue

            gx = gx[keep]
            gy = gy[keep]
            gz = gz[keep]
            centers = centers[keep]
            sizes_gt = sizes_gt[keep]

            cell_cx = cell_cx[keep]
            cell_cy = cell_cy[keep]
            cell_cz = cell_cz[keep]

            p_dx = dxyz[b, 0, gz, gx, gy]
            p_dy = dxyz[b, 1, gz, gx, gy]
            p_dz = dxyz[b, 2, gz, gx, gy]

            p_sx = size[b, 0, gz, gx, gy]
            p_sy = size[b, 1, gz, gx, gy]
            p_sz = size[b, 2, gz, gx, gy]

            centers_pred = torch.stack([cell_cx + p_dx, cell_cy + p_dy, cell_cz + p_dz], dim=1)
            sizes_pred = torch.stack([p_sx, p_sy, p_sz], dim=1)

            gt_box = torch.cat([centers - sizes_gt / 2.0, centers + sizes_gt / 2.0], dim=1)
            pr_box = torch.cat([centers_pred - sizes_pred / 2.0, centers_pred + sizes_pred / 2.0], dim=1)

            iou = box_iou_3d_xyzxyz(gt_box, pr_box)
            all_ious.append(iou.detach().cpu())
            total_used += int(iou.numel())

    if len(all_ious) == 0:
        return {"mean_iou": 0.0, "num_boxes": total_used}

    all_ious = torch.cat(all_ious, dim=0)
    return {"mean_iou": float(all_ious.mean().item()), "num_boxes": int(all_ious.numel())}


def main():
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path("dataset")
    log_path = "logs/log.txt"
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1000
    batch_size = 8
    lr = 1e-4
    num_workers = 2

    # IMPORTANT: set this
    num_classes = 10  # <-- change to your K

    logger = setup_logger(log_path)

    logger.info("==== ONE-STAGE TRAIN START ====")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, batch_size={batch_size}, lr={lr}, num_classes={num_classes}")

    all_samples = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(all_samples) == 0:
        raise RuntimeError(f"No samples found in {root}")

    n_val = max(1, int(0.2 * len(all_samples)))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    logger.info(f"Samples: train={len(train_samples)} val={len(val_samples)}")

    ds_train = VoxelDetDataset(train_samples)
    ds_val = VoxelDetDataset(val_samples)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=VoxelDetDataset.collate_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=VoxelDetDataset.collate_fn,
    )

    model = VoxelOneStage3D(
        num_classes=num_classes,
        grid_stride=10,
        base_size_vox=30.0,
        stem_channels=16,
        backbone_channels=64,
        head_channels=64,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    assigner = VoxelAssigner(grid_stride=10, grid_size=(50, 50, 50), ignore_index=-1)
    criterion = VoxelOneStageLoss(
        obj_weight=1.0,
        cls_weight=1.0,   # <-- tune this
        offset_weight=1.0,
        size_weight=1.0,
        neg_pos_ratio=5.0,
        ignore_index=-1,
    )

    trainer = OneStageTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=dl_train,
        device=device,
        assigner=assigner,
        criterion=criterion,
        amp=True,
        grad_clip_norm=1.0,
        log_every=5,
    )

    best_iou = -1.0

    for epoch in range(epochs):
        logger.info(f"--- Epoch {epoch} ---")

        train_stats = trainer.train_one_epoch(epoch=epoch)
        logger.info(
            f"TRAIN: loss={train_stats.loss_total:.6f} "
            f"(obj={train_stats.loss_obj:.6f} cls={train_stats.loss_cls:.6f} "
            f"off={train_stats.loss_offset:.6f} size={train_stats.loss_size:.6f}) "
            f"pos/crop={train_stats.num_pos:.2f}"
        )

        val_stats = trainer.validate_one_epoch(dl_val, epoch=epoch)
        logger.info(
            f"VAL:   loss={val_stats.loss_total:.6f} "
            f"(obj={val_stats.loss_obj:.6f} cls={val_stats.loss_cls:.6f} "
            f"off={val_stats.loss_offset:.6f} size={val_stats.loss_size:.6f}) "
            f"pos/crop={val_stats.num_pos:.2f}"
        )

        val_metric = validation_mean_iou(model, dl_val, assigner=assigner, device=device, score_thresh=0.2)
        logger.info(
            f"VAL METRIC: mean_iou={val_metric['mean_iou']:.4f} "
            f"(num_boxes_used={val_metric['num_boxes']})"
        )

        if val_metric["mean_iou"] > best_iou:
            best_iou = val_metric["mean_iou"]
            best_path = ckpt_dir / "best.pt"
            trainer.save_checkpoint(str(best_path), epoch=epoch, extra={"val_mean_iou": best_iou})
            logger.info(f"✅ New best: mean_iou={best_iou:.4f} -> saved {best_path}")

    logger.info("==== ONE-STAGE TRAIN DONE ====")


if __name__ == "__main__":
    main()