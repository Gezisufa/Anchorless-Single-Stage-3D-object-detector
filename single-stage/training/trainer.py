# training/trainer.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader

from training.assigner import VoxelAssigner
from training.losses import VoxelOneStageLoss


@dataclass
class TrainStats:
    loss_total: float
    loss_obj: float
    loss_cls: float
    loss_offset: float
    loss_size: float
    num_pos: float


class OneStageTrainer:
    """
    Minimal trainer for the one-stage anchorless 3D detector.

    Expects DataLoader batches from VoxelDetDataset.collate_fn.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device | str = "cuda",
        *,
        assigner: Optional[VoxelAssigner] = None,
        criterion: Optional[VoxelOneStageLoss] = None,
        amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        log_every: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = torch.device(device)

        self.assigner = assigner if assigner is not None else VoxelAssigner(grid_stride=10, grid_size=(50, 50, 50))
        self.criterion = criterion if criterion is not None else VoxelOneStageLoss()

        self.amp = bool(amp)
        self.grad_clip_norm = grad_clip_norm
        self.log_every = int(log_every)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp and self.device.type == "cuda"))

        self.model.to(self.device)
        self.criterion.to(self.device)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            return x.to(self.device, non_blocking=True)
        return x.to(self.device)

    def _build_dense_targets(self, raw_targets_list, device) -> Dict[str, torch.Tensor]:
        dense_list = [self.assigner(t.to(device), device=device) for t in raw_targets_list]
        dense = {
            "obj_logit": torch.stack([d["obj_logit"] for d in dense_list], dim=0),
            "dxyz": torch.stack([d["dxyz"] for d in dense_list], dim=0),
            "size": torch.stack([d["size"] for d in dense_list], dim=0),
            "cls": torch.stack([d["cls"] for d in dense_list], dim=0),
            "pos_mask": torch.stack([d["pos_mask"] for d in dense_list], dim=0),
        }
        return dense

    def train_one_epoch(self, epoch: int = 0) -> TrainStats:
        self.model.train()

        running_total = 0.0
        running_obj = 0.0
        running_cls = 0.0
        running_off = 0.0
        running_size = 0.0
        running_pos = 0.0
        n_steps = 0

        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            x = self._to_device(batch["x"])

            raw_list = batch["targets"]["raw"]
            dense_targets = self._build_dense_targets(raw_list, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.scaler.is_enabled())):
                preds = self.model(x)
                loss_dict = self.criterion(preds, dense_targets)
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            n_steps += 1
            running_total += float(loss.detach().item())
            running_obj += float(loss_dict["obj"].item())
            running_cls += float(loss_dict["cls"].item())
            running_off += float(loss_dict["offset"].item())
            running_size += float(loss_dict["size"].item())
            running_pos += float(loss_dict["num_pos"].item())

            if self.log_every > 0 and (step % self.log_every == 0):
                dt = time.time() - t0
                avg_total = running_total / max(1, n_steps)
                avg_obj = running_obj / max(1, n_steps)
                avg_cls = running_cls / max(1, n_steps)
                avg_off = running_off / max(1, n_steps)
                avg_size = running_size / max(1, n_steps)
                avg_pos = running_pos / max(1, n_steps)
                print(
                    f"[epoch {epoch:03d} step {step:05d}] "
                    f"loss={avg_total:.4f} (obj={avg_obj:.4f} cls={avg_cls:.4f} "
                    f"off={avg_off:.4f} size={avg_size:.4f}) "
                    f"pos/crop={avg_pos:.2f}  dt={dt:.1f}s"
                )

        avg_total = running_total / max(1, n_steps)
        avg_obj = running_obj / max(1, n_steps)
        avg_cls = running_cls / max(1, n_steps)
        avg_off = running_off / max(1, n_steps)
        avg_size = running_size / max(1, n_steps)
        avg_pos = running_pos / max(1, n_steps)

        return TrainStats(
            loss_total=avg_total,
            loss_obj=avg_obj,
            loss_cls=avg_cls,
            loss_offset=avg_off,
            loss_size=avg_size,
            num_pos=avg_pos,
        )

    @torch.no_grad()
    def validate_one_epoch(self, val_loader: DataLoader, epoch: int = 0) -> TrainStats:
        self.model.eval()

        running_total = 0.0
        running_obj = 0.0
        running_cls = 0.0
        running_off = 0.0
        running_size = 0.0
        running_pos = 0.0
        n_steps = 0

        for step, batch in enumerate(val_loader):
            x = self._to_device(batch["x"])
            raw_list = batch["targets"]["raw"]
            dense_targets = self._build_dense_targets(raw_list, device=self.device)

            preds = self.model(x)
            loss_dict = self.criterion(preds, dense_targets)

            n_steps += 1
            running_total += float(loss_dict["total"].item())
            running_obj += float(loss_dict["obj"].item())
            running_cls += float(loss_dict["cls"].item())
            running_off += float(loss_dict["offset"].item())
            running_size += float(loss_dict["size"].item())
            running_pos += float(loss_dict["num_pos"].item())

        avg_total = running_total / max(1, n_steps)
        avg_obj = running_obj / max(1, n_steps)
        avg_cls = running_cls / max(1, n_steps)
        avg_off = running_off / max(1, n_steps)
        avg_size = running_size / max(1, n_steps)
        avg_pos = running_pos / max(1, n_steps)

        print(
            f"[val epoch {epoch:03d}] loss={avg_total:.4f} "
            f"(obj={avg_obj:.4f} cls={avg_cls:.4f} off={avg_off:.4f} size={avg_size:.4f}) "
            f"pos/crop={avg_pos:.2f}"
        )

        return TrainStats(
            loss_total=avg_total,
            loss_obj=avg_obj,
            loss_cls=avg_cls,
            loss_offset=avg_off,
            loss_size=avg_size,
            num_pos=avg_pos,
        )

    def save_checkpoint(self, path: str, epoch: int, extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=strict)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        return int(ckpt.get("epoch", 0))