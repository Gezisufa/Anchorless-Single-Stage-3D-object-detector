# training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelOneStageLoss(nn.Module):
    def __init__(
        self,
        obj_weight=1.0,
        cls_weight=1.0,
        offset_weight=1.0,
        size_weight=1.0,
        neg_pos_ratio=5.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.obj_weight = float(obj_weight)
        self.cls_weight = float(cls_weight)
        self.offset_weight = float(offset_weight)
        self.size_weight = float(size_weight)
        self.neg_pos_ratio = float(neg_pos_ratio)
        self.ignore_index = int(ignore_index)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smoothl1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, preds, targets):
        """
        preds dict:
            obj_logit: (B,1,Gz,Gx,Gy)
            dxyz:      (B,3,Gz,Gx,Gy)
            size:      (B,3,Gz,Gx,Gy)
            cls_logit: (B,K,Gz,Gx,Gy)

        targets dict:
            obj_logit: (B,1,Gz,Gx,Gy) 0/1
            dxyz:      (B,3,Gz,Gx,Gy)
            size:      (B,3,Gz,Gx,Gy)
            cls:       (B,Gz,Gx,Gy) int64, -1 ignore
            pos_mask:  (B,1,Gz,Gx,Gy) bool
        """
        obj_logit_p = preds["obj_logit"]
        dxyz_p = preds["dxyz"]
        size_p = preds["size"]
        cls_logit_p = preds["cls_logit"]

        obj_logit_t = targets["obj_logit"]
        dxyz_t = targets["dxyz"]
        size_t = targets["size"]
        cls_t = targets["cls"]
        pos_mask = targets["pos_mask"]

        # -------------------------
        # Objectness BCE (with neg downweight)
        # -------------------------
        obj_loss_all = self.bce(obj_logit_p, obj_logit_t)  # (B,1,Gz,Gx,Gy)

        pos = pos_mask
        neg = ~pos

        num_pos = pos.sum().clamp(min=1).float()
        num_neg = neg.sum().float()

        neg_weight = min(self.neg_pos_ratio * num_pos / (num_neg + 1e-6), 1.0)

        obj_loss = (
            obj_loss_all[pos].sum() +
            neg_weight * obj_loss_all[neg].sum()
        ) / num_pos

        # -------------------------
        # Regression losses (pos only)
        # -------------------------
        if pos.any():
            pos3 = pos.expand_as(dxyz_p)
            posS = pos.expand_as(size_p)

            offset_loss = self.smoothl1(dxyz_p[pos3], dxyz_t[pos3]).mean()

            size_loss = self.smoothl1(
                torch.log(size_p[posS] + 1e-6),
                torch.log(size_t[posS] + 1e-6),
            ).mean()
        else:
            offset_loss = torch.tensor(0.0, device=obj_logit_p.device)
            size_loss = torch.tensor(0.0, device=obj_logit_p.device)

        # -------------------------
        # Classification loss (pos only)
        # -------------------------
        if pos.any():
            # pos: (B,1,Gz,Gx,Gy) -> (B,Gz,Gx,Gy)
            pos0 = pos[:, 0]

            # (B,K,Gz,Gx,Gy) -> (B,Gz,Gx,Gy,K) then index positives -> (Npos,K)
            cls_pred = cls_logit_p.permute(0, 2, 3, 4, 1)[pos0]

            cls_target = cls_t[pos0]  # (Npos,)
            # safety: ignore_index should not appear on pos0, but keep it robust
            cls_loss = F.cross_entropy(cls_pred, cls_target, ignore_index=self.ignore_index)
        else:
            cls_loss = torch.tensor(0.0, device=obj_logit_p.device)

        total = (
            self.obj_weight * obj_loss +
            self.cls_weight * cls_loss +
            self.offset_weight * offset_loss +
            self.size_weight * size_loss
        )

        return {
            "total": total,
            "obj": obj_loss.detach(),
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "num_pos": num_pos.detach(),
        }