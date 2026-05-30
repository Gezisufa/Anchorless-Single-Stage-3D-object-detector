# validate.py
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import VoxelDetDataset
from production.model import VoxelOneStage3D
from training.assigner import VoxelAssigner


# ----------------------------
# Utils
# ----------------------------
def draw_bbox_3d(ax, cx, cy, cz, sx, sy, sz, color="lime", lw=1.5):
    x1, x2 = cx - sx / 2, cx + sx / 2
    y1, y2 = cy - sy / 2, cy + sy / 2
    z1, z2 = cz - sz / 2, cz + sz / 2

    corners = np.array([
        [x1, y1, z1],
        [x2, y1, z1],
        [x2, y2, z1],
        [x1, y2, z1],
        [x1, y1, z2],
        [x2, y1, z2],
        [x2, y2, z2],
        [x1, y2, z2],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            linewidth=lw,
        )


@torch.no_grad()
def box_iou_3d_xyzxyz(a, b):
    """
    a: (N,6) [x1,y1,z1,x2,y2,z2]
    b: (N,6) [x1,y1,z1,x2,y2,z2]
    returns IoU: (N,)
    """
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
def decode_predictions_one_stage(out, assigner: VoxelAssigner, score_thresh=0.3):
    """
    Decode dense one-stage output into a list of detections:
      each: dict(cx,cy,cz,sx,sy,sz, score, cls)

    Strategy:
      - threshold objectness
      - for each active cell: decode center/size + argmax class
      - no NMS
    """
    obj = out["obj_logit"].sigmoid()[0, 0]     # (Gz,Gx,Gy)
    dxyz = out["dxyz"][0]                      # (3,Gz,Gx,Gy)
    size = out["size"][0]                      # (3,Gz,Gx,Gy)
    cls_logit = out["cls_logit"][0]            # (K,Gz,Gx,Gy)
    cls_pred = cls_logit.argmax(dim=0)         # (Gz,Gx,Gy) long

    stride = float(assigner.grid_stride)

    dets = []
    idx = (obj > score_thresh).nonzero(as_tuple=False)  # (M,3) gz,gx,gy
    for gz, gx, gy in idx:
        cell_cx = (gx.float() + 0.5) * stride
        cell_cy = (gy.float() + 0.5) * stride
        cell_cz = (gz.float() + 0.5) * stride

        cx = cell_cx + dxyz[0, gz, gx, gy]
        cy = cell_cy + dxyz[1, gz, gx, gy]
        cz = cell_cz + dxyz[2, gz, gx, gy]

        sx = size[0, gz, gx, gy]
        sy = size[1, gz, gx, gy]
        sz = size[2, gz, gx, gy]

        dets.append({
            "cx": float(cx.item()),
            "cy": float(cy.item()),
            "cz": float(cz.item()),
            "sx": float(sx.item()),
            "sy": float(sy.item()),
            "sz": float(sz.item()),
            "score": float(obj[gz, gx, gy].item()),
            "cls": int(cls_pred[gz, gx, gy].item()),
            "cell": (int(gz.item()), int(gx.item()), int(gy.item())),
        })

    return dets


@torch.no_grad()
def per_gt_cell_metrics(model, ds: VoxelDetDataset, assigner: VoxelAssigner, device, score_thresh=0.2):
    """
    Like your training metric: for each GT, read prediction from the GT-assigned cell.
    Reports:
      - mean_iou (only GTs where obj score passes thresh)
      - cls_acc  (same subset)
      - num_used / num_gt
    """
    model.eval()

    stride = float(assigner.grid_stride)
    Gz, Gx, Gy = assigner.grid_size

    all_ious = []
    all_cls_ok = []
    num_gt = 0
    num_used = 0

    for i in range(len(ds)):
        x, targets, _ = ds[i]
        x = x.unsqueeze(0).to(device)

        out = model(x)
        obj_prob = out["obj_logit"].sigmoid()[0, 0]   # (Gz,Gx,Gy)
        dxyz = out["dxyz"][0]                         # (3,Gz,Gx,Gy)
        size = out["size"][0]                         # (3,Gz,Gx,Gy)
        cls_pred = out["cls_logit"][0].argmax(dim=0)  # (Gz,Gx,Gy)

        t = targets.to(device)
        if t.numel() == 0:
            continue

        cls_gt = t[:, 0].long()
        centers = t[:, 1:4]
        sizes_gt = t[:, 4:7]
        num_gt += int(t.shape[0])

        g = torch.floor(centers / stride).long()
        gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]

        valid = (
            (gz >= 0) & (gz < Gz) &
            (gx >= 0) & (gx < Gx) &
            (gy >= 0) & (gy < Gy)
        )
        if not valid.any():
            continue

        cls_gt = cls_gt[valid]
        centers = centers[valid]
        sizes_gt = sizes_gt[valid]
        gx, gy, gz = gx[valid], gy[valid], gz[valid]

        # cell centers
        cell_cx = (gx.float() + 0.5) * stride
        cell_cy = (gy.float() + 0.5) * stride
        cell_cz = (gz.float() + 0.5) * stride

        # keep only if objectness passes threshold
        p_obj = obj_prob[gz, gx, gy]
        keep = p_obj > score_thresh
        if not keep.any():
            continue

        cls_gt = cls_gt[keep]
        centers = centers[keep]
        sizes_gt = sizes_gt[keep]
        gx, gy, gz = gx[keep], gy[keep], gz[keep]
        cell_cx, cell_cy, cell_cz = cell_cx[keep], cell_cy[keep], cell_cz[keep]

        # decode prediction from GT cell
        p_dx = dxyz[0, gz, gx, gy]
        p_dy = dxyz[1, gz, gx, gy]
        p_dz = dxyz[2, gz, gx, gy]

        p_sx = size[0, gz, gx, gy]
        p_sy = size[1, gz, gx, gy]
        p_sz = size[2, gz, gx, gy]

        centers_pred = torch.stack([cell_cx + p_dx, cell_cy + p_dy, cell_cz + p_dz], dim=1)
        sizes_pred = torch.stack([p_sx, p_sy, p_sz], dim=1)

        gt_box = torch.cat([centers - sizes_gt / 2.0, centers + sizes_gt / 2.0], dim=1)
        pr_box = torch.cat([centers_pred - sizes_pred / 2.0, centers_pred + sizes_pred / 2.0], dim=1)

        iou = box_iou_3d_xyzxyz(gt_box, pr_box)  # (N,)
        all_ious.append(iou.detach().cpu())

        p_cls = cls_pred[gz, gx, gy]
        cls_ok = (p_cls == cls_gt).detach().cpu().float()
        all_cls_ok.append(cls_ok)

        num_used += int(iou.numel())

    if len(all_ious) == 0:
        return {
            "mean_iou": 0.0,
            "cls_acc": 0.0,
            "num_used": int(num_used),
            "num_gt": int(num_gt),
        }

    all_ious = torch.cat(all_ious, dim=0)
    all_cls_ok = torch.cat(all_cls_ok, dim=0)

    return {
        "mean_iou": float(all_ious.mean().item()),
        "cls_acc": float(all_cls_ok.mean().item()),
        "num_used": int(all_ious.numel()),
        "num_gt": int(num_gt),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path("dataset")
    ckpt_path = Path("checkpoints/best.pt")  # one-stage best

    # IMPORTANT: must match training
    num_classes = 10

    score_thresh = 0.9
    max_samples = 5  # how many samples to visualize

    out_dir = Path("val_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # Load samples (NO shuffle)
    # ----------------------------------
    all_samples = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(all_samples) == 0:
        raise RuntimeError(f"No samples found in {root}")

    n_val = max(1, int(0.2 * len(all_samples)))
    val_samples = all_samples[:n_val][:max_samples]

    ds = VoxelDetDataset(val_samples)

    # ----------------------------------
    # Load model
    # ----------------------------------
    model = VoxelOneStage3D(
        num_classes=num_classes,
        grid_stride=10,
        base_size_vox=30.0,
        stem_channels=16,
        backbone_channels=64,
        head_channels=64,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    assigner = VoxelAssigner(grid_stride=10, grid_size=(50, 50, 50), ignore_index=-1)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Visualizing {len(ds)} samples -> {out_dir}")

    # ----------------------------------
    # Print quick quantitative metrics
    # ----------------------------------
    metrics = per_gt_cell_metrics(model, ds, assigner, device=device, score_thresh=score_thresh)
    print(
        f"[GT-cell metric @ obj>{score_thresh:.2f}] "
        f"mean_iou={metrics['mean_iou']:.4f}  cls_acc={metrics['cls_acc']:.4f}  "
        f"used={metrics['num_used']}/{metrics['num_gt']}"
    )

    # ----------------------------------
    # Loop samples and visualize
    # ----------------------------------
    for i in range(len(ds)):
        x, targets, name = ds[i]
        x_b = x.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x_b)

        dets = decode_predictions_one_stage(out, assigner, score_thresh=score_thresh)

        # ----------------------------------
        # Plot
        # ----------------------------------
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"{Path(name).name}  |  pred={len(dets)}  thr={score_thresh}")


        vis_text_offset = 10
        # GT boxes (green) + class text
        for t in targets.numpy():
            cls, cx, cy, cz, sx, sy, sz = t
            draw_bbox_3d(ax, cx, cy, cz, sx, sy, sz, color="lime", lw=2.0)
            ax.text(cx, cy, cz - vis_text_offset, f"GT:{int(cls)}", fontsize=9)

        # Pred boxes (red) + class text
        for det in dets:
            draw_bbox_3d(ax, det["cx"], det["cy"], det["cz"], det["sx"], det["sy"], det["sz"], color="red", lw=1.2)
            ax.text(det["cx"], det["cy"], det["cz"] + vis_text_offset, f"P:{det['cls']} ({det['score']:.2f})", fontsize=8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_zlim(0, 500)

        fig.tight_layout()
        fig.savefig(out_dir / f"val_{i:03d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Print per-sample summary (helpful when eyeballing)
        if targets.numel() == 0:
            gt_list = []
        else:
            gt_list = [int(c) for c in targets[:, 0].tolist()]
        pred_list = [(d["cls"], round(d["score"], 3)) for d in dets[:10]]
        print(f"[{i:03d}] {Path(name).name}  GT_cls={gt_list}  pred(top10)={pred_list}")

    print("Done.")


if __name__ == "__main__":
    main()