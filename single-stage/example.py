# infer_one_npz_min.py
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from production.model import VoxelOneStage3D
from training.assigner import VoxelAssigner


# ----------------------------
# NPZ -> tensor
# ----------------------------
def load_packed_npz(path: Path, key_name="packed") -> torch.Tensor:
    """
    Loads packed binary npz with fields:
      - packed (uint8)
      - shape  (3D)
    Returns: torch.float32 tensor (Z,X,Y) with values 0/1.
    """
    npz = np.load(path)
    packed = npz[key_name]
    shape = tuple(npz["shape"])

    flat = np.unpackbits(packed).astype(np.float32)
    flat = flat[: int(np.prod(shape))]  # trim padding bits
    arr = flat.reshape(shape)           # (Z,X,Y) as stored

    return torch.from_numpy(arr)        # float32 (0/1)


# ----------------------------
# Vis utils
# ----------------------------
def draw_bbox_3d(ax, cx, cy, cz, sx, sy, sz, color="red", lw=1.2):
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
def decode_predictions_one_stage(out, assigner: VoxelAssigner, score_thresh=0.9):
    """
    Same decode as your validate.py:
      - threshold objectness
      - decode center/size + argmax class
      - no NMS
    Returns list of dicts: cx,cy,cz,sx,sy,sz,score,cls,cell
    """
    obj = out["obj_logit"].sigmoid()[0, 0]     # (Gz,Gx,Gy)
    dxyz = out["dxyz"][0]                      # (3,Gz,Gx,Gy)
    size = out["size"][0]                      # (3,Gz,Gx,Gy)
    cls_logit = out["cls_logit"][0]            # (K,Gz,Gx,Gy)
    cls_pred = cls_logit.argmax(dim=0)         # (Gz,Gx,Gy)

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


def main():
    # ----------------------------
    # Hardcoded paths
    # ----------------------------
    NPZ_PATH = Path("dataset/input.npz")      # <- change if needed
    CKPT_PATH = Path("checkpoints/best.pt")               # <- change if needed
    OUT_PNG = Path("out/infer_one.png")
    THRESHOLD = 0.9

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Load input npz -> tensor
    # ----------------------------
    vol = load_packed_npz(NPZ_PATH, key_name="packed")    # (Z,X,Y) float32 0/1

    # match your dataset __getitem__:
    x = vol.unsqueeze(0)                  # (1,Z,X,Y)
    x = x.permute(0, 1, 3, 2)             # (1,Z,Y,X)  <-- your "fix"
    x = torch.flip(x, dims=(3,))          # flip last dim
    x = x.unsqueeze(0).to(device)         # (B=1,1,Z,*,*) -> here (1,1,Z,Y,X)

    # ----------------------------
    # Load model + checkpoint
    # ----------------------------
    model = VoxelOneStage3D(
        num_classes=10,
        grid_stride=10,
        base_size_vox=30.0,
        stem_channels=16,
        backbone_channels=64,
        head_channels=64,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    assigner = VoxelAssigner(grid_stride=10, grid_size=(50, 50, 50), ignore_index=-1)

    # ----------------------------
    # Infer + decode
    # ----------------------------
    with torch.no_grad():
        out = model(x)

    dets = decode_predictions_one_stage(out, assigner, score_thresh=THRESHOLD)

    # Print boxes to terminal
    print(f"NPZ: {NPZ_PATH}")
    print(f"CKPT: {CKPT_PATH}")
    print(f"Pred count (thr={THRESHOLD}): {len(dets)}")
    for i, d in enumerate(dets):
        print(
            f"[{i:03d}] cls={d['cls']} score={d['score']:.3f} "
            f"cx={d['cx']:.1f} cy={d['cy']:.1f} cz={d['cz']:.1f} "
            f"sx={d['sx']:.1f} sy={d['sy']:.1f} sz={d['sz']:.1f} "
            f"cell={d['cell']}"
        )

    # ----------------------------
    # Plot: voxel cloud + pred boxes
    # ----------------------------
    vol_np = x[0, 0].detach().cpu().numpy()  # (Z, Y, X)
    mask = vol_np > 0.5
    coords = np.argwhere(mask)               # rows are [z, y, x]

    zz = coords[:, 0]
    xx = coords[:, 1]   # IMPORTANT: same as your dataset plotting (treat y-index as X)
    yy = coords[:, 2]   # IMPORTANT: treat x-index as Y

    if coords.shape[0] == 0:
        print("[WARN] no voxels > 0.5 to plot")
        return

    max_points = 100000
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]

    zz = coords[:, 0]
    xx = coords[:, 1]
    yy = coords[:, 2]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{NPZ_PATH.parent.name} | pred={len(dets)}")

    # now cloud matches dataset convention and should align with boxes
    ax.scatter(xx, yy, zz, s=1, alpha=0.05, marker=".")

    vis_text_offset = 10
    for i, d in enumerate(dets):
        draw_bbox_3d(ax, d["cx"], d["cy"], d["cz"], d["sx"], d["sy"], d["sz"], color="red", lw=1.2)
        ax.text(d["cx"], d["cy"], d["cz"] + vis_text_offset, f"{i}:{d['cls']}({d['score']:.2f})", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_zlim(0, 500)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()