# infer.py
# Universal sliding-window inference on voxel volumes.
# Supports three input formats (auto-detected):
#   - STL file   -> voxelized at 1mm with stl-to-voxel
#   - NPZ file   -> packed binary (convert_data.py format)
#   - PNG folder  -> grayscale slices stacked into volume
#
# Dependencies:
#   pip install numpy-stl stl-to-voxel torch matplotlib opencv-python
#
# Usage:
#   python infer.py
#   (edit CONFIG section below)

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import cv2 as cv
import math
import os
import pandas as pd

from stl import mesh as stl_mesh
import stltovoxel

from production.model import VoxelOneStage3D
from training.assigner import VoxelAssigner
from export_excel import main as export_excel


# ============================================================
# CONFIG
# ============================================================
# INPUT_MODE: choose which input to use
#   "stl"  -> voxelize STL file at 1mm resolution
#   "npz"  -> load packed binary npz (convert_data.py format)
#   "png"  -> load PNG slices from folder
INPUT_MODE   = "stl"

STL_PATH     = Path("in/SG868975R_Suppressed.stl")
NPZ_PATH     = Path("dataset/input.npz")
PNG_DIR      = Path("in/density")

CKPT_PATH    = Path("checkpoints/best.pt")
DENSITY_DIR  = Path("out/density")                   # where to save PNG slices (STL mode only)
OUT_PNG      = Path("out/infer_output.png")

OUT_XLSX     = Path("out/detections_raw.xlsx")        # raw detections for export_excel.py
OUT_REPORT   = Path("out/detections_report.xlsx")     # formatted report

WINDOW_SIZE  = 500                # model expects 500³
OVERLAP      = (450, 450, 450)       # (overlap_z, overlap_x, overlap_y)
THRESHOLD    = 0.2
NMS_IOU_THR  = 0.3                # reserved for future NMS

SAVE_PNGS    = True               # save density slices for visual inspection (STL mode)


# ============================================================
# STAGE 1: STL -> voxel volume
# ============================================================
def get_stl_dimensions(stl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads STL with numpy-stl and returns (dimensions_mm, min_coords, max_coords).
    Assumes STL units are millimeters.
    """
    m = stl_mesh.Mesh.from_file(str(stl_path))

    min_coords = m.vectors.reshape(-1, 3).min(axis=0)
    max_coords = m.vectors.reshape(-1, 3).max(axis=0)
    dimensions = max_coords - min_coords

    return dimensions, min_coords, max_coords


def voxelize_stl(stl_path: Path) -> np.ndarray:
    """
    Voxelizes STL at 1 voxel = 1 mm using stl-to-voxel.

    Returns: np.ndarray (Z, Y, X) with values 0/1 (int8).
             Shape matches ceil(dimensions_mm) per axis.
    """
    dims, mn, mx = get_stl_dimensions(stl_path)
    grid_shape = np.ceil(dims).astype(int)

    print(f"STL bounding box:")
    print(f"  min = ({mn[0]:.1f}, {mn[1]:.1f}, {mn[2]:.1f}) mm")
    print(f"  max = ({mx[0]:.1f}, {mx[1]:.1f}, {mx[2]:.1f}) mm")
    print(f"  size = ({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f}) mm")
    print(f"  voxel grid = ({grid_shape[0]}, {grid_shape[1]}, {grid_shape[2]})")

    # load mesh in stltovoxel format
    m = stl_mesh.Mesh.from_file(str(stl_path))
    org_mesh = np.hstack((
        m.v0[:, np.newaxis],
        m.v1[:, np.newaxis],
        m.v2[:, np.newaxis],
    ))

    vol, scale, shift = stltovoxel.convert_meshes(
        [org_mesh],
        voxel_size=1,        # 1 voxel = 1 mm
        parallel=True,
    )

    # stltovoxel returns (Z, Y, X) internally — transpose to (Z, X, Y)
    # to match convert_data.py convention (PNG stack axis order)
    vol = np.swapaxes(vol, 1, 2)          # (Z, Y, X) -> (Z, X, Y)

    print(f"  voxelized volume shape = {vol.shape} (Z, X, Y)")
    print(f"  filled voxels = {np.count_nonzero(vol)}")

    return vol


def save_density_pngs(vol: np.ndarray, output_dir: Path):
    """
    Saves each Z-slice as a grayscale PNG (white=filled, black=empty).
    Matches the format expected by load_png_slices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # clear old PNGs
    for old in output_dir.glob("*.png"):
        old.unlink()

    n_digits = len(str(vol.shape[0]))

    for z in range(vol.shape[0]):
        # vol values are 0 or 1 (int8), convert to 0/255
        img = (vol[z] * 255).astype(np.uint8)
        fname = output_dir / f"slice_{z:0{n_digits}d}.png"
        cv.imwrite(str(fname), img)

    print(f"Saved {vol.shape[0]} density PNGs to: {output_dir}")


# ============================================================
# STAGE 2: Volume loading (NPZ / PNG / direct numpy)
# ============================================================
def load_packed_npz(path: Path, key_name: str = "packed") -> np.ndarray:
    """
    Loads packed binary NPZ (original dataset format from convert_data.py).
    Fields: packed (uint8 bitarray), shape (3D).
    Returns: np.float32 (Z, X, Y) with values 0/1.
    """
    npz = np.load(path)
    packed = npz[key_name]
    shape = tuple(npz["shape"])

    flat = np.unpackbits(packed).astype(np.float32)
    flat = flat[: int(np.prod(shape))]
    arr = flat.reshape(shape)               # (Z, X, Y) as stored

    return arr


def load_png_slices(folder: Path, threshold: int = 127) -> np.ndarray:
    """
    Fallback: loads PNG slices from folder when volume is not in memory.
    Returns: np.float32 (Z, X, Y) with values 0/1.
    """
    image_paths = sorted(folder.glob("*.png"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No PNG files found in: {folder}")

    images = []
    for p in image_paths:
        im = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to read image: {p}")
        images.append(im)

    stack = np.stack(images, axis=0)
    binary = (stack > threshold).astype(np.float32)
    return binary


def vol_to_float(vol: np.ndarray) -> np.ndarray:
    """
    Converts stltovoxel output (int8, 0/1) to float32 for inference.
    Expects (Z, X, Y) after axis swap.
    """
    return (vol > 0).astype(np.float32)


def load_volume(
    mode: str,
    stl_path: Path = None,
    npz_path: Path = None,
    png_dir: Path = None,
    density_dir: Path = None,
    save_pngs: bool = False,
) -> np.ndarray:
    """
    Loads voxel volume based on explicit mode selection.

    Args:
        mode: "stl", "npz", or "png"

    Returns: np.float32 (Z, X, Y) with values 0/1.
    """
    if mode == "png":
        print(f"[INPUT] PNG folder: {png_dir}")
        volume = load_png_slices(png_dir, threshold=127)
        print(f"  Loaded {volume.shape[0]} slices -> shape={volume.shape}")
        return volume

    if mode == "npz":
        print(f"[INPUT] NPZ file: {npz_path}")
        volume = load_packed_npz(npz_path, key_name="packed")
        print(f"  shape={volume.shape} (Z, X, Y)")
        print(f"  filled voxels={np.count_nonzero(volume > 0.5)}")
        return volume

    if mode == "stl":
        print(f"[INPUT] STL file: {stl_path}")
        vol_raw = voxelize_stl(stl_path)          # (Z, X, Y) int8

        if save_pngs and density_dir is not None:
            save_density_pngs(vol_raw, density_dir)

        return vol_to_float(vol_raw)

    raise ValueError(
        f"Unknown INPUT_MODE: '{mode}'. Expected: 'stl', 'npz', or 'png'"
    )


# ============================================================
# STAGE 3: Sliding window infrastructure
# ============================================================
def compute_window_starts(volume_size: int, window: int, overlap: int) -> List[int]:
    if volume_size <= window:
        return [0]

    stride = window - overlap
    starts = list(range(0, volume_size, stride))

    if starts[-1] + window < volume_size:
        starts.append(volume_size - window)

    return starts


def generate_windows(
    volume_shape: Tuple[int, int, int],
    window: int = 500,
    overlap: Tuple[int, int, int] = (50, 50, 50),
) -> List[Tuple[int, int, int]]:
    sz, sx, sy = volume_shape
    oz, ox, oy = overlap

    starts_z = compute_window_starts(sz, window, oz)
    starts_x = compute_window_starts(sx, window, ox)
    starts_y = compute_window_starts(sy, window, oy)

    windows = []
    for z0 in starts_z:
        for x0 in starts_x:
            for y0 in starts_y:
                windows.append((z0, x0, y0))

    return windows


def extract_window(
    volume: np.ndarray,
    origin: Tuple[int, int, int],
    window: int = 500,
) -> np.ndarray:
    z0, x0, y0 = origin
    sz, sx, sy = volume.shape

    z1 = min(z0 + window, sz)
    x1 = min(x0 + window, sx)
    y1 = min(y0 + window, sy)

    chunk = volume[z0:z1, x0:x1, y0:y1]

    pad_z = window - (z1 - z0)
    pad_x = window - (x1 - x0)
    pad_y = window - (y1 - y0)

    if pad_z > 0 or pad_x > 0 or pad_y > 0:
        chunk = np.pad(chunk, ((0, pad_z), (0, pad_x), (0, pad_y)), mode="constant")

    return chunk


# ============================================================
# STAGE 4: Model inference
# ============================================================
def preprocess_volume(vol: np.ndarray) -> np.ndarray:
    """
    Applies the same axis transform as dataset __getitem__
    to the ENTIRE volume, once. After this, all slicing and
    visualization happen in model-coordinate space.

    Input:  (Z, X, Y) — raw volume as loaded
    Output: (Z, Y, X) flipped — same as model sees
    """
    vol = np.swapaxes(vol, 1, 2)      # (Z, X, Y) -> (Z, Y, X)
    vol = np.flip(vol, axis=2)         # flip X axis
    return vol.copy()                  # contiguous copy


def preprocess_chunk(chunk: np.ndarray, device: str) -> torch.Tensor:
    """
    Volume is already in model space (preprocess_volume applied).
    Just add batch + channel dims.
    """
    x = torch.from_numpy(chunk)
    x = x.unsqueeze(0).unsqueeze(0)    # (1, 1, Z, Y, X)
    return x.to(device)


@torch.no_grad()
def decode_predictions_one_stage(
    out, assigner: VoxelAssigner, score_thresh: float = 0.9
) -> List[Dict]:
    obj = out["obj_logit"].sigmoid()[0, 0]
    dxyz = out["dxyz"][0]
    size = out["size"][0]
    cls_logit = out["cls_logit"][0]
    cls_pred = cls_logit.argmax(dim=0)

    stride = float(assigner.grid_stride)

    dets = []
    idx = (obj > score_thresh).nonzero(as_tuple=False)
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


def to_global_coords(
    dets: List[Dict],
    origin: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    window: int = 500,
) -> List[Dict]:
    """
    Shifts detection centers from local 500³ window coords to global volume coords.
    Filters out detections whose center lands in the zero-padded region
    (i.e. beyond the actual volume boundary).

    origin       = (z0, x0, y0) — window start in voxels
    volume_shape = (Z,  X,  Y)  — actual volume size
    """
    z0, x0, y0 = origin
    sz, sx, sy = volume_shape

    # how many valid (non-padded) voxels in this window per axis
    valid_z = min(z0 + window, sz) - z0
    valid_x = min(x0 + window, sx) - x0
    valid_y = min(y0 + window, sy) - y0

    global_dets = []
    skipped = 0
    for d in dets:
        # check if detection center is inside the valid (non-padded) part
        if d["cx"] >= valid_x or d["cy"] >= valid_y or d["cz"] >= valid_z:
            skipped += 1
            continue

        gd = d.copy()
        gd["cx"] += x0
        gd["cy"] += y0
        gd["cz"] += z0
        gd["window_origin"] = origin
        global_dets.append(gd)

    if skipped > 0:
        print(f"    -> filtered {skipped} detections in padded region")

    return global_dets


# ============================================================
# STAGE 5: NMS / deduplication (STUB)
# ============================================================
def compute_iou_3d(box_a: Dict, box_b: Dict) -> float:
    def overlap_1d(c1, s1, c2, s2):
        lo = max(c1 - s1 / 2, c2 - s2 / 2)
        hi = min(c1 + s1 / 2, c2 + s2 / 2)
        return max(0.0, hi - lo)

    ix = overlap_1d(box_a["cx"], box_a["sx"], box_b["cx"], box_b["sx"])
    iy = overlap_1d(box_a["cy"], box_a["sy"], box_b["cy"], box_b["sy"])
    iz = overlap_1d(box_a["cz"], box_a["sz"], box_b["cz"], box_b["sz"])

    inter = ix * iy * iz
    vol_a = box_a["sx"] * box_a["sy"] * box_a["sz"]
    vol_b = box_b["sx"] * box_b["sy"] * box_b["sz"]
    union = vol_a + vol_b - inter

    return inter / union if union > 0 else 0.0


def nms_3d(dets: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    if len(dets) == 0:
        return dets

    sorted_dets = sorted(dets, key=lambda d: d["score"], reverse=True)

    keep = []
    suppressed = [False] * len(sorted_dets)

    for i in range(len(sorted_dets)):
        if suppressed[i]:
            continue
        keep.append(sorted_dets[i])
        for j in range(i + 1, len(sorted_dets)):
            if suppressed[j]:
                continue
            if sorted_dets[i]["cls"] != sorted_dets[j]["cls"]:
                continue
            if compute_iou_3d(sorted_dets[i], sorted_dets[j]) > iou_threshold:
                suppressed[j] = True

    return keep


def deduplicate(
    all_dets: List[Dict],
    iou_threshold: float = 0.3,
    enabled: bool = False,
) -> List[Dict]:
    if not enabled:
        print(f"[INFO] Deduplication DISABLED — returning all {len(all_dets)} detections")
        return all_dets

    print(f"[INFO] Running 3D NMS (IoU thr={iou_threshold}) on {len(all_dets)} detections ...")
    result = nms_3d(all_dets, iou_threshold=iou_threshold)
    print(f"[INFO] After NMS: {len(result)} detections kept")
    return result


# ============================================================
# Excel export (raw)
# ============================================================
def save_detections_xlsx(dets: List[Dict], out_path: Path):
    """
    Saves raw detections to Excel for downstream processing by export_excel.py.
    """
    rows = []
    for d in dets:
        rows.append({
            "class_id": d["cls"],
            "score": round(d["score"], 4),
            "cx": round(d["cx"], 2),
            "cy": round(d["cy"], 2),
            "cz": round(d["cz"], 2),
            "sx": round(d["sx"], 2),
            "sy": round(d["sy"], 2),
            "sz": round(d["sz"], 2),
        })

    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f"Saved raw detections: {out_path} ({len(rows)} rows)")


# ============================================================
# Visualization
# ============================================================
def draw_bbox_3d(ax, cx, cy, cz, sx, sy, sz, color="red", lw=1.2):
    x1, x2 = cx - sx / 2, cx + sx / 2
    y1, y2 = cy - sy / 2, cy + sy / 2
    z1, z2 = cz - sz / 2, cz + sz / 2

    corners = np.array([
        [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
        [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2],
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
            color=color, linewidth=lw,
        )


def visualize(volume: np.ndarray, dets: List[Dict], out_path: Path, title: str = ""):
    mask = volume > 0.5
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        print("[WARN] no voxels > 0.5 to plot")
        return

    max_points = 200000
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]

    zz = coords[:, 0]
    xx = coords[:, 1]
    yy = coords[:, 2]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title or f"pred={len(dets)}")

    ax.scatter(xx, yy, zz, s=0.5, alpha=0.03, marker=".")

    for i, d in enumerate(dets):
        draw_bbox_3d(ax, d["cx"], d["cy"], d["cz"],
                     d["sx"], d["sy"], d["sz"], color="red", lw=1.0)
        ax.text(d["cx"], d["cy"], d["cz"] + 10,
                f"{i}:{d['cls']}({d['score']:.2f})", fontsize=6)

    sz, sx, sy = volume.shape
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_zlim(0, sz)

    # true proportions — no stretching
    ax.set_box_aspect([sx, sy, sz])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main pipeline
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------
    # STAGE 1: Load volume
    # ----------------------------------------------------------
    print("=" * 60)
    print(f"STAGE 1: Loading volume (mode={INPUT_MODE})")
    print("=" * 60)

    volume = load_volume(
        mode=INPUT_MODE,
        stl_path=STL_PATH,
        npz_path=NPZ_PATH,
        png_dir=PNG_DIR,
        density_dir=DENSITY_DIR,
        save_pngs=SAVE_PNGS,
    )

    print(f"\nVolume loaded: shape={volume.shape} (Z, X, Y)")
    print(f"Filled voxels: {np.count_nonzero(volume > 0.5)}")

    # Apply axis transform once (same as dataset __getitem__)
    volume = preprocess_volume(volume)
    print(f"After preprocess: shape={volume.shape} (Z, Y, X) — model space")

    # ----------------------------------------------------------
    # STAGE 2: Generate sliding windows
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Sliding windows")
    print("=" * 60)

    windows = generate_windows(volume.shape, window=WINDOW_SIZE, overlap=OVERLAP)
    print(f"Volume shape: {volume.shape}")
    print(f"Windows: {len(windows)}  (size={WINDOW_SIZE}, overlap={OVERLAP})")

    for i, (z0, x0, y0) in enumerate(windows):
        print(f"  [{i}] origin=({z0}, {x0}, {y0})  "
              f"-> Z[{z0}:{z0+WINDOW_SIZE}] X[{x0}:{x0+WINDOW_SIZE}] Y[{y0}:{y0+WINDOW_SIZE}]")

    # ----------------------------------------------------------
    # STAGE 3: Load model
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3: Loading model")
    print("=" * 60)

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
    print(f"Loaded: {CKPT_PATH}")

    assigner = VoxelAssigner(grid_stride=10, grid_size=(50, 50, 50), ignore_index=-1)

    # ----------------------------------------------------------
    # STAGE 4: Infer each window
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: Inference")
    print("=" * 60)

    all_dets: List[Dict] = []

    for i, origin in enumerate(windows):
        chunk = extract_window(volume, origin, window=WINDOW_SIZE)
        x = preprocess_chunk(chunk, device)

        with torch.no_grad():
            out = model(x)

        local_dets = decode_predictions_one_stage(out, assigner, score_thresh=THRESHOLD)
        global_dets = to_global_coords(local_dets, origin, volume.shape, window=WINDOW_SIZE)
        all_dets.extend(global_dets)

        print(f"  Window [{i}] origin={origin}: {len(local_dets)} detections")

    print(f"\nTotal raw detections: {len(all_dets)}")

    # ----------------------------------------------------------
    # STAGE 5: Deduplicate
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 5: Deduplication")
    print("=" * 60)

    final_dets = deduplicate(
        all_dets,
        iou_threshold=NMS_IOU_THR,
        enabled=True,               # <- flip to True when ready
    )

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(final_dets)} detections")
    print("=" * 60)

    for i, d in enumerate(final_dets):
        print(
            f"[{i:03d}] cls={d['cls']} score={d['score']:.3f} "
            f"cx={d['cx']:.1f} cy={d['cy']:.1f} cz={d['cz']:.1f} "
            f"sx={d['sx']:.1f} sy={d['sy']:.1f} sz={d['sz']:.1f} "
            f"window={d.get('window_origin', '?')}"
        )

    # ----------------------------------------------------------
    # Visualize + export
    # ----------------------------------------------------------
    save_detections_xlsx(final_dets, OUT_XLSX)

    input_name = {
        "stl": STL_PATH.stem,
        "npz": NPZ_PATH.stem,
        "png": PNG_DIR.name,
    }.get(INPUT_MODE, "unknown")

    visualize(
        volume, final_dets, OUT_PNG,
        title=f"{input_name} | {volume.shape} | dets={len(final_dets)}",
    )

    # ----------------------------------------------------------
    # Generate formatted Excel report
    # ----------------------------------------------------------
    export_excel(detections_xlsx=OUT_XLSX, output_xlsx=OUT_REPORT)


if __name__ == "__main__":
    main()
