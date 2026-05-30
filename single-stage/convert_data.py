import json
from pathlib import Path
import shutil
import cv2 as cv
import numpy as np

# CONFIG ---------------------------------------
size = (500, 500, 500)     # (Z, X, Y) in mm and also voxel resolution here
input_dir = Path("../data/collage")
output_dir = Path("dataset")
sample_id = 1
# ------------------------------------------------

def save_packed_bool_npz(path: Path, array_bool: np.ndarray, key_name: str):
    """
    Packs boolean array into bits and stores:
      - <key_name> : packed uint8 bit array
      - shape      : original shape

    Unpack using:
      npz = np.load(...)
      packed = npz[key_name]
      shape = tuple(npz["shape"])
      arr = np.unpackbits(packed).astype(bool)[:np.prod(shape)].reshape(shape)
    """
    packed = np.packbits(array_bool.reshape(-1))
    np.savez(path, **{
        key_name: packed,
        "shape": np.array(array_bool.shape, dtype=np.int32)
    })




for mid in sorted(input_dir.iterdir()):
    if not mid.is_dir():
        continue

    path = mid / "global_collage"
    image_folder = path / "voxel-global_collage" / "density"
    
    if not image_folder.exists():
        print(f"[SKIP] Missing voxel folder: {image_folder}")
        continue

    image_paths = sorted(image_folder.glob("*.png"))
    if len(image_paths) == 0:
        print(f"[SKIP] No PNGs found in: {image_folder}")
        continue

    # ======================================================
    # CREATE SAMPLE FOLDER
    # ======================================================
    sample_dir = output_dir / f"sample{sample_id:05d}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing: {sample_dir}")

    # ======================================================
    # LOAD IMAGES AS FULL-RES STACK [500, 500, 500]
    # ======================================================
    images = []
    for p in image_paths:
        im = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to read image: {p}")
        images.append(im)

    input_array = np.stack(images, axis=0)  # [Z, X, Y] if each PNG is [X, Y]


    # quick visual control
    stacked = np.clip(np.sum(input_array, axis=1), 0, 255).astype(np.uint8)
    cv.imwrite(str(sample_dir / "stacked.jpg"), stacked)

    # binarize -> pack -> save npz
    input_bool = input_array > 127
    save_packed_bool_npz(sample_dir / "input.npz", input_bool, key_name="packed")

    # ======================================================
    # LOAD LABELS -> annotation.txt (class, center, size)
    # ======================================================
    scale = 1000

    ann_path = path / "annotation_bolts.json"
    label_data = json.load(open(ann_path, "r"))
    annotations = label_data.get("parts", [])


    ann_txt = sample_dir / "annotation.txt"
    with open(ann_txt, "w") as f:
        for obj in annotations:
            cls = int(obj["class_id"])

            c = obj["bounding_box"]["center"]
            s = obj["bounding_box"]["size"]

            c = [v * scale for v in c]
            s = [v * scale for v in s]

            f.write(
                f"{cls} "
                f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}\n"
            )


    ann_txt_bolts = sample_dir / "annotation_bolts.txt"
    with open(ann_txt_bolts, "w") as f:
        for part in annotations:
            for obj in part["bolts"]:
                cls = int(0)

                c = obj["bounding_box"]["center"]
                s = obj["bounding_box"]["size"]

                c = [v * scale for v in c]
                s = [v * scale for v in s]

                f.write(
                    f"{cls} "
                    f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                    f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}\n"
                )


    sample_id += 1

print("Done.")
