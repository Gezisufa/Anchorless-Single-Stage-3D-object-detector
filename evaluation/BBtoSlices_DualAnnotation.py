import json
from pathlib import Path
import cv2 as cv
import numpy as np

# Scene size (voxel grid)
SIZE_X = 500
SIZE_Y = 500
SIZE_Z = 500

# Path to collage
base_path = Path(r"C:\Users\andriievskyi\git-repo\voxel-nn\data\collage")
sample_path = base_path / "collage_id=0" / "global_collage"

# Path to slices
image_folder = sample_path / "voxel-global_collage" / "density"

print("Image folder:", image_folder)
print("Exists:", image_folder.exists())

# Load slices
files = sorted(image_folder.glob("*.png"))

if len(files) == 0:
    raise RuntimeError(f"No PNG files found in {image_folder}")

images = []
for p in files:
    img = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read:", p)
        continue
    images.append(img)

if len(images) == 0:
    raise RuntimeError("All images failed to load")

print("Loaded slices:", len(images))

# Load annotation
annotation_path = sample_path / "annotation_bolts.json"
label_data = json.load(open(annotation_path, "r"))

# Iterate through new structure
for part in label_data["parts"]:
    part_label = part["label"]
    class_id = part["class_id"]

    for bolt in part.get("bolts", []):
        bolt_type = bolt["bolt_type"]

        center = bolt["bounding_box"]["center"]
        size = bolt["bounding_box"]["size"]

        # Convert normalized coordinates -> voxel coordinates
        x = int(center[0] * SIZE_X * 2)
        y = int(center[1] * SIZE_Y * 2)
        z = int(center[2] * SIZE_Z * 2)

        print("----")
        print("Part:", part_label)
        print("Bolt:", bolt_type)
        print("Class:", class_id)
        print("Voxel center:", x, y, z)

        # Safety clamp
        z = np.clip(z, 3, len(images) - 3)

        # Stack nearby slices
        stacked = np.sum(
            np.stack(images[z - 3 : z + 3], axis=0, dtype=np.uint16),
            axis=0
        )
        stacked = np.clip(stacked, 0, 255).astype(np.uint8)

        # Prepare RGB display
        display = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)
        display[:, :, 0] = stacked

        # Draw marker
        cv.drawMarker(
            display,
            (x, SIZE_Y - y),
            (0, 0, 255),
            markerSize=12,
            thickness=2
        )

        cv.imshow("Bolt center visualization", display)
        cv.waitKey()

cv.destroyAllWindows()