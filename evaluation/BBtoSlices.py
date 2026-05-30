import json
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
 
size = (500, 500)
 
sample_path = Path(r"C:\Users\andriievskyi\git-repo\voxel-nn\data\collage") / "collage_id=0" / "global_collage" # Path("source_data") / "collage_id=2" / "global_collage"
 
 
image_folder = sample_path / "voxel-global_collage" / "density"
images = [cv.imread(str(p), cv.IMREAD_GRAYSCALE) for p in sorted(image_folder.glob("*.png"))]
stacked = np.sum(np.stack(images, axis=0, dtype=np.uint16), axis=0)
stacked = np.clip(stacked, 0, 255).astype(np.uint8)
# cv.imshow("Stacked", stacked)
# cv.waitKey(0)
 
label_data = json.load(open(sample_path / "annotation_bolts.json", "r"))
for item in label_data["annotations"]:
    x, y, z = item["center"]
    cls_id = item["class_id"]
 
 
 
    x, y, z = [float(n)  for n in [x ,y, z]]
 
 
    z_idx = int(z)
    x_idx = int(x)
    y_idx = int(y)
 
    print(x, y, z, cls_id)
 
    # print(item["class_name"], item["center"])
    display = np.zeros((500, 500, 3), dtype="uint8")
 
    idx = z_idx
    ax_x = x_idx
    ax_y = y_idx
 
 
    stacked = np.sum(np.stack(images[idx - 3: idx + 3], axis=0, dtype=np.uint16), axis=0)
    # stacked = np.sum(np.stack(images[0: 500], axis=0, dtype=np.uint16), axis=0)
    stacked = np.clip(stacked, 0, 255).astype(np.uint8)
    display[:,:,0] = stacked
 
 
 
    # cv.drawMarker(display, (ax_x, ax_y), (0, 255, 0), markerSize=10)
    # cv.drawMarker(display, (500 - ax_x, 500 - ax_y), (0, 0, 255), markerSize=10)
    cv.drawMarker(display, (ax_x, 500 - ax_y), (0, 0, 255), markerSize=10)
    # cv.drawMarker(display, (500 - ax_x, ax_y), (0, 0, 255), markerSize=10)
 
    cv.imshow("Image", display)
    cv.waitKey()
 
 
 
 