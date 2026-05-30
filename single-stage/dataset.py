import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class VoxelDetDataset(Dataset):
    def __init__(self, sample_dirs):
        """
        sample_dirs: list of sample directories, e.g.
        [
            "dataset/sample00001",
            "dataset/sample00007",
            ...
        ]
        """
        self.samples = [Path(s) for s in sample_dirs]

    def __len__(self):
        return len(self.samples)

    def _load_packed_npz(self, path: Path, key_name="packed"):
        """
        Loads packed binary npz with fields:
            - packed (uint8)
            - shape (3D)
        Returns float32 tensor (0/1).
        """
        npz = np.load(path)
        packed = npz[key_name]
        shape = tuple(npz["shape"])

        flat = np.unpackbits(packed).astype(np.float32)
        flat = flat[:np.prod(shape)]  # trim padding
        arr = flat.reshape(shape)

        return torch.from_numpy(arr)  # float32 (0/1)

    def _load_targets(self, ann_path: Path):
        """
        Loads annotation.txt:
            class cx cy cz sx sy sz
        Returns:
            targets: float32 tensor [N, 7]
        """
        if not ann_path.exists():
            # no objects -> return empty tensor
            return torch.zeros((0, 7), dtype=torch.float32)

        rows = []
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                vals = list(map(float, line.split()))
                if len(vals) != 7:
                    raise ValueError(f"Bad annotation line in {ann_path}: {line}")

                rows.append(vals)

        if len(rows) == 0:
            return torch.zeros((0, 7), dtype=torch.float32)

        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # input: (500, 500, 500) float32
        x = self._load_packed_npz(sample_dir / "input.npz", key_name="packed")

        # add channel dim for CNNs: (1, 500, 500, 500)
        x = x.unsqueeze(0)
        x = x.permute(0, 1, 3, 2) # This is fix
        x = torch.flip(x, dims=(3, )) # This is also fix

        # targets: [N, 7]
        targets = self._load_targets(sample_dir / "annotation.txt")

        return x, targets, str(sample_dir)


    @staticmethod
    def collate_fn(batch):
        """
        Keeps variable-length GT boxes as a list.

        Input batch items:
            x:       (1,500,500,500)
            targets: (N,7)  [cls,cx,cy,cz,sx,sy,sz]
            name:    str

        Returns:
            {
              "x": (B,1,500,500,500),
              "targets": {
                 "raw":  [ (Ni,7), ... ],
                 "cls":  [ (Ni,), ... ],
                 "geom": [ (Ni,6), ... ],  # cx,cy,cz,sx,sy,sz
              },
              "names": [str,...]
            }
        """
        xs, targets_list, names = zip(*batch)

        # stack X (all same size)
        x = torch.stack(xs, dim=0)

        targets_raw = []
        targets_cls = []
        targets_geom = []

        for t in targets_list:
            # ensure tensor
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32)

            # t: (N,7) or (0,7)
            targets_raw.append(t)

            if t.numel() == 0:
                targets_cls.append(t.new_zeros((0,), dtype=torch.long))
                targets_geom.append(t.new_zeros((0, 6), dtype=torch.float32))
            else:
                targets_cls.append(t[:, 0].long())     # (N,)
                targets_geom.append(t[:, 1:].float())  # (N,6)



        return {
            "x": x,
            "targets": {
                "raw": targets_raw,
                "cls": targets_cls,
                "geom": targets_geom,
            },
            "names": list(names),
        }




if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  

    root = Path("dataset")
    all_samples = sorted([p for p in root.iterdir() if p.is_dir()])

    print(f"Found {len(all_samples)} samples.")

    #test_samples = random.sample(all_samples, min(2, len(all_samples)))
    test_samples = all_samples[:2]
    print("Testing dataset with samples:")
    for s in test_samples:
        print("  ", s.name)

    ds = VoxelDetDataset(test_samples)

    def draw_bbox_3d(ax, cx, cy, cz, sx, sy, sz, color="lime", lw=1.0):
        """
        Draw axis-aligned 3D bounding box in voxel coords.

        NOTE: volume indexing is (z, x, y)
        but plotting axes will be (x, y, z) because it's more human-friendly.
        """
        x1, x2 = cx - sx / 2, cx + sx / 2
        y1, y2 = cy - sy / 2, cy + sy / 2
        z1, z2 = cz - sz / 2, cz + sz / 2

        # 8 corners in (x,y,z)
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

        # edges between corners
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for a, b in edges:
            xs = [corners[a, 0], corners[b, 0]]
            ys = [corners[a, 1], corners[b, 1]]
            zs = [corners[a, 2], corners[b, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=lw)

    for i in range(len(ds)):
        x, targets, name = ds[i]
        print(f"\nSample: {name}")
        print("  input shape :", tuple(x.shape), x.dtype)
        print("  targets     :", tuple(targets.shape), targets.dtype)

        # -------------------------------------------------------
        # x is torch-like indexing:
        #   x[0, z, x, y]  (because you later do vol = x[0])
        # -------------------------------------------------------
        vol = x[0].numpy()  # (Z, X, Y)
        Z, X, Y = vol.shape

        # -------------------------------------------------------
        # Extract voxel cloud (subsample)
        # -------------------------------------------------------
        mask = vol > 0.5
        coords = np.argwhere(mask)  # (N, 3) with rows: [z, x, y]

        if coords.shape[0] == 0:
            print("  [WARN] No voxels > 0.5 found")
            continue

        # subsample to keep plot usable
        max_points = 100000
        if coords.shape[0] > max_points:
            idxs = np.random.choice(coords.shape[0], max_points, replace=False)
            coords = coords[idxs]

        zz = coords[:, 0]
        xx = coords[:, 1]
        yy = coords[:, 2]

        # -------------------------------------------------------
        # 3D plot
        # -------------------------------------------------------
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"Voxel cloud + targets: {Path(name).name}")

        # voxel cloud: plot as points (x,y,z)
        # NOTE: coords are (z,x,y) -> plot as (x,y,z)
        ax.scatter(
            xx, yy, zz,
            s=1,
            alpha=0.05,   # opacity
            marker="."
        )

        # plot target centers
        for t in targets.numpy():
            cls, cx, cy, cz, sx, sy, sz = t

            # center marker
            ax.scatter(
                [cx], [cy], [cz],
                s=80,
                alpha=1.0,
                marker="o"
            )

            # bbox wireframe (optional)
            draw_bbox_3d(ax, cx, cy, cz, sx, sy, sz, lw=1.2)

            ax.text(cx, cy, cz, f"{int(cls)}", fontsize=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # match volume bounds
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_zlim(0, Z)

        plt.tight_layout()
        plt.savefig(f"dataset_sample_{i}.png")
        plt.close()
