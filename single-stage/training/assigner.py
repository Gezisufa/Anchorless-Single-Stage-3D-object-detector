# training/assigner.py
import torch


class VoxelAssigner:
    """
    Builds dense targets aligned with model outputs:

        obj_logit target (0/1):  (1,Gz,Gx,Gy)
        dxyz target (voxels):    (3,Gz,Gx,Gy)
        size target (voxels):    (3,Gz,Gx,Gy)
        cls target (int64):      (Gz,Gx,Gy)  with -1 for ignore/background
        pos_mask (bool):         (1,Gz,Gx,Gy)

    Input GT per object:
        [cls, cx, cy, cz, sx, sy, sz]
    """

    def __init__(self, grid_stride=10, grid_size=(50, 50, 50), ignore_index: int = -1):
        self.grid_stride = float(grid_stride)
        self.grid_size = grid_size  # (Gz,Gx,Gy)
        self.ignore_index = int(ignore_index)

    def __call__(self, targets: torch.Tensor, device):
        """
        targets: (N,7) float tensor, may be empty
        returns dict with target tensors on `device`
        """
        Gz, Gx, Gy = self.grid_size

        obj_t = torch.zeros((1, Gz, Gx, Gy), device=device)
        dxyz_t = torch.zeros((3, Gz, Gx, Gy), device=device)
        size_t = torch.zeros((3, Gz, Gx, Gy), device=device)
        pos_mask = torch.zeros((1, Gz, Gx, Gy), dtype=torch.bool, device=device)

        cls_t = torch.full((Gz, Gx, Gy), fill_value=self.ignore_index, dtype=torch.long, device=device)

        if targets.numel() == 0:
            return {
                "obj_logit": obj_t,
                "dxyz": dxyz_t,
                "size": size_t,
                "cls": cls_t,
                "pos_mask": pos_mask,
            }

        # split
        cls = targets[:, 0].long()
        gt = targets[:, 1:]        # (N,6)
        centers = gt[:, 0:3]       # cx,cy,cz (vox)
        sizes = gt[:, 3:6]         # sx,sy,sz (vox)

        # grid index = floor(center / stride)
        g = torch.floor(centers / self.grid_stride).long()
        gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]

        valid = (
            (gz >= 0) & (gz < Gz) &
            (gx >= 0) & (gx < Gx) &
            (gy >= 0) & (gy < Gy)
        )

        if not valid.any():
            return {
                "obj_logit": obj_t,
                "dxyz": dxyz_t,
                "size": size_t,
                "cls": cls_t,
                "pos_mask": pos_mask,
            }

        gx, gy, gz = gx[valid], gy[valid], gz[valid]
        centers = centers[valid]
        sizes = sizes[valid]
        cls = cls[valid]

        # cell centers in voxels
        cell_cx = (gx.float() + 0.5) * self.grid_stride
        cell_cy = (gy.float() + 0.5) * self.grid_stride
        cell_cz = (gz.float() + 0.5) * self.grid_stride

        dx = centers[:, 0] - cell_cx
        dy = centers[:, 1] - cell_cy
        dz = centers[:, 2] - cell_cz

        # assign 1 GT -> 1 cell (collisions: last one wins)
        obj_t[0, gz, gx, gy] = 1.0
        pos_mask[0, gz, gx, gy] = True

        dxyz_t[0, gz, gx, gy] = dx
        dxyz_t[1, gz, gx, gy] = dy
        dxyz_t[2, gz, gx, gy] = dz

        size_t[0, gz, gx, gy] = sizes[:, 0]
        size_t[1, gz, gx, gy] = sizes[:, 1]
        size_t[2, gz, gx, gy] = sizes[:, 2]

        cls_t[gz, gx, gy] = cls

        return {
            "obj_logit": obj_t,
            "dxyz": dxyz_t,
            "size": size_t,
            "cls": cls_t,
            "pos_mask": pos_mask,
        }