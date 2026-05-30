# production/model.py
import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    """Conv3d -> GroupNorm -> SiLU"""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, groups=8):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv3d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)

        g = min(groups, c_out)
        while c_out % g != 0:
            g -= 1
        self.gn = nn.GroupNorm(g, c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class VoxelOneStage3D(nn.Module):
    """
    Anchorless, single-stride 3D one-stage detector.

    Input:  (B,1,500,500,500) occupancy {0,1}
    Output: (B, 7+K, 50,50,50)

    Channels:
        0:      objectness logit
        1-3:    center offset dxyz (voxels), bounded to ±stride/2
        4-6:    size (sx,sy,sz) via exp(log_size)*base_size
        7..:    class logits (K classes, integer labels 0..K-1)
    """

    def __init__(
        self,
        num_classes: int,
        grid_stride=10,
        base_size_vox=30.0,
        stem_channels=16,
        backbone_channels=64,
        head_channels=64,
        init_obj_bias=-4.0,  # sigmoid(-4)=0.018
    ):
        super().__init__()
        assert num_classes >= 1, "num_classes must be >= 1"

        self.num_classes = int(num_classes)
        self.grid_stride = int(grid_stride)
        self.base_size_vox = float(base_size_vox)

        # 500^3 -> 50^3 by averaging occupancy density
        self.prepool = nn.AvgPool3d(kernel_size=self.grid_stride, stride=self.grid_stride)

        self.stem = nn.Sequential(
            ConvGNAct(1, stem_channels, k=3),
            ConvGNAct(stem_channels, stem_channels, k=3),
        )

        self.backbone = nn.Sequential(
            ConvGNAct(stem_channels, backbone_channels, k=3),
            ConvGNAct(backbone_channels, backbone_channels, k=3),
            ConvGNAct(backbone_channels, backbone_channels, k=3),
        )

        out_ch = 7 + self.num_classes
        self.head = nn.Sequential(
            ConvGNAct(backbone_channels, head_channels, k=3),
            nn.Conv3d(head_channels, out_ch, kernel_size=1, stride=1, padding=0),
        )

        last = self.head[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.constant_(last.bias, 0.0)

        with torch.no_grad():
            last.bias[0] = float(init_obj_bias)

    def forward(self, x):
        """
        Returns dict:
            obj_logit: (B,1,Gz,Gx,Gy)
            dxyz:      (B,3,Gz,Gx,Gy) in voxels
            size:      (B,3,Gz,Gx,Gy) in voxels
            cls_logit: (B,K,Gz,Gx,Gy)
            raw:       (B,7+K,Gz,Gx,Gy)
        """
        x = self.prepool(x)  # (B,1,50,50,50)
        x = self.stem(x)
        x = self.backbone(x)
        raw = self.head(x)   # (B,7+K,50,50,50)

        obj_logit = raw[:, 0:1]

        dxyz = torch.tanh(raw[:, 1:4]) * (self.grid_stride / 2.0)

        dlog = raw[:, 4:7]
        size = torch.exp(dlog) * self.base_size_vox

        cls_logit = raw[:, 7:7 + self.num_classes]

        return {
            "obj_logit": obj_logit,
            "dxyz": dxyz,
            "size": size,
            "cls_logit": cls_logit,
            "raw": raw,
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    K = 5

    model = VoxelOneStage3D(
        num_classes=K,
        grid_stride=10,
        base_size_vox=30.0,
        stem_channels=16,
        backbone_channels=64,
        head_channels=64,
    ).to(device)

    x = torch.randint(0, 2, (B, 1, 500, 500, 500), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x)

    print("obj_logit:", out["obj_logit"].shape)
    print("dxyz     :", out["dxyz"].shape)
    print("size     :", out["size"].shape)
    print("cls_logit:", out["cls_logit"].shape)
    print("raw      :", out["raw"].shape)