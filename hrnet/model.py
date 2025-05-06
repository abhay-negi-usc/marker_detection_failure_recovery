import torch
import torch.nn as nn
from hrnet.hrnet_backbone import HRNetPose  # or use relative import: from .hrnet_backbone import HRNetPose

class HRNetSE3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HRNetPose(out_dim=6)

    def forward(self, x):
        return self.backbone(x)
