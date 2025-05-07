# keypoint_hrnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class HRNetKeypoint(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Example using resnet34 backbone
        from torchvision.models import resnet34
        backbone = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # [B, 512, H/32, W/32]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * 2),
            nn.Sigmoid()  # clamp predictions to [0, 1]
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        return self.head(x)  # → [B, 2K]
    
class HRNetCorners(nn.Module):
    def __init__(self):
        super().__init__()

        # Example using resnet34 backbone
        from torchvision.models import resnet34
        backbone = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # [B, 512, H/32, W/32]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4*2),
            # nn.Sigmoid()  # clamp predictions to [0, 1]
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.head(x)  # → [B, 2K]

class HRNetKeypointHeatmap(nn.Module):
    def __init__(self, num_keypoints=8, heatmap_size=64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        resnet = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, H/32, W/32]

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_keypoints, kernel_size=1)  # K channels = K heatmaps
        )

        self.upsample = nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.upsample(x)  # [B, K, H, W]
