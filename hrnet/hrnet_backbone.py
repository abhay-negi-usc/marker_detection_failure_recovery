import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c1, c2 = channels
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c2, c1, kernel_size=1)
        )

    def forward(self, x1, x2):
        f1 = x1 + self.up2(x2)
        f2 = x2 + self.down1(x1)
        return f1, f2

class HRNetBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.branch1 = BasicBlock(c1, c1)
        self.branch2 = BasicBlock(c2, c2)
        self.fusion = MultiScaleFusion((c1, c2))

    def forward(self, x1, x2):
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        return self.fusion(x1, x2)

class HRNetPose(nn.Module):
    def __init__(self, out_dim=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 256x256 → 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 128x128 → 64x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = BasicBlock(64, 64)
        self.transition = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # HR branch
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # HRNet branches: high-res (32), low-res (64)
        self.branch1 = BasicBlock(32, 32)
        self.branch2_downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.hrnet_block = HRNetBlock(32, 64)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x1 = self.transition(x)
        x2 = self.branch2_downsample(x1)
        x1, x2 = self.hrnet_block(x1, x2)
        x = torch.cat([x1, F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        return self.head(x)
