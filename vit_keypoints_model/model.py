import torch
import torch.nn as nn
import torchvision

class ViTKeypointRegressor(nn.Module):
    def __init__(self, num_keypoints=11**2):
        super(ViTKeypointRegressor, self).__init__()
        self.num_keypoints = num_keypoints

        # Create ViT backbone
        self.vit = torchvision.models.vision_transformer.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()  # Remove default classifier head

        # Add linear head for keypoints
        self.fc = nn.Linear(768, num_keypoints * 2)

    def forward(self, x):
        features = self.vit(x)       # shape: [B, 768]
        out = self.fc(features)      # shape: [B, N*2]
        return out
