import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models

class RegressorMobileNetV3(nn.Module):
    def __init__(self):
        super(RegressorMobileNetV3, self).__init__()

        # Load MobileNetV3Small as the backbone without the fully connected layers
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove the last fully connected layer

        # Define the fully connected layers
        self.fc1 = nn.Linear(576, 512)  # 1024 is the output size of the MobileNetV3 backbone
        self.prelu1 = nn.PReLU()

        self.fc2 = nn.Linear(512, 512)
        self.prelu2 = nn.PReLU()

        self.fc3 = nn.Linear(512, 512)
        self.prelu3 = nn.PReLU()

        self.fc4 = nn.Linear(512, 256)
        self.prelu4 = nn.PReLU()

        self.output_layer = nn.Linear(256, 2*(11**2))

    def forward(self, x):
        # Pass through the backbone (MobileNetV3)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Fully connected layers with PReLU activation
        x = self.fc1(x)
        x = self.prelu1(x)

        x = self.fc2(x)
        x = self.prelu2(x)

        x = self.fc3(x)
        x = self.prelu3(x)

        x = self.fc4(x)
        x = self.prelu4(x)

        # Output layer
        x = self.output_layer(x) 

        return x

class RegressorMobileNetV3_with_dropouts(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(RegressorMobileNetV3_with_dropouts, self).__init__()

        # Load MobileNetV3Small backbone (without classifier head)
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove classifier

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(576, 512)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(512, 512)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(512, 512)
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.fc4 = nn.Linear(512, 256)
        self.prelu4 = nn.PReLU()
        self.dropout4 = nn.Dropout(p=dropout_p)

        self.output_layer = nn.Linear(256, 2 * (11 ** 2))  # Output 2D keypoints

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout1(self.prelu1(self.fc1(x)))
        x = self.dropout2(self.prelu2(self.fc2(x)))
        x = self.dropout3(self.prelu3(self.fc3(x)))
        x = self.dropout4(self.prelu4(self.fc4(x)))

        return self.output_layer(x)
