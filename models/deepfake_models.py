import torch
import torch.nn as nn
import torchvision.models.video as video_models

class R3DDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(R3DDeepfakeModel, self).__init__()
        # Load pretrained R3D-18 (trained on Kinetics-400)
        self.model = video_models.r3d_18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: (B, C=3, T, H, W)
        return self.model(x)
