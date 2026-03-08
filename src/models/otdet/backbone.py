"""ResNet-18 backbone for feature extraction.

Returns multi-scale intermediate features (C3, C4, C5) for use
by the FPN neck.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone returning multi-scale features.

    Output strides: C3 → /8, C4 → /16, C5 → /32
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        resnet = models.resnet18(weights=weights)

        # Stem: conv1 + bn1 + relu + maxpool → stride /4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        # C3: stride /8, 128 channels
        self.layer2 = resnet.layer2
        # C4: stride /16, 256 channels
        self.layer3 = resnet.layer3
        # C5: stride /32, 512 channels
        self.layer4 = resnet.layer4

        # Need layer1 (stride /4, 64 channels) as input to layer2
        self.layer1 = resnet.layer1

        self.out_channels = {"c3": 128, "c4": 256, "c5": 512}

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (B, 3, H, W)

        Returns:
            dict with 'c3', 'c4', 'c5' feature maps
        """
        x = self.stem(x)  # (B, 64, H/4, W/4)
        x = self.layer1(x)  # (B, 64, H/4, W/4)
        c3 = self.layer2(x)  # (B, 128, H/8, W/8)
        c4 = self.layer3(c3)  # (B, 256, H/16, W/16)
        c5 = self.layer4(c4)  # (B, 512, H/32, W/32)
        return {"c3": c3, "c4": c4, "c5": c5}
