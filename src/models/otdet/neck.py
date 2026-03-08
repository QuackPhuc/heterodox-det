"""Lightweight Feature Pyramid Network (FPN) neck.

Fuses multi-scale backbone features (C3, C4, C5) into a unified
feature map at stride /8 for the OT module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """Top-down FPN that produces a single fused feature map at stride /8.

    Args:
        in_channels: dict mapping scale name to channel count
        out_channels: number of output feature channels
    """

    def __init__(
        self,
        in_channels: dict = None,
        out_channels: int = 256,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = {"c3": 128, "c4": 256, "c5": 512}

        # Lateral connections (1x1 conv to unify channels)
        self.lat_c5 = nn.Conv2d(in_channels["c5"], out_channels, 1)
        self.lat_c4 = nn.Conv2d(in_channels["c4"], out_channels, 1)
        self.lat_c3 = nn.Conv2d(in_channels["c3"], out_channels, 1)

        # Smooth convolutions after upsampling + addition
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Final fused output
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    def forward(self, features: dict) -> torch.Tensor:
        """Fuse multi-scale features into single stride-/8 map.

        Args:
            features: dict with 'c3', 'c4', 'c5' tensors

        Returns:
            fused: (B, out_channels, H/8, W/8)
        """
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]

        # Top-down pathway
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p4 = self.smooth_p4(p4)
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p3 = self.smooth_p3(p3)

        # Output at stride /8
        fused = self.fuse(p3)
        return fused
