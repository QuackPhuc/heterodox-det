"""ScaleNet: Continuous Scale-Space Detection model.

Full model: Image → CNN Encoder → SIREN Scale-Space → Extrema → OBB Detections

Objects are detected as extrema in a continuous 3D scale-space (x, y, σ),
where σ is a learned continuous scale parameter. The characteristic scale
σ* directly encodes object size — no discrete feature pyramid needed.
"""

import torch
import torch.nn as nn

from .backbone import ScaleNetBackbone
from models._common import InferenceMixin


class ScaleDetHead(nn.Module):
    """Detection head for ScaleNet.

    For each scale-space extremum, decode OBB parameters using features
    extracted at the detected position and scale.

    The characteristic scale σ* serves as a strong prior for object size:
      w ≈ σ* · learned_ratio_w
      h ≈ σ* · learned_ratio_h
    """

    def __init__(self, feat_channels: int = 128, num_classes: int = 15):
        super().__init__()
        self.num_classes = num_classes

        # Feature enhancement from scale-aware features + peak score
        self.feat_enhance = nn.Sequential(
            nn.Linear(feat_channels + 2, feat_channels),  # +2 for (σ, score)
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )

        # Center offset refinement
        self.center_offset = nn.Sequential(
            nn.Linear(feat_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh(),
        )

        # Size: scale-relative prediction
        # w = σ* · exp(pred_w), h = σ* · exp(pred_h)
        self.size_head = nn.Sequential(
            nn.Linear(feat_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        # Angle
        self.angle_head = nn.Sequential(
            nn.Linear(feat_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # Classification
        self.cls_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 2, num_classes),
        )

        # Confidence
        self.conf_head = nn.Sequential(
            nn.Linear(feat_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        base_feat: torch.Tensor,
        peak_scores: torch.Tensor,
        peak_coords: torch.Tensor,
        peak_scales: torch.Tensor,
        peak_spatial_idx: torch.Tensor,
        img_size: int = 640,
    ) -> dict:
        B, C, H, W = base_feat.shape
        K = peak_coords.shape[1]

        # Extract features at peak positions
        feat_flat = base_feat.reshape(B, C, H * W)  # (B, C, N)
        idx_exp = peak_spatial_idx.unsqueeze(1).expand(B, C, K)
        local_feat = torch.gather(feat_flat, 2, idx_exp).permute(0, 2, 1)  # (B, K, C)

        # Augment with scale and score info
        aug_input = torch.cat(
            [
                local_feat,
                peak_scales,  # (B, K, 1)
                peak_scores.unsqueeze(-1),  # (B, K, 1)
            ],
            dim=-1,
        )

        feat = self.feat_enhance(aug_input)  # (B, K, C)

        # Center offset
        offsets = self.center_offset(feat)  # (B, K, 2) in [-1, 1]
        offset_scale = feat.new_tensor([1.0 / W, 1.0 / H])
        centers = (peak_coords + offsets * offset_scale) * img_size

        # Size: scale-relative
        size_raw = self.size_head(feat)  # (B, K, 2) log scale
        # Object size ∝ characteristic scale σ*
        scale_factor = peak_scales * (img_size / 8.0)  # Convert to pixel units
        wh = scale_factor * size_raw.clamp(-5, 5).exp()  # (B, K, 2)

        # Angle
        angles = self.angle_head(feat)  # (B, K, 1)
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))

        # Classification
        cls_logits = self.cls_head(feat)  # (B, K, num_classes)

        # Confidence
        conf = torch.sigmoid(self.conf_head(feat))  # (B, K, 1)

        return {
            "centers": centers,
            "wh": wh,
            "angles": angles,
            "cls_logits": cls_logits,
            "conf": conf,
        }


class ScaleNet(InferenceMixin, nn.Module):
    """Continuous Scale-Space Detection — scale-covariant object detector.

    Objects exist at a continuous scale σ* ∈ ℝ⁺. Instead of P3/P4/P5,
    a SIREN network produces features at any scale, and extrema in 3D
    scale-space provide both position and characteristic scale.

    Args:
        num_classes: number of object categories
        feat_channels: feature dimension
        num_proposals: max proposals per image
        num_scales: number of scale samples
        sigma_range: (min, max) scale range
        img_size: input image size
    """

    def __init__(
        self,
        num_classes: int = 15,
        feat_channels: int = 128,
        num_proposals: int = 100,
        num_scales: int = 8,
        sigma_range: tuple = (0.5, 8.0),
        img_size: int = 640,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        self.backbone = ScaleNetBackbone(
            feat_channels=feat_channels,
            num_scales=num_scales,
            sigma_range=sigma_range,
            num_proposals=num_proposals,
        )
        self.det_head = ScaleDetHead(
            feat_channels=feat_channels,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor) -> dict:
        bb = self.backbone(images)

        preds = self.det_head(
            bb["base_feat"],
            bb["peak_scores"],
            bb["peak_coords"],
            bb["peak_scales"],
            bb["peak_spatial_idx"],
            self.img_size,
        )
        preds["scale_feats"] = bb["scale_feats"]
        preds["sigmas"] = bb["sigmas"]
        preds["mass"] = bb["peak_scores"]  # compatibility alias

        return preds
