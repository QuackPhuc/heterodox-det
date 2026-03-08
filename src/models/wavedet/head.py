"""Wave Resonance Detection Head.

Extracts objects from resonance energy maps:
  - Energy peaks → object candidates
  - Spectral features → object scale
  - Feature aggregation → class + OBB parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonancePeakProposer(nn.Module):
    """Propose object candidates from wave energy peaks.

    Uses learned scoring on energy + spectral features to identify
    resonance regions, then extracts top-K candidates.
    """

    def __init__(self, feat_channels: int = 128, num_proposals: int = 100):
        super().__init__()
        self.num_proposals = num_proposals

        self.score_net = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_feat: torch.Tensor) -> tuple:
        """Score all spatial positions and select top-K.

        Args:
            fused_feat: (B, C, H, W) fused wave features

        Returns:
            scores:    (B, K) — candidate scores
            indices:   (B, K) — flat spatial indices
            coords:    (B, K, 2) — normalized (x, y) coordinates
        """
        B, C, H, W = fused_feat.shape
        N = H * W
        K = min(self.num_proposals, N)

        scores_map = self.score_net(fused_feat)  # (B, 1, H, W)
        scores_flat = scores_map.reshape(B, N)  # (B, N)

        # Top-K selection
        top_scores, top_indices = torch.topk(scores_flat, K, dim=1)  # (B, K)

        # Convert flat indices to (x, y) coordinates (normalized)
        ys = (top_indices // W).float() / H
        xs = (top_indices % W).float() / W
        coords = torch.stack([xs, ys], dim=-1)  # (B, K, 2)

        return top_scores, top_indices, coords


class WaveDetHead(nn.Module):
    """Detection head for WaveDetNet.

    For each proposal:
      1. Extract local features via bilinear sampling around the peak
      2. Decode (cx, cy, w, h, angle) offset from peak position
      3. Classify object category
      4. Score confidence from resonance strength + features

    Args:
        feat_channels: feature dimension
        num_classes: number of object categories
        num_proposals: max number of object proposals
    """

    def __init__(
        self, feat_channels: int = 128, num_classes: int = 15, num_proposals: int = 100
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        # Feature extraction using multi-scale pooling around peaks
        self.local_pool = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )

        # Center offset from peak (small refinement)
        self.center_offset = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 2, 2),
            nn.Tanh(),  # offsets in [-1, 1] normalized
        )

        # Size prediction (from spectral features — frequency ∝ 1/scale)
        self.size_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 2, 2),
        )

        # Angle prediction
        self.angle_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 4, 1),
        )

        # Classification
        self.cls_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 2, num_classes),
        )

        # Confidence refinement (combines peak score + features)
        self.conf_head = nn.Sequential(
            nn.Linear(feat_channels + 1, feat_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 4, 1),
        )

    def _extract_features_at_indices(
        self,
        feat_map: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extract feature vectors at flat spatial indices.

        Args:
            feat_map: (B, C, H, W)
            indices:  (B, K) flat indices

        Returns:
            features: (B, K, C)
        """
        B, C, H, W = feat_map.shape
        K = indices.shape[1]

        # Flatten spatial dims and gather
        feat_flat = feat_map.reshape(B, C, H * W)  # (B, C, N)
        indices_exp = indices.unsqueeze(1).expand(B, C, K)  # (B, C, K)
        features = torch.gather(feat_flat, 2, indices_exp)  # (B, C, K)
        return features.permute(0, 2, 1)  # (B, K, C)

    def forward(
        self,
        fused_feat: torch.Tensor,
        peak_scores: torch.Tensor,
        peak_indices: torch.Tensor,
        peak_coords: torch.Tensor,
        img_size: int = 640,
    ) -> dict:
        """Decode detection from resonance peaks.

        Args:
            fused_feat:   (B, C, H, W)
            peak_scores:  (B, K)
            peak_indices: (B, K)
            peak_coords:  (B, K, 2) normalized
            img_size:     image size for denormalization

        Returns:
            dict with 'centers', 'wh', 'angles', 'cls_logits', 'conf'
        """
        # Extract features at peak locations
        local_feat = self._extract_features_at_indices(
            fused_feat, peak_indices
        )  # (B, K, C)
        local_feat = self.local_pool(local_feat)

        # Center = peak_coord + small learned offset
        offsets = self.center_offset(local_feat)  # (B, K, 2) in [-1, 1]
        _, _, H, W = fused_feat.shape
        offset_scale = fused_feat.new_tensor([1.0 / W, 1.0 / H])
        centers = (
            peak_coords + offsets * offset_scale
        ) * img_size  # (B, K, 2) absolute

        # Size from features (exp for positivity)
        wh = self.size_head(local_feat).clamp(-5, 5).exp() * (
            img_size / 10.0
        )  # (B, K, 2)

        # Angle
        angles = self.angle_head(local_feat)  # (B, K, 1)
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))

        # Classification
        cls_logits = self.cls_head(local_feat)  # (B, K, num_classes)

        # Confidence
        conf_input = torch.cat([local_feat, peak_scores.unsqueeze(-1)], dim=-1)
        conf = torch.sigmoid(self.conf_head(conf_input))  # (B, K, 1)

        return {
            "centers": centers,
            "wh": wh,
            "angles": angles,
            "cls_logits": cls_logits,
            "conf": conf,
        }
