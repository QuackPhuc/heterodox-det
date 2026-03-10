"""OBB Detection Head.

Converts the OT transport plan into object detections:
  - Weighted centroid of pixel coordinates → object center (cx, cy)
  - Weighted feature aggregation → MLP → (w, h, angle, class_logits)
  - Total transported mass → object confidence
"""

import math

import torch
import torch.nn as nn


class OBBHead(nn.Module):
    """Decode transport plan into oriented bounding box predictions.

    For each object slot j, the head:
      1. Computes the π-weighted centroid of pixel spatial coordinates
      2. Aggregates pixel features via the transport plan
      3. Decodes (w, h, angle, class) from aggregated features
      4. Derives confidence from total mass transported to slot j

    Args:
        feat_dim: feature dimension (from FPN)
        num_classes: number of object categories
        num_slots: number of object slots (must match OT module)
    """

    def __init__(
        self, feat_dim: int = 256, num_classes: int = 15, num_slots: int = 100
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots

        # MLP to decode box parameters from aggregated features
        self.box_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
        )
        # Separate outputs for geometry parameters
        self.wh_pred = nn.Linear(feat_dim // 2, 2)  # width, height
        self.angle_pred = nn.Linear(feat_dim // 2, 1)  # angle (radians)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, num_classes),
        )

        # Confidence head (from mass + features)
        self.conf_head = nn.Sequential(
            nn.Linear(feat_dim + 1, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 1),
        )

        # Prior-based initialization for stable early training
        self._init_prior_biases()

    def _init_prior_biases(self):
        """Set output layer biases to domain-aware priors.

        Confidence prior: sigmoid(-2) ≈ 0.12 — most slots are background.
        Class prior: sigmoid(log(π/(1-π))) ≈ π=0.01 — matches focal loss
            assumption that foreground is rare (RetinaNet-style init).
        """
        # Confidence: most slots should start with low confidence
        nn.init.constant_(self.conf_head[-1].bias, -2.0)

        # Class logits: rare foreground assumption (π=0.01 → bias ≈ -4.6)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head[-1].bias, bias_value)

    def forward(
        self,
        transport_plan: torch.Tensor,
        pixel_features: torch.Tensor,
        pixel_coords: torch.Tensor,
        img_size: int = 640,
    ) -> dict:
        """Decode transport plan into detections.

        Args:
            transport_plan: (B, N, K) — OT coupling from Sinkhorn
            pixel_features: (B, N, D) — per-pixel feature vectors
            pixel_coords: (B, N, 2) — per-pixel (x, y) spatial coordinates
            img_size: original image size for denormalization

        Returns:
            dict with:
                'centers':   (B, K, 2) — predicted cx, cy (absolute)
                'wh':        (B, K, 2) — predicted w, h (absolute)
                'angles':    (B, K, 1) — predicted angle (radians)
                'cls_logits':(B, K, num_classes) — class logits
                'conf':      (B, K, 1) — confidence scores
                'mass':      (B, K) — total transported mass per slot
        """
        B, N, K = transport_plan.shape
        D = pixel_features.shape[-1]

        # Normalize transport plan per slot (column-wise) for weighted pooling
        mass = transport_plan.sum(dim=1)  # (B, K) — total mass per slot
        plan_normalized = transport_plan / (mass.unsqueeze(1).clamp(min=1e-8))

        # 1. Weighted centroid for center prediction
        # pixel_coords: (B, N, 2), plan_normalized: (B, N, K)
        # centers = Σᵢ πᵢⱼ · coordᵢ / Σᵢ πᵢⱼ
        centers = torch.bmm(plan_normalized.transpose(1, 2), pixel_coords)  # (B, K, 2)

        # 2. Feature aggregation via transport plan
        # agg_feats = Σᵢ πᵢⱼ · fᵢ / Σᵢ πᵢⱼ
        agg_feats = torch.bmm(
            plan_normalized.transpose(1, 2), pixel_features
        )  # (B, K, D)

        # 3. Decode box parameters
        box_hidden = self.box_head(agg_feats)  # (B, K, D//2)
        wh = (
            self.wh_pred(box_hidden).clamp(-5, 5).exp()
        )  # (B, K, 2) — clamped for stability
        angles = self.angle_pred(box_hidden)  # (B, K, 1)
        # Normalize angle to [-π/2, π/2) via atan2 (NaN-safe)
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))

        # 4. Classification
        cls_logits = self.cls_head(agg_feats)  # (B, K, num_classes)

        # 5. Confidence from mass + feature context
        conf_input = torch.cat([agg_feats, mass.unsqueeze(-1)], dim=-1)  # (B, K, D+1)
        conf = torch.sigmoid(self.conf_head(conf_input))  # (B, K, 1)

        # Scale centers and wh to absolute pixel coordinates
        centers_abs = centers * img_size
        wh_abs = wh * (img_size / 10.0)

        return {
            "centers": centers_abs,
            "wh": wh_abs,
            "angles": angles,
            "cls_logits": cls_logits,
            "conf": conf,
            "mass": mass,
        }
