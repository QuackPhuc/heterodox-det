"""TopoNet: Topological Persistence Detection.

Core idea: objects create "topological holes" on the feature manifold.
A learned filtration function defines sublevel sets, and persistent
features (high birth-death persistence) correspond to objects.

Detection = finding persistent topological features via differentiable
persistence-like analysis.

Mathematical foundation:
  Filtration: X_a = {(x,y) : f_θ(x,y) ≤ a}
  As a increases, connected components (H₀) appear and merge.
  Objects = components with high persistence (death - birth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models._common import InferenceMixin


class FiltrationEncoder(nn.Module):
    """Encodes image into a filtration function and feature map.

    The filtration function f(x,y) defines the order in which spatial
    regions "appear" in the sublevel set filtration. Objects should
    appear early (low birth) and persist long (high death).
    """

    def __init__(self, feat_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
        )

        # Filtration function: per-pixel "birth value"
        self.filtration_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
        )

        # Feature map for classification/regression
        self.feat_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
        )

        self.out_channels = feat_channels

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        filtration = self.filtration_head(feat)  # (B, 1, H', W')
        features = self.feat_head(feat)  # (B, C, H', W')
        return filtration, features


class DifferentiablePersistence(nn.Module):
    """Differentiable approximation of persistent homology (H₀).

    Instead of exact persistent homology (non-differentiable, O(n³)),
    uses a soft thresholding approach:

    1. Sample N threshold levels from the filtration range
    2. At each level, compute soft connected components via
       iterative diffusion with thresholded adjacency
    3. Track component birth/death across levels
    4. Components with high persistence = object candidates

    This is an approximation designed to be:
      - Fully differentiable (soft operations)
      - GPU-friendly (parallel across thresholds)
      - O(N·T·K) where N=pixels, T=thresholds, K=diffusion steps
    """

    def __init__(
        self,
        num_levels: int = 16,
        num_diffusion_steps: int = 5,
        persistence_thresh: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_diffusion_steps = num_diffusion_steps
        self.persistence_thresh = persistence_thresh

        # Learnable level spacing (initialized uniform)
        self.level_offsets = nn.Parameter(torch.linspace(0, 1, num_levels))

        # Pre-register constant diffusion kernels as non-persistent buffers
        self.register_buffer(
            "_diff_kernel",
            torch.tensor(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32
            ).reshape(1, 1, 3, 3)
            / 4.0,
            persistent=False,
        )
        self.register_buffer(
            "_count_kernel",
            torch.tensor(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32
            ).reshape(1, 1, 3, 3)
            / 4.0,
            persistent=False,
        )

    def _soft_threshold(
        self, filtration: torch.Tensor, level: float, temperature: float = 5.0
    ) -> torch.Tensor:
        """Soft indicator of sublevel set: σ(temperature * (level - f(x)))."""
        return torch.sigmoid(temperature * (level - filtration))

    def _diffusion_components(
        self, mask: torch.Tensor, feat: torch.Tensor, steps: int = 5
    ) -> torch.Tensor:
        """Approximate connected components via feature diffusion.

        Within the sublevel set (mask), diffuse features so that
        connected regions converge to similar values, while
        disconnected regions remain different.

        Returns per-pixel "component ID" as a continuous vector.
        """
        B, C, H, W = feat.shape

        # Initialize diffusion field with features masked by sublevel set
        field = feat * mask  # (B, C, H, W)
        kernel = self._diff_kernel.expand(C, 1, 3, 3)

        for _ in range(steps):
            neighbors = F.conv2d(field, kernel, padding=1, groups=C)
            neighbor_count = F.conv2d(mask, self._count_kernel, padding=1)

            # Blend current with neighbors (within mask only)
            field = mask * (
                0.5 * field + 0.5 * neighbors / neighbor_count.clamp(min=0.25)
            )

        return field

    def forward(self, filtration: torch.Tensor, features: torch.Tensor) -> dict:
        """Compute differentiable persistence.

        Args:
            filtration: (B, 1, H, W) filtration function values
            features:   (B, C, H, W) image features

        Returns:
            dict with 'persistence_map', 'birth_map', 'stability', 'final_components'
        """
        B, _, H, W = filtration.shape
        device = filtration.device

        # Determine threshold levels from learned offsets
        f_min = filtration.reshape(B, -1).min(dim=1)[0].reshape(B, 1, 1, 1)
        f_max = filtration.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1, 1)
        f_range = (f_max - f_min).clamp(min=1e-6)

        levels = torch.sigmoid(self.level_offsets).to(device)
        levels = levels.sort()[0]  # Ensure monotonic

        # Track when each pixel "appears" (birth) and "merges/dies" (death)
        # Birth ≈ first level where pixel is in sublevel set
        # Persistence ≈ how long pixel remains a local maximum component

        birth_map = torch.zeros(B, 1, H, W, device=device)
        prev_mask = torch.zeros(B, 1, H, W, device=device)

        # Online variance via Welford's algorithm (O(1) memory vs O(L) list)
        mean_feat = torch.zeros_like(features)
        m2_feat = torch.zeros_like(features)
        last_comp_feat = None

        for i, level_frac in enumerate(levels):
            threshold = f_min + level_frac * f_range
            mask = self._soft_threshold(filtration, threshold, temperature=10.0)

            # New pixels appearing at this level
            new_pixels = (mask - prev_mask).clamp(min=0)

            # Record birth time for pixels appearing
            birth_map = birth_map + new_pixels * level_frac

            # Diffuse features within current sublevel set
            comp_feat = self._diffusion_components(
                mask, features, self.num_diffusion_steps
            )

            # Welford online update for temporal variance
            count = i + 1
            delta = comp_feat - mean_feat
            mean_feat = mean_feat + delta / count
            m2_feat = m2_feat + delta * (comp_feat - mean_feat)
            last_comp_feat = comp_feat

            prev_mask = mask

        # Final temporal variance (replaces torch.stack + .var)
        num_levels = len(levels)
        temporal_var = m2_feat / max(num_levels - 1, 1)

        # High variance = feature changed across levels = boundary/death
        # Low final variance = stable component = persistent object
        stability = 1.0 / (1.0 + temporal_var.mean(dim=1, keepdim=True))  # (B, 1, H, W)

        # Persistence = early birth + high stability
        persistence_map = (1.0 - birth_map) * stability

        return {
            "persistence_map": persistence_map,  # (B, 1, H, W) — high = likely object
            "birth_map": birth_map,  # (B, 1, H, W) — when pixel appeared
            "stability": stability,  # (B, 1, H, W) — component stability
            "final_components": last_comp_feat,  # (B, C, H, W)
        }


class TopoNet(InferenceMixin, nn.Module):
    """Topological Persistence Detection — objects from topology.

    Objects are detected as persistent topological features:
    regions that appear early in the filtration and maintain
    distinct component identity across many threshold levels.

    Args:
        num_classes: number of object categories
        feat_channels: feature dimension
        num_proposals: max object proposals
        num_filtration_steps: number of threshold levels
        persistence_thresh: minimum persistence for candidate
        img_size: input image size
    """

    def __init__(
        self,
        num_classes: int = 15,
        feat_channels: int = 128,
        num_proposals: int = 100,
        num_filtration_steps: int = 16,
        persistence_thresh: float = 0.1,
        img_size: int = 640,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        self.encoder = FiltrationEncoder(feat_channels)
        self.persistence = DifferentiablePersistence(
            num_levels=num_filtration_steps,
            persistence_thresh=persistence_thresh,
        )

        # Proposal: top-K from persistence map
        self.score_refine = nn.Sequential(
            nn.Conv2d(feat_channels + 3, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Detection head
        self.det_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )
        self.center_offset = nn.Sequential(
            nn.Linear(feat_channels, 32), nn.ReLU(), nn.Linear(32, 2), nn.Tanh()
        )
        self.size_head = nn.Sequential(
            nn.Linear(feat_channels, 32), nn.ReLU(), nn.Linear(32, 2)
        )
        self.angle_head = nn.Sequential(
            nn.Linear(feat_channels, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(),
            nn.Linear(feat_channels // 2, num_classes),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(feat_channels + 1, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, images: torch.Tensor) -> dict:
        B = images.shape[0]
        filtration, features = self.encoder(images)
        topo = self.persistence(filtration, features)

        # Combine persistence info + features for scoring
        score_input = torch.cat(
            [features, topo["persistence_map"], topo["birth_map"], topo["stability"]],
            dim=1,
        )
        scores_map = self.score_refine(score_input)  # (B, 1, H, W)

        _, _, H, W = scores_map.shape
        N = H * W
        K = min(self.num_proposals, N)

        scores_flat = scores_map.reshape(B, N)
        top_scores, top_idx = torch.topk(scores_flat, K, dim=1)

        # Coordinates
        ys = (top_idx // W).float() / H
        xs = (top_idx % W).float() / W
        coords = torch.stack([xs, ys], dim=-1)

        # Extract features at proposals
        feat_flat = features.reshape(B, -1, N)
        idx_exp = top_idx.unsqueeze(1).expand(B, features.shape[1], K)
        local_feat = torch.gather(feat_flat, 2, idx_exp).permute(0, 2, 1)

        feat = self.det_head(local_feat)

        offsets = self.center_offset(feat)
        offset_scale = images.new_tensor([1.0 / W, 1.0 / H])
        centers = (coords + offsets * offset_scale) * self.img_size

        wh = self.size_head(feat).clamp(-5, 5).exp() * (self.img_size / 10.0)
        angles = self.angle_head(feat)
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))
        cls_logits = self.cls_head(feat)
        conf_input = torch.cat([feat, top_scores.unsqueeze(-1)], dim=-1)
        conf = torch.sigmoid(self.conf_head(conf_input))

        return {
            "centers": centers,
            "wh": wh,
            "angles": angles,
            "cls_logits": cls_logits,
            "conf": conf,
            "mass": top_scores,
            "persistence_map": topo["persistence_map"],
            "filtration": filtration,
        }
