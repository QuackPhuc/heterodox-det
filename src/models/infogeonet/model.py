"""InfoGeoNet: Information Geometry Detection.

Core idea: pixels belonging to objects carry high Fisher Information —
small perturbations to object features cause large changes in the
predicted class distribution. Background pixels carry low Fisher
Information because their features are uninformative.

Detection = finding spatial peaks in the Fisher Information map.

Mathematical foundation:
  Fisher Information (diagonal approximation):
    F_ii(x) = Σ_c p_c(x) · (1 - p_c(x))   (Bernoulli variance)
  Fisher magnitude:
    mag(x) = Σ_i F_ii(x)  (trace of diagonal Fisher)
  Objects: peaks of mag(x)
"""

import torch
import torch.nn as nn

from models._common import InferenceMixin


class FisherEncoder(nn.Module):
    """Encodes image into feature map and per-pixel class distribution.

    The class distribution head produces logits whose sigmoid outputs
    are used to compute the diagonal Fisher Information.
    """

    def __init__(self, feat_channels: int = 128, num_classes: int = 15):
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

        # Per-pixel class distribution for Fisher computation
        self.distribution_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels // 2, num_classes, 1),
        )

        # Feature refinement head for detection
        self.feat_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
        )

        self.out_channels = feat_channels

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        logits = self.distribution_head(feat)  # (B, num_classes, H', W')
        features = self.feat_head(feat)  # (B, C, H', W')
        return logits, features


class DiagonalFisherModule(nn.Module):
    """Computes diagonal Fisher Information map from class distribution.

    For a Bernoulli distribution with parameter p, the Fisher Information
    is F = p(1-p). For multi-class sigmoid outputs, the diagonal Fisher
    is the sum of per-class Bernoulli variances.

    The Fisher magnitude (trace) is high where the network is most
    sensitive to input changes — typically at object locations where
    features carry discriminative information.

    Args:
        fisher_eps: minimum clamp value for Fisher diagonal entries
        num_fisher_samples: number of MC perturbation samples for
            empirical Fisher refinement (0 = use analytical formula only)
        mc_blend: weight of the MC correction term in the analytical+MC
            blend (0.0 = pure analytical, 1.0 = pure MC)
    """

    def __init__(
        self,
        fisher_eps: float = 1e-6,
        num_fisher_samples: int = 0,
        mc_blend: float = 0.3,
    ):
        super().__init__()
        self.fisher_eps = fisher_eps
        self.num_fisher_samples = num_fisher_samples
        self.mc_blend = mc_blend

        # Learnable temperature for sharpening/softening the distribution
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits: torch.Tensor) -> dict:
        """Compute Fisher Information map.

        Args:
            logits: (B, num_classes, H, W) raw class logits

        Returns:
            dict with 'fisher_map' (B, 1, H, W) and 'class_probs' (B, C, H, W)
        """
        # Apply learned temperature scaling
        temp = self.temperature.clamp(min=0.1, max=10.0)
        scaled_logits = logits / temp

        # Per-class probabilities via sigmoid (independent binary classifiers)
        probs = torch.sigmoid(scaled_logits)  # (B, C, H, W)

        # Diagonal Fisher Information: F_ii = p_i * (1 - p_i)
        # Lower-clamped to eps for numerical stability; upper bound 0.25
        # is guaranteed by the sigmoid → p*(1-p) identity
        fisher_diag = (probs * (1.0 - probs)).clamp(min=self.fisher_eps)  # (B, C, H, W)

        # Fisher magnitude: trace of diagonal Fisher matrix
        # sum over class dimension → scalar per pixel
        fisher_magnitude = fisher_diag.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Empirical Fisher refinement via MC gradient samples (training only)
        if self.num_fisher_samples > 0 and self.training:
            fisher_magnitude = self._mc_refine(fisher_magnitude, probs)

        return {
            "fisher_map": fisher_magnitude,
            "class_probs": probs,
        }

    def _mc_refine(
        self,
        analytical: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Refine Fisher estimate with Monte Carlo score-function samples.

        Samples from the predicted Bernoulli distribution and computes
        the empirical score-function variance as a correction term.
        Gradients flow through `probs` via the score (y - p) term,
        while sampling itself is non-differentiable (REINFORCE-style).
        """
        B, C, H, W = probs.shape
        mc_var = torch.zeros(B, 1, H, W, device=probs.device, dtype=probs.dtype)

        for _ in range(self.num_fisher_samples):
            # Sample binary labels (no gradient through sampling op)
            samples = torch.bernoulli(probs.detach())
            # Score function: gradient flows through probs, not sampling
            score = samples - probs
            score_sq = (score**2).sum(dim=1, keepdim=True)
            mc_var = mc_var + score_sq

        mc_fisher = mc_var / max(self.num_fisher_samples, 1)

        # Blend analytical (stable gradient) with MC correction
        alpha = self.mc_blend
        return (1.0 - alpha) * analytical + alpha * mc_fisher


class InfoGeoNet(InferenceMixin, nn.Module):
    """Information Geometry Detection — objects from Fisher Information.

    Objects are detected as spatial peaks in the Fisher Information map.
    High Fisher Information indicates pixels whose features are highly
    discriminative — changing them significantly affects the class
    distribution, which is characteristic of object pixels.

    Args:
        num_classes: number of object categories
        feat_channels: feature dimension
        num_proposals: max object proposals
        num_fisher_samples: MC samples for empirical Fisher (0 = analytical only)
        fisher_eps: numerical stability constant
        mc_blend: weight of MC correction in analytical+MC blend
        img_size: input image size
    """

    def __init__(
        self,
        num_classes: int = 15,
        feat_channels: int = 128,
        num_proposals: int = 100,
        num_fisher_samples: int = 0,
        fisher_eps: float = 1e-6,
        mc_blend: float = 0.3,
        img_size: int = 640,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        self.encoder = FisherEncoder(feat_channels, num_classes)
        self.fisher = DiagonalFisherModule(
            fisher_eps=fisher_eps,
            num_fisher_samples=num_fisher_samples,
            mc_blend=mc_blend,
        )

        # Score refinement: features + fisher map → objectness
        self.score_refine = nn.Sequential(
            nn.Conv2d(feat_channels + 1, feat_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Detection head (same pattern as TopoNet/FlowNet)
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

        # Encode image → class logits + features
        logits, features = self.encoder(images)

        # Compute Fisher Information map
        fisher_out = self.fisher(logits)
        fisher_map = fisher_out["fisher_map"]  # (B, 1, H', W')

        # Score proposals using features + Fisher map
        score_input = torch.cat([features, fisher_map], dim=1)
        scores_map = self.score_refine(score_input)  # (B, 1, H', W')

        _, C, H, W = features.shape
        N = H * W
        K = min(self.num_proposals, N)

        scores_flat = scores_map.reshape(B, N)
        top_scores, top_idx = torch.topk(scores_flat, K, dim=1)

        # Coordinates (normalized grid → absolute)
        ys = (top_idx // W).float() / H
        xs = (top_idx % W).float() / W
        coords = torch.stack([xs, ys], dim=-1)

        # Extract features at proposal locations
        feat_flat = features.reshape(B, C, N)
        idx_exp = top_idx.unsqueeze(1).expand(B, C, K)
        local_feat = torch.gather(feat_flat, 2, idx_exp).permute(0, 2, 1)

        feat = self.det_head(local_feat)

        # Predict OBB parameters
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
            "fisher_map": fisher_map,
        }
