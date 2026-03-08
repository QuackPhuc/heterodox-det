"""Differentiable Sinkhorn Optimal Transport module.

Core innovation: object detection as optimal transport from pixel evidence
to object candidates. The transport plan π assigns pixel features to
learnable object slots, enabling objects to emerge organically.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornOT(nn.Module):
    """Differentiable optimal transport via Sinkhorn-Knopp algorithm.

    Solves the entropically-regularized OT problem:
        min_π  <C, π> - ε·H(π)
        s.t.   π·1 = μ,  πᵀ·1 = ν

    where C is the cost matrix, ε is the regularization, and H is entropy.

    The Sinkhorn algorithm alternates row/column normalizations on the
    Gibbs kernel K = exp(-C/ε) to find the optimal coupling π.
    """

    def __init__(
        self,
        num_iters: int = 20,
        eps: float = 0.05,
        cost_type: str = "l2",
    ):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
        self.cost_type = cost_type

    def _compute_cost(
        self,
        pixel_feats: torch.Tensor,
        slot_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost matrix between pixel features and slot features.

        Args:
            pixel_feats: (B, N, D) — per-pixel feature vectors
            slot_feats:  (B, K, D) — per-slot query features

        Returns:
            C: (B, N, K) — cost matrix
        """
        if self.cost_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            pf = F.normalize(pixel_feats, dim=-1)
            sf = F.normalize(slot_feats, dim=-1)
            sim = torch.bmm(pf, sf.transpose(1, 2))
            return 1.0 - sim
        else:
            # Squared L2 distance
            # ||a - b||² = ||a||² + ||b||² - 2·aᵀb
            p_sq = (pixel_feats**2).sum(dim=-1, keepdim=True)  # (B, N, 1)
            s_sq = (slot_feats**2).sum(dim=-1, keepdim=True)  # (B, K, 1)
            cross = torch.bmm(pixel_feats, slot_feats.transpose(1, 2))  # (B, N, K)
            return p_sq + s_sq.transpose(1, 2) - 2 * cross

    def forward(
        self,
        pixel_feats: torch.Tensor,
        slot_feats: torch.Tensor,
        source_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run Sinkhorn OT.

        Args:
            pixel_feats: (B, N, D) — pixel-level features
            slot_feats:  (B, K, D) — object slot features
            source_weights: (B, N) — per-pixel objectness weights (source μ).
                            If None, uses uniform distribution.

        Returns:
            transport_plan: (B, N, K) — optimal coupling π(pixel, slot)
        """
        B, N, D = pixel_feats.shape
        K = slot_feats.shape[1]

        # Cost matrix
        C = self._compute_cost(pixel_feats, slot_feats)  # (B, N, K)

        # Log-domain Sinkhorn for numerical stability
        # Clamp to prevent underflow when eps is small relative to cost scale
        log_K = (-C / self.eps).clamp(min=-100)  # (B, N, K) — log Gibbs kernel

        # Source distribution (objectness)
        if source_weights is not None:
            log_mu = torch.log(source_weights.clamp(min=1e-8))  # (B, N)
        else:
            log_mu = torch.full((B, N), -math.log(N), device=pixel_feats.device)

        # Target distribution (uniform over slots)
        log_nu = torch.full((B, K), -math.log(K), device=pixel_feats.device)

        # Sinkhorn iterations in log domain
        log_u = torch.zeros(B, N, device=pixel_feats.device)
        log_v = torch.zeros(B, K, device=pixel_feats.device)

        for _ in range(self.num_iters):
            # Row normalization: log_u ← log_mu - logsumexp(log_K + log_v, dim=K)
            log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            # Column normalization: log_v ← log_nu - logsumexp(log_K + log_u, dim=N)
            log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

        # Recover transport plan
        log_pi = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        transport_plan = torch.exp(log_pi)

        return transport_plan


class ObjectSlots(nn.Module):
    """Learnable object slot embeddings.

    Each slot represents a potential object candidate. The OT module
    determines which slots correspond to actual objects by examining
    how much pixel evidence is transported to each slot.

    Args:
        num_slots: number of object candidate slots
        feat_dim: feature dimension (must match FPN output)
    """

    def __init__(self, num_slots: int = 100, feat_dim: int = 256):
        super().__init__()
        # Learnable slot embeddings initialized from truncated normal
        self.slot_embeds = nn.Parameter(torch.randn(1, num_slots, feat_dim) * 0.02)
        self.num_slots = num_slots
        self.feat_dim = feat_dim

    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand slot embeddings to batch size.

        Returns:
            slots: (B, K, D)
        """
        return self.slot_embeds.expand(batch_size, -1, -1)
