"""Continuous Scale-Space Detection Backbone.

Core idea: instead of a discrete feature pyramid (P3/P4/P5), objects
exist at a CONTINUOUS scale σ ∈ ℝ⁺. A SIREN-based network takes
(x, y, σ) as input and outputs features at any arbitrary scale.

Detection = finding extrema in 3D scale-space (x, y, σ).

Grounded in Lindeberg's scale-space theory (1998):
  Objects are detected as normalized Laplacian extrema in scale-space,
  providing scale-covariant blob detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIRENLayer(nn.Module):
    """Sinusoidal Representation Network (SIREN) layer.

    Uses sin() activation which naturally represents spatial frequencies
    and enables exact derivative computation — ideal for scale-space analysis.

    sin(ω₀ · (Wx + b))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        # Special initialization for SIREN stability
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ScaleSpaceEncoder(nn.Module):
    """Lightweight CNN encoder for extracting base image features.

    These features condition the SIREN scale-space function.
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
        self.out_channels = feat_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to conditioning features.

        Args:
            x: (B, 3, H, W)
        Returns:
            feat: (B, C, H/4, W/4)
        """
        return self.encoder(x)


class ContinuousScaleFunction(nn.Module):
    """SIREN-based continuous scale-space function.

    Maps (image_features, σ) → features at scale σ.

    Instead of discrete pyramid levels, provides features at ANY scale
    via a continuous function parameterized by SIREN.

    Args:
        feat_channels: input/output feature dimension
        hidden_dim: SIREN hidden dimension
        num_scales_sample: number of discrete scale samples during training
        sigma_range: (min_sigma, max_sigma) scale range
    """

    def __init__(
        self,
        feat_channels: int = 128,
        hidden_dim: int = 128,
        num_scales_sample: int = 8,
        sigma_range: tuple = (0.5, 8.0),
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_scales_sample = num_scales_sample
        self.sigma_min, self.sigma_max = sigma_range

        # Scale encoding via positional encoding
        self.scale_embed_dim = 32
        self.num_freqs = 8  # Multi-frequency encoding of σ

        # SIREN: takes (image_features + scale_encoding) → scaled features
        siren_in = feat_channels + self.scale_embed_dim
        self.siren = nn.Sequential(
            SIRENLayer(siren_in, hidden_dim, omega_0=30.0, is_first=True),
            SIRENLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SIRENLayer(hidden_dim, hidden_dim, omega_0=30.0),
            nn.Linear(hidden_dim, feat_channels),  # Final linear (no sin)
        )

        # Scale embedding MLP
        self.scale_encoder = nn.Sequential(
            nn.Linear(self.num_freqs * 2 + 1, self.scale_embed_dim),
            nn.ReLU(inplace=True),
        )

    def _encode_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        """Multi-frequency positional encoding of scale parameter.

        γ(σ) = [σ, sin(2⁰πσ), cos(2⁰πσ), ..., sin(2^Lπσ), cos(2^Lπσ)]
        """
        freqs = 2.0 ** torch.arange(
            self.num_freqs, device=sigma.device, dtype=sigma.dtype
        )
        scaled = sigma.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi
        encoding = torch.cat(
            [sigma.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)], dim=-1
        )
        return self.scale_encoder(encoding)

    def forward(
        self,
        base_feat: torch.Tensor,
        sigma_values: torch.Tensor = None,
    ) -> tuple:
        """Query scale-space at given scales.

        Args:
            base_feat: (B, C, H, W) base image features
            sigma_values: (S,) scale values to sample.
                          If None, uses linearly spaced samples.

        Returns:
            scale_feats: (B, S, C, H, W) features at each scale
            sigmas:      (S,) the scale values used
        """
        B, C, H, W = base_feat.shape
        device = base_feat.device

        if sigma_values is None:
            sigma_values = torch.linspace(
                self.sigma_min,
                self.sigma_max,
                self.num_scales_sample,
                device=device,
            )

        S = len(sigma_values)

        # Encode scale parameters → (S, scale_embed_dim)
        scale_embeds = self._encode_scale(sigma_values)  # (S, D_scale)

        # Reshape base features for processing
        # base_feat: (B, C, H, W) → (B, H*W, C)
        feat_flat = base_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Process each scale level
        scale_feats = []
        for s in range(S):
            # Broadcast scale embedding to all spatial positions
            se = scale_embeds[s : s + 1].expand(B, H * W, -1)  # (B, HW, D_scale)
            # Concatenate features + scale encoding
            siren_input = torch.cat([feat_flat, se], dim=-1)  # (B, HW, C + D_scale)
            # SIREN forward
            out = self.siren(siren_input)  # (B, HW, C)
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            scale_feats.append(out)

        scale_feats = torch.stack(scale_feats, dim=1)  # (B, S, C, H, W)
        return scale_feats, sigma_values


class ScaleSpaceExtrema(nn.Module):
    """Find extrema in continuous scale-space for object detection.

    Computes normalized Laplacian response at each (x, y, σ) and finds
    3D extrema — positions that are local maxima in both space and scale.

    Based on Lindeberg scale-space theory: objects are blobs detected
    at their characteristic scale σ*.
    """

    def __init__(self, feat_channels: int = 128, num_proposals: int = 100):
        super().__init__()
        self.num_proposals = num_proposals

        # Learned normalized Laplacian (approximation via 1x1 conv)
        self.response_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
        )

    def forward(
        self,
        scale_feats: torch.Tensor,
        sigma_values: torch.Tensor,
    ) -> tuple:
        """Find scale-space extrema.

        Args:
            scale_feats: (B, S, C, H, W) features at S scales
            sigma_values: (S,) scale values

        Returns:
            scores:   (B, K) — extrema response scores
            coords:   (B, K, 2) — normalized (x, y) positions
            scales:   (B, K, 1) — detected scale σ* for each extremum
            indices:  (B, K) — flat spatial indices (at best scale)
        """
        B, S, C, H, W = scale_feats.shape
        device = scale_feats.device

        # Compute response at each scale
        responses = []
        for s in range(S):
            # σ²-normalized Laplacian response
            sigma_sq = sigma_values[s] ** 2
            resp = self.response_head(scale_feats[:, s])  # (B, 1, H, W)
            resp = resp * sigma_sq  # Normalize by σ²
            responses.append(resp)

        response_volume = torch.cat(responses, dim=1)  # (B, S, H, W)

        # Find 3D extrema: suppress non-maxima in 3×3×3 neighborhood
        # Pad scale and spatial dims for max pooling
        padded = F.pad(response_volume, (1, 1, 1, 1, 1, 1), mode="constant", value=-1e9)
        # (B, S+2, H+2, W+2) → max pool 3D
        local_max = F.max_pool3d(
            padded.unsqueeze(1), kernel_size=3, stride=1, padding=0
        )
        local_max = local_max.squeeze(1)  # (B, S, H, W)

        # Extrema: points matching their 3D neighborhood max (tolerance for float precision)
        is_peak = ((response_volume - local_max).abs() < 1e-6) & (response_volume > 0)

        # Flatten and select top-K
        flat_response = response_volume.reshape(B, S * H * W)
        flat_mask = is_peak.reshape(B, S * H * W).float()
        masked = flat_response * flat_mask + (-1e9) * (1 - flat_mask)

        K = min(self.num_proposals, S * H * W)
        top_scores, top_flat_idx = torch.topk(masked, K, dim=1)

        # Decode (scale_idx, y, x) from flat index
        scale_idx = top_flat_idx // (H * W)
        spatial_idx = top_flat_idx % (H * W)
        y_idx = spatial_idx // W
        x_idx = spatial_idx % W

        coords = torch.stack(
            [
                x_idx.float() / W,
                y_idx.float() / H,
            ],
            dim=-1,
        )  # (B, K, 2) normalized

        # Scale values for each detection
        scales = sigma_values[scale_idx.clamp(0, S - 1)].unsqueeze(-1)  # (B, K, 1)

        # Clamp scores to [0, ∞) for sigmoid later
        top_scores = top_scores.clamp(min=0)

        return top_scores, coords, scales, spatial_idx


class ScaleNetBackbone(nn.Module):
    """Complete Scale-Space backbone.

    Pipeline: Image → CNN Encoder → SIREN Scale-Space → Extrema Detection

    Returns multi-scale features sampled continuously, plus scale-space
    extrema as object proposals with their characteristic scales.
    """

    def __init__(
        self,
        feat_channels: int = 128,
        num_scales: int = 8,
        sigma_range: tuple = (0.5, 8.0),
        num_proposals: int = 100,
    ):
        super().__init__()
        self.encoder = ScaleSpaceEncoder(feat_channels)
        self.scale_func = ContinuousScaleFunction(
            feat_channels=feat_channels,
            num_scales_sample=num_scales,
            sigma_range=sigma_range,
        )
        self.extrema = ScaleSpaceExtrema(
            feat_channels=feat_channels,
            num_proposals=num_proposals,
        )
        self.out_channels = feat_channels

    def forward(self, x: torch.Tensor) -> dict:
        base_feat = self.encoder(x)
        scale_feats, sigmas = self.scale_func(base_feat)
        scores, coords, scales, spatial_idx = self.extrema(scale_feats, sigmas)

        return {
            "base_feat": base_feat,
            "scale_feats": scale_feats,
            "sigmas": sigmas,
            "peak_scores": scores,
            "peak_coords": coords,
            "peak_scales": scales,
            "peak_spatial_idx": spatial_idx,
        }
