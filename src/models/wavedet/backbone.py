"""Wave Propagation Backbone — learned wave dynamics on feature graphs.

Core idea: instead of feedforward layers, propagate waves on the spatial
feature graph. Information spreads via physics-inspired wave equations
with learned parameters (wave speed, damping, source terms).

Wave equation on image lattice (discrete):
  u(x, t+1) = 2u(x,t) - u(x,t-1) + Δt²·c(x)²·Δu(x,t) - γ(x)·(u(x,t) - u(x,t-1))

Where:
  c(x)   = spatially-varying wave speed (learned from image)
  γ(x)   = damping coefficient (learned, determines resonance sharpness)
  Δu     = graph Laplacian (learned edge weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveInitEncoder(nn.Module):
    """Lightweight CNN that encodes image into wave parameters.

    Produces:
      - Initial wave field u(x, 0)
      - Wave speed map c(x)
      - Damping map γ(x)
      - Feature map for final readout
    """

    def __init__(self, feat_channels: int = 128):
        super().__init__()
        self.feat_channels = feat_channels

        # Lightweight encoder (4x downsample)
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

        # Wave parameter heads
        self.wave_field_init = nn.Conv2d(feat_channels, feat_channels, 1)
        self.wave_speed = nn.Sequential(
            nn.Conv2d(feat_channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),  # Positive wave speed
        )
        self.damping = nn.Sequential(
            nn.Conv2d(feat_channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),  # Damping in [0, 1]
        )

    def forward(self, x: torch.Tensor):
        """Encode image to wave parameters.

        Args:
            x: (B, 3, H, W)

        Returns:
            u0:      (B, C, H', W') initial wave field
            speed:   (B, 1, H', W') wave speed
            damp:    (B, 1, H', W') damping
            base_feat: (B, C, H', W') base features for readout
        """
        feat = self.encoder(x)  # (B, C, H/4, W/4)
        u0 = self.wave_field_init(feat)  # (B, C, H/4, W/4)
        speed = self.wave_speed(feat) + 0.1  # (B, 1, H/4, W/4), min speed
        damp = self.damping(feat) * 0.3  # (B, 1, H/4, W/4), moderate damping
        return u0, speed, damp, feat


class LearnedGraphLaplacian(nn.Module):
    """Learned anisotropic graph Laplacian for wave propagation.

    Instead of fixed 3x3 Laplacian kernel, learns spatially-varying
    edge weights that define how waves propagate through features.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Predict per-pixel directional diffusion weights (8 neighbors)
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 4, 1),  # 4 directions (up, down, left, right)
            nn.Softplus(),
        )

    def forward(self, u: torch.Tensor, weights_input: torch.Tensor) -> torch.Tensor:
        """Apply learned Laplacian to wave field.

        Args:
            u: (B, C, H, W) wave field
            weights_input: (B, C, H, W) features to derive edge weights

        Returns:
            laplacian_u: (B, C, H, W)
        """
        # Predict diffusion weights (4 cardinal directions)
        w = self.weight_net(weights_input)  # (B, 4, H, W)
        w_up, w_down, w_left, w_right = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]

        # Shifted versions of u
        u_up = F.pad(u[:, :, :-1, :], (0, 0, 1, 0))  # shift down → neighbor up
        u_down = F.pad(u[:, :, 1:, :], (0, 0, 0, 1))  # shift up → neighbor down
        u_left = F.pad(u[:, :, :, :-1], (1, 0, 0, 0))  # shift right → neighbor left
        u_right = F.pad(u[:, :, :, 1:], (0, 1, 0, 0))  # shift left → neighbor right

        # Weighted Laplacian: Δu = Σ w_dir · (u_neighbor - u)
        laplacian = (
            w_up * (u_up - u)
            + w_down * (u_down - u)
            + w_left * (u_left - u)
            + w_right * (u_right - u)
        )
        return laplacian


class WavePropagation(nn.Module):
    """Wave equation solver on feature lattice.

    Propagates waves for T timesteps using the learned wave equation:
      u(t+1) = 2u(t) - u(t-1) + Δt²·c²·Δu(t) - γ·(u(t) - u(t-1))

    Collects energy accumulation E(x) = Σ_t |u(x,t)|² at each spatial
    location to identify resonance regions (objects).
    """

    def __init__(self, channels: int, num_steps: int = 8, dt: float = 0.3):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.laplacian = LearnedGraphLaplacian(channels)

        # Learnable per-step mixing — allows model to control
        # contribution of each timestep to final energy
        self.step_weights = nn.Parameter(torch.ones(num_steps) / num_steps)

    def forward(
        self,
        u0: torch.Tensor,
        speed: torch.Tensor,
        damping: torch.Tensor,
        base_feat: torch.Tensor,
    ) -> tuple:
        """Propagate waves and collect resonance energy.

        Args:
            u0:        (B, C, H, W) initial wave field
            speed:     (B, 1, H, W) wave speed
            damping:   (B, 1, H, W) damping coefficient
            base_feat: (B, C, H, W) features for Laplacian weights

        Returns:
            energy:       (B, C, H, W) accumulated resonance energy
            final_field:  (B, C, H, W) wave field at final timestep
            spectral_feat: (B, C, H, W) spectral features (from temporal FFT-like)
        """
        B, C, H, W = u0.shape
        dt2 = self.dt**2

        # Initialize: u(t=0) = u0, u(t=-1) = u0 (zero initial velocity)
        u_prev = u0
        u_curr = u0

        energy = torch.zeros_like(u0)

        # Online variance via Welford's algorithm (O(1) memory vs O(T) list)
        mean_u = torch.zeros_like(u0)
        m2_u = torch.zeros_like(u0)

        weights = F.softmax(self.step_weights, dim=0)

        for t in range(self.num_steps):
            # Learned graph Laplacian
            lap_u = self.laplacian(u_curr, base_feat)

            # Wave equation update
            # u_next = 2u - u_prev + dt²·c²·Δu - γ·(u - u_prev)
            velocity = u_curr - u_prev
            u_next = 2 * u_curr - u_prev + dt2 * speed**2 * lap_u - damping * velocity

            # Accumulate energy (weighted)
            energy = energy + weights[t] * u_next**2

            # Welford online update for spectral variance
            count = t + 1
            delta = u_next - mean_u
            mean_u = mean_u + delta / count
            m2_u = m2_u + delta * (u_next - mean_u)

            # Advance timestep
            u_prev = u_curr
            u_curr = u_next

        # Spectral features: temporal variance across steps (approximates frequency content)
        # High variance = oscillation = resonance at some frequency
        spectral_feat = m2_u / max(self.num_steps - 1, 1)

        return energy, u_curr, spectral_feat


class WaveBackbone(nn.Module):
    """Complete wave-based backbone.

    Pipeline: Image → Encoder → Wave Parameters → Propagation → Resonance Features

    Returns multi-representation features:
      - base_feat: static CNN features
      - energy: accumulated wave energy (object likelihood map)
      - spectral: temporal frequency content (object scale proxy)
    """

    def __init__(self, feat_channels: int = 128, num_steps: int = 8, dt: float = 0.3):
        super().__init__()
        self.encoder = WaveInitEncoder(feat_channels)
        self.propagation = WavePropagation(feat_channels, num_steps, dt)

        # Fuse all representations into unified feature output
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 2, 3, padding=1),
            nn.BatchNorm2d(feat_channels * 2),
            nn.GELU(),
            nn.Conv2d(feat_channels * 2, feat_channels, 1),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
        )

        self.out_channels = feat_channels

    def forward(self, x: torch.Tensor) -> dict:
        """Full wave backbone forward.

        Args:
            x: (B, 3, H, W)

        Returns:
            dict with 'fused', 'energy', 'spectral', 'base_feat'
        """
        u0, speed, damping, base_feat = self.encoder(x)
        energy, final_field, spectral = self.propagation(u0, speed, damping, base_feat)

        # Concatenate and fuse
        combined = torch.cat([base_feat, energy, spectral], dim=1)
        fused = self.fuse(combined)

        return {
            "fused": fused,
            "energy": energy,
            "spectral": spectral,
            "base_feat": base_feat,
            "wave_speed": speed,
        }
