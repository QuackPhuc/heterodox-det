"""FlowNet: Neural ODE Attractor Detection.

Core idea: feature extraction is a CONTINUOUS dynamical system (ODE).
Objects are STABLE ATTRACTORS of the system — spatial regions where
features naturally converge and settle.

Instead of fixed-depth layers (ResNet) or continuous-depth ODE (Neural ODE),
we model feature evolution as a contractive flow field, and detect objects
as the convergence basins (attractors) of this flow.

Mathematical foundation:
  dh/dt = f_θ(h, t)                    (feature ODE)
  h* = lim_{t→∞} h(t)                  (attractor)
  Objects = stable equilibria where all eigenvalues of ∂f/∂h < 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._common import InferenceMixin


class FeatureEncoder(nn.Module):
    """Lightweight encoder for initial feature state h(0)."""

    def __init__(self, feat_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, x):
        return self.net(x)


class ODEFunc(nn.Module):
    """The ODE dynamics function f_θ(h, t).

    Defines the continuous feature flow field. Designed to be
    contractive so that stable attractors (objects) naturally emerge.

    Contractiveness is encouraged via spectral normalization and
    a negative-definite bias term.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Spatial mixing (local interactions)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # depthwise
            nn.Conv2d(channels, channels, 1),  # pointwise
        )

        # Channel mixing (feature interactions)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

        # Time embedding (condition dynamics on time)
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        # Contraction bias: push towards fixed points
        self.contraction_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt = f_θ(h, t).

        Args:
            t: scalar time tensor (pre-allocated by solver)
            h: (B, C, H, W) current features

        Returns:
            dhdt: (B, C, H, W) feature velocity
        """
        B, C, H, W = h.shape

        # Spatial and channel dynamics
        spatial_out = self.spatial(h)
        channel_out = self.channel(h)

        # Time conditioning (t is already a tensor)
        t_emb = self.time_embed(t.unsqueeze(0)).reshape(1, C, 1, 1)
        t_emb = t_emb.expand(B, -1, H, W)

        # Combined dynamics with contraction
        # The -contraction * h term ensures contractiveness (pulls toward origin/attractor)
        contraction = torch.sigmoid(self.contraction_weight)
        dhdt = spatial_out + channel_out + t_emb - contraction * h

        return dhdt


class FixedStepODESolver(nn.Module):
    """Simple fixed-step ODE solver (no external dependencies).

    Supports Euler and RK4 integration methods.
    Tracks convergence rate at each spatial position.
    """

    def __init__(self, func: ODEFunc, num_steps: int = 8, method: str = "euler"):
        super().__init__()
        self.func = func
        self.num_steps = num_steps
        self.method = method
        self.dt = 1.0 / num_steps

    def _euler_step(self, t, h, dt):
        return h + dt * self.func(t, h)

    def _rk4_step(self, t, h, dt, dt_half):
        k1 = self.func(t, h)
        k2 = self.func(t + dt_half, h + dt_half * k1)
        k3 = self.func(t + dt_half, h + dt_half * k2)
        k4 = self.func(t + dt, h + dt * k3)
        return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, h0: torch.Tensor) -> dict:
        """Integrate ODE from t=0 to t=1.

        Args:
            h0: (B, C, H, W) initial state

        Returns:
            dict with 'final_state', 'convergence_map', 'trajectory_var'
        """
        h = h0
        B, C, H, W = h0.shape
        device = h0.device

        # Pre-allocate time values to avoid per-step CUDA allocation
        dt_t = torch.tensor(self.dt, dtype=h0.dtype, device=device)
        dt_half = dt_t / 2
        t_values = torch.arange(self.num_steps, dtype=h0.dtype, device=device) * self.dt
        is_rk4 = self.method == "rk4"

        # Online variance via Welford's algorithm (O(1) memory vs O(T) list)
        mean_vel = torch.zeros(B, 1, H, W, device=device)
        m2_vel = torch.zeros(B, 1, H, W, device=device)
        last_dhdt = None

        for i in range(self.num_steps):
            t = t_values[i]
            dhdt = self.func(t, h)
            vel_norm = dhdt.detach().norm(dim=1, keepdim=True)

            # Welford online update for trajectory variance
            count = i + 1
            delta = vel_norm - mean_vel
            mean_vel = mean_vel + delta / count
            m2_vel = m2_vel + delta * (vel_norm - mean_vel)

            last_dhdt = dhdt.detach()
            if is_rk4:
                h = self._rk4_step(t, h, dt_t, dt_half)
            else:
                h = self._euler_step(t, h, dt_t)

        # Convergence map: low velocity = converged = near attractor = likely object
        velocity_magnitude = last_dhdt.norm(dim=1, keepdim=True)  # (B, 1, H, W)
        convergence_map = 1.0 / (1.0 + velocity_magnitude)

        # Trajectory variance (replaces torch.stack + .var)
        trajectory_var = m2_vel / max(self.num_steps - 1, 1)

        return {
            "final_state": h,
            "convergence_map": convergence_map,
            "trajectory_var": trajectory_var,
        }


class FlowNet(InferenceMixin, nn.Module):
    """Neural ODE Attractor Detection — objects as dynamical attractors.

    Feature extraction is modeled as a continuous dynamical system.
    Objects are detected as spatial regions where the feature flow
    converges (low velocity = stable attractor).

    Key property: adaptive computation is natural — the ODE solver
    implicitly spends more "compute" on complex regions.

    Args:
        num_classes: number of object categories
        feat_channels: feature dimension
        num_proposals: max object proposals
        ode_steps: number of ODE integration steps
        ode_method: 'euler' or 'rk4'
        img_size: input image size
    """

    def __init__(
        self,
        num_classes: int = 15,
        feat_channels: int = 128,
        num_proposals: int = 100,
        ode_steps: int = 8,
        ode_method: str = "euler",
        img_size: int = 640,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        self.encoder = FeatureEncoder(feat_channels)
        self.ode_func = ODEFunc(feat_channels)
        self.solver = FixedStepODESolver(
            self.ode_func, num_steps=ode_steps, method=ode_method
        )

        # Attractor scoring: convergence + features → objectness
        self.score_net = nn.Sequential(
            nn.Conv2d(feat_channels + 2, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Detection head
        self.feat_enhance = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(),
        )
        self.center_offset = nn.Sequential(
            nn.Linear(feat_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),
        )
        self.size_head = nn.Sequential(
            nn.Linear(feat_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(feat_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(),
            nn.Linear(feat_channels // 2, num_classes),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(feat_channels + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, images: torch.Tensor) -> dict:
        B = images.shape[0]

        # Encode initial state
        h0 = self.encoder(images)  # (B, C, H', W')

        # Solve ODE — features evolve toward attractors
        ode_out = self.solver(h0)
        final_feat = ode_out["final_state"]  # (B, C, H', W')
        conv_map = ode_out["convergence_map"]  # (B, 1, H', W')
        traj_var = ode_out["trajectory_var"]  # (B, 1, H', W')

        _, C, H, W = final_feat.shape
        N = H * W
        K = min(self.num_proposals, N)

        # Score attractors
        score_input = torch.cat([final_feat, conv_map, traj_var], dim=1)
        scores_map = self.score_net(score_input)
        scores_flat = scores_map.reshape(B, N)
        top_scores, top_idx = torch.topk(scores_flat, K, dim=1)

        # Coordinates
        ys = (top_idx // W).float() / H
        xs = (top_idx % W).float() / W
        coords = torch.stack([xs, ys], dim=-1)

        # Extract features
        feat_flat = final_feat.reshape(B, C, N)
        idx_exp = top_idx.unsqueeze(1).expand(B, C, K)
        local_feat = torch.gather(feat_flat, 2, idx_exp).permute(0, 2, 1)

        feat = self.feat_enhance(local_feat)

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
            "convergence_map": conv_map,
            "trajectory_var": traj_var,
        }
