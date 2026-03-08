"""WaveDetNet: Wave Resonance Detection model.

Full model: Image → Wave Backbone → Resonance Analysis → OBB Detections

Objects are detected as regions of wave resonance — spatial locations
where propagated waves reflect, accumulate energy, and create standing
wave patterns. Multi-scale detection emerges naturally from the frequency
content of wave oscillations.
"""

import torch
import torch.nn as nn

from .backbone import WaveBackbone
from .head import ResonancePeakProposer, WaveDetHead
from models._common import InferenceMixin


class WaveDetNet(InferenceMixin, nn.Module):
    """Wave Resonance Detection — physics-inspired object detector.

    Instead of feedforward layers or attention, information propagates
    via learned wave dynamics on the feature lattice. Objects emerge
    as resonance regions with high accumulated wave energy.

    Args:
        num_classes: number of object categories
        feat_channels: wave feature dimension
        num_proposals: max object proposals per image
        num_wave_steps: number of wave propagation timesteps
        wave_dt: wave equation time step
        img_size: input image size
    """

    def __init__(
        self,
        num_classes: int = 15,
        feat_channels: int = 128,
        num_proposals: int = 100,
        num_wave_steps: int = 8,
        wave_dt: float = 0.3,
        img_size: int = 640,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        self.backbone = WaveBackbone(
            feat_channels=feat_channels,
            num_steps=num_wave_steps,
            dt=wave_dt,
        )
        self.proposer = ResonancePeakProposer(
            feat_channels=feat_channels,
            num_proposals=num_proposals,
        )
        self.det_head = WaveDetHead(
            feat_channels=feat_channels,
            num_classes=num_classes,
            num_proposals=num_proposals,
        )

    def forward(self, images: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            images: (B, 3, H, W)

        Returns:
            dict with 'centers', 'wh', 'angles', 'cls_logits', 'conf',
                       'energy', 'wave_speed'
        """
        wave_out = self.backbone(images)

        peak_scores, peak_indices, peak_coords = self.proposer(wave_out["fused"])

        preds = self.det_head(
            wave_out["fused"],
            peak_scores,
            peak_indices,
            peak_coords,
            self.img_size,
        )
        preds["energy"] = wave_out["energy"]
        preds["wave_speed"] = wave_out["wave_speed"]
        preds["mass"] = peak_scores  # compatibility with shared loss/eval

        return preds
