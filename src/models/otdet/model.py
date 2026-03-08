"""OTDet: Optimal Transport Detection model.

Full model assembly combining backbone, FPN neck, Sinkhorn OT module,
and OBB detection head into a single end-to-end trainable detector.

Architecture:
  Image → Backbone → FPN → (Objectness | Features) → Sinkhorn OT → OBB Head → Detections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet18Backbone
from .neck import SimpleFPN
from .ot_module import SinkhornOT, ObjectSlots
from .head import OBBHead
from models._common import InferenceMixin


class OTDet(InferenceMixin, nn.Module):
    """Optimal Transport Detection — novel OBB object detector.

    Detection is framed as an optimal transport problem: pixel-level
    objectness evidence is transported to learnable object slots via
    the Sinkhorn algorithm. Objects emerge as slots that receive
    significant transport mass.

    Args:
        num_classes: number of object categories
        num_slots: number of learnable object candidate slots
        feat_channels: FPN output channel dimension
        pretrained_backbone: use ImageNet-pretrained ResNet-18
        sinkhorn_iters: number of Sinkhorn iterations
        sinkhorn_eps: entropic regularization parameter
        ot_cost_type: cost function ('l2' or 'cosine')
        img_size: input image size (assumes square)
    """

    def __init__(
        self,
        num_classes: int = 15,
        num_slots: int = 100,
        feat_channels: int = 256,
        pretrained_backbone: bool = True,
        sinkhorn_iters: int = 20,
        sinkhorn_eps: float = 0.05,
        ot_cost_type: str = "l2",
        img_size: int = 640,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.feat_channels = feat_channels

        # Backbone: extracts multi-scale features
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)

        # Neck: fuses multi-scale → single stride-8 feature map
        self.neck = SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=feat_channels,
        )

        # Objectness head: per-pixel objectness score (source distribution μ)
        self.objectness_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Feature projection: project FPN features to OT feature space
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 1),
        )

        # Object slots: learnable query embeddings
        self.slots = ObjectSlots(num_slots=num_slots, feat_dim=feat_channels)

        # Sinkhorn OT solver
        self.sinkhorn = SinkhornOT(
            num_iters=sinkhorn_iters,
            eps=sinkhorn_eps,
            cost_type=ot_cost_type,
        )

        # Detection head: transport plan → OBB predictions
        self.det_head = OBBHead(
            feat_dim=feat_channels,
            num_classes=num_classes,
            num_slots=num_slots,
        )

    def _get_coord_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate normalized spatial coordinate grid.

        Returns:
            coords: (1, H*W, 2) — (x, y) in [0, 1]
        """
        ys = torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device)
        xs = torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H * W, 2)
        return coords

    def forward(self, images: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            images: (B, 3, H, W) — input image batch

        Returns:
            dict with all prediction components:
                'centers', 'wh', 'angles', 'cls_logits', 'conf', 'mass',
                'objectness_map', 'transport_plan'
        """
        B = images.shape[0]

        # 1. Backbone → multi-scale features
        feat_dict = self.backbone(images)

        # 2. Neck → fused single-scale feature map (B, D, H', W')
        fused = self.neck(feat_dict)
        _, D, Hf, Wf = fused.shape
        N = Hf * Wf

        # 3. Objectness map (source distribution)
        objectness = self.objectness_head(fused)  # (B, 1, Hf, Wf)
        obj_flat = objectness.reshape(B, N)  # (B, N)
        # Normalize to probability distribution
        source_weights = obj_flat / obj_flat.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # 4. Pixel features
        pixel_feats_map = self.feat_proj(fused)  # (B, D, Hf, Wf)
        pixel_feats = pixel_feats_map.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

        # 5. Object slot features
        slot_feats = self.slots(B)  # (B, K, D)

        # 6. Sinkhorn OT
        transport_plan = self.sinkhorn(
            pixel_feats, slot_feats, source_weights
        )  # (B, N, K)

        # 7. Spatial coordinate grid
        coords = self._get_coord_grid(Hf, Wf, images.device).expand(
            B, -1, -1
        )  # (B, N, 2)

        # 8. Detection head: transport plan → OBB predictions
        preds = self.det_head(transport_plan, pixel_feats, coords, self.img_size)
        preds["objectness_map"] = objectness
        preds["transport_plan"] = transport_plan

        return preds
