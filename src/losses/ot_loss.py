"""OT-aware loss for OTDet.

Combines:
  - OT objectness loss: encourage objectness map to highlight real objects
  - Classification loss: focal loss on per-slot predictions
  - Regression loss: smooth L1 on (cx, cy, w, h) + angular loss
  - OBB IoU loss: differentiable Gaussian proxy for oriented IoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.obb_utils import obb_iou_tensor


class OTDetLoss(nn.Module):
    """Combined loss for Optimal Transport Detection.

    Args:
        num_classes: number of object categories
        cls_weight: classification loss weight
        reg_weight: regression loss weight
        ot_weight: objectness / OT assignment loss weight
        angle_weight: angle prediction loss weight
        iou_weight: OBB IoU loss weight
        focal_alpha: focal loss alpha
        focal_gamma: focal loss gamma
    """

    def __init__(
        self,
        num_classes: int = 15,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
        ot_weight: float = 1.0,
        angle_weight: float = 1.0,
        iou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ot_weight = ot_weight
        self.angle_weight = angle_weight
        self.iou_weight = iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _assign_targets(
        self,
        preds: dict,
        targets: list,
        img_size: int,
    ) -> tuple:
        """Assign ground truth targets to object slots via cost-based matching.

        For each image, computes a cost matrix between predicted slots and
        GT objects, then uses the Hungarian algorithm for optimal assignment.

        Returns:
            slot_labels: (B, K) int — class label per slot (-1 for background)
            slot_targets: (B, K, 5) float — target OBB (cx, cy, w, h, angle)
            fg_mask: (B, K) bool — foreground slots
        """
        B = len(targets)
        K = preds["centers"].shape[1]
        device = preds["centers"].device

        slot_labels = torch.full((B, K), -1, dtype=torch.long, device=device)
        slot_targets = torch.zeros(B, K, 5, device=device)
        fg_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        for b in range(B):
            gt_obbs = targets[b]["obbs"].to(device)  # (G, 5)
            gt_classes = targets[b]["classes"].to(device)  # (G,)
            G = len(gt_obbs)
            if G == 0:
                continue

            # Detach predictions to avoid building unnecessary graph in assignment
            pred_cx = preds["centers"][b].detach()  # (K, 2)
            pred_wh = preds["wh"][b].detach()  # (K, 2)

            # Cost: L1 distance of centers (normalized)
            gt_centers = gt_obbs[:, :2]  # (G, 2)
            center_cost = torch.cdist(
                pred_cx / img_size, gt_centers / img_size, p=1
            )  # (K, G)

            # Cost: L1 distance of wh (normalized)
            gt_wh = gt_obbs[:, 2:4]
            wh_cost = torch.cdist(pred_wh / img_size, gt_wh / img_size, p=1)

            total_cost = center_cost + 0.5 * wh_cost  # (K, G)

            # Optimal assignment via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(total_cost.cpu().numpy())
            for r, c in zip(row_ind, col_ind):
                slot_labels[b, r] = gt_classes[c]
                slot_targets[b, r] = gt_obbs[c]
                fg_mask[b, r] = True

        return slot_labels, slot_targets, fg_mask

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary focal loss for classification.

        Args:
            logits: (N, C) — raw class logits
            targets: (N,) — class indices (0-indexed, or -1 for background)
        """
        N, C = logits.shape
        device = logits.device

        # Create one-hot targets (background slots get all-zeros)
        one_hot = torch.zeros(N, C, device=device)
        fg = targets >= 0
        if fg.any():
            one_hot[fg] = F.one_hot(targets[fg], C).float()

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")

        # Focal modulation with class-conditional α (Lin et al., 2017)
        p_t = probs * one_hot + (1 - probs) * (1 - one_hot)
        alpha_t = self.focal_alpha * one_hot + (1 - self.focal_alpha) * (1 - one_hot)
        focal = alpha_t * (1 - p_t) ** self.focal_gamma * bce
        return focal.sum() / max(fg.sum().item(), 1)

    def forward(self, preds: dict, targets: list, img_size: int = 640) -> dict:
        """Compute total loss.

        Args:
            preds: model output dict
            targets: list of target dicts per image
            img_size: image size

        Returns:
            dict with 'total', 'cls', 'reg', 'angle', 'iou', 'objectness' losses
        """
        B, K = preds["conf"].shape[0], preds["conf"].shape[1]
        device = preds["conf"].device

        # Assign GT to slots
        slot_labels, slot_targets, fg_mask = self._assign_targets(
            preds, targets, img_size
        )

        # --- Classification loss (focal) ---
        cls_logits = preds["cls_logits"].reshape(-1, self.num_classes)
        labels_flat = slot_labels.reshape(-1)
        loss_cls = self._focal_loss(cls_logits, labels_flat)

        # --- Regression loss (only foreground slots) ---
        num_fg = fg_mask.sum().float().clamp(min=1.0)

        if fg_mask.any():
            pred_centers_fg = preds["centers"][fg_mask]  # (Nfg, 2)
            pred_wh_fg = preds["wh"][fg_mask]  # (Nfg, 2)
            pred_angles_fg = preds["angles"][fg_mask]  # (Nfg, 1)

            gt_centers_fg = slot_targets[fg_mask][:, :2]
            gt_wh_fg = slot_targets[fg_mask][:, 2:4]
            gt_angles_fg = slot_targets[fg_mask][:, 4:5]

            loss_reg = (
                F.smooth_l1_loss(
                    torch.cat([pred_centers_fg, pred_wh_fg], dim=-1) / img_size,
                    torch.cat([gt_centers_fg, gt_wh_fg], dim=-1) / img_size,
                    reduction="sum",
                )
                / num_fg
            )

            # Angular loss: smooth L1 on sin/cos representation
            loss_angle = (
                F.smooth_l1_loss(
                    torch.cat(
                        [torch.sin(2 * pred_angles_fg), torch.cos(2 * pred_angles_fg)],
                        dim=-1,
                    ),
                    torch.cat(
                        [torch.sin(2 * gt_angles_fg), torch.cos(2 * gt_angles_fg)],
                        dim=-1,
                    ),
                    reduction="sum",
                )
                / num_fg
            )

            # OBB IoU loss (differentiable Gaussian proxy)
            pred_obb_fg = torch.cat(
                [pred_centers_fg, pred_wh_fg, pred_angles_fg], dim=-1
            )
            gt_obb_fg = slot_targets[fg_mask]
            iou_scores = obb_iou_tensor(pred_obb_fg, gt_obb_fg)
            loss_iou = (1 - iou_scores).sum() / num_fg
        else:
            loss_reg = torch.zeros(1, device=device).squeeze()
            loss_angle = torch.zeros(1, device=device).squeeze()
            loss_iou = torch.zeros(1, device=device).squeeze()

        # --- Objectness / confidence loss ---
        # Foreground slots should have high confidence, background low
        conf = preds["conf"].squeeze(-1)  # (B, K)
        conf_target = fg_mask.float()
        loss_obj = F.binary_cross_entropy(conf, conf_target, reduction="sum") / (B * K)

        # --- Total loss ---
        total = (
            self.cls_weight * loss_cls
            + self.reg_weight * loss_reg
            + self.angle_weight * loss_angle
            + self.iou_weight * loss_iou
            + self.ot_weight * loss_obj
        )

        return {
            "total": total,
            "cls": loss_cls.detach(),
            "reg": loss_reg.detach(),
            "angle": loss_angle.detach(),
            "iou": loss_iou.detach(),
            "objectness": loss_obj.detach(),
        }
