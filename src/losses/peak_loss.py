"""Shared loss for peak-based detectors (WaveDetNet, ScaleNet).

Both WaveDetNet and ScaleNet use a peak-proposal architecture,
so they share the same loss structure:
  - Focal classification loss on proposals
  - Smooth L1 regression on (cx, cy, w, h)
  - Angular sin/cos loss
  - Differentiable OBB IoU loss
  - Confidence loss (peak score vs. actual objectness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.obb_utils import obb_iou_tensor


class PeakDetLoss(nn.Module):
    """Loss function for peak-based detection (WaveDetNet, ScaleNet).

    Args:
        num_classes: number of object categories
        cls_weight: classification loss weight
        reg_weight: regression loss weight
        angle_weight: angle prediction loss weight
        iou_weight: OBB IoU loss weight
        conf_weight: confidence loss weight
        focal_alpha: focal loss alpha
        focal_gamma: focal loss gamma
    """

    def __init__(
        self,
        num_classes: int = 15,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
        angle_weight: float = 1.0,
        iou_weight: float = 2.0,
        conf_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.angle_weight = angle_weight
        self.iou_weight = iou_weight
        self.conf_weight = conf_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _assign_targets(self, preds: dict, targets: list, img_size: int):
        """Assign GT to proposals via closest center distance."""
        B = len(targets)
        K = preds["centers"].shape[1]
        device = preds["centers"].device

        slot_labels = torch.full((B, K), -1, dtype=torch.long, device=device)
        slot_targets = torch.zeros(B, K, 5, device=device)
        fg_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        for b in range(B):
            gt_obbs = targets[b]["obbs"].to(device)
            gt_classes = targets[b]["classes"].to(device)
            G = len(gt_obbs)
            if G == 0:
                continue

            pred_centers = preds["centers"][b].detach()  # (K, 2)
            gt_centers = gt_obbs[:, :2]  # (G, 2)

            # L2 distance between predictions and GT centers
            dist = torch.cdist(
                pred_centers / img_size, gt_centers / img_size, p=2
            )  # (K, G)

            # Optimal assignment via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())
            for r, c in zip(row_ind, col_ind):
                slot_labels[b, r] = gt_classes[c]
                slot_targets[b, r] = gt_obbs[c]
                fg_mask[b, r] = True

        return slot_labels, slot_targets, fg_mask

    def _focal_loss(self, logits, targets):
        N, C = logits.shape
        device = logits.device
        one_hot = torch.zeros(N, C, device=device)
        fg = targets >= 0
        if fg.any():
            one_hot[fg] = F.one_hot(targets[fg], C).float()
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
        p_t = probs * one_hot + (1 - probs) * (1 - one_hot)
        alpha_t = self.focal_alpha * one_hot + (1 - self.focal_alpha) * (1 - one_hot)
        focal = alpha_t * (1 - p_t) ** self.focal_gamma * bce
        return focal.sum() / max(fg.sum().item(), 1)

    def forward(self, preds: dict, targets: list, img_size: int = 640) -> dict:
        B, K = preds["conf"].shape[0], preds["conf"].shape[1]
        device = preds["conf"].device

        slot_labels, slot_targets, fg_mask = self._assign_targets(
            preds, targets, img_size
        )
        num_fg = fg_mask.sum().float().clamp(min=1.0)

        # Classification
        cls_logits = preds["cls_logits"].reshape(-1, self.num_classes)
        loss_cls = self._focal_loss(cls_logits, slot_labels.reshape(-1))

        # Regression, angle, IoU
        if fg_mask.any():
            pred_c = preds["centers"][fg_mask]
            pred_wh = preds["wh"][fg_mask]
            pred_a = preds["angles"][fg_mask]
            gt_c = slot_targets[fg_mask][:, :2]
            gt_wh = slot_targets[fg_mask][:, 2:4]
            gt_a = slot_targets[fg_mask][:, 4:5]

            loss_reg = (
                F.smooth_l1_loss(
                    torch.cat([pred_c, pred_wh], -1) / img_size,
                    torch.cat([gt_c, gt_wh], -1) / img_size,
                    reduction="sum",
                )
                / num_fg
            )

            loss_angle = (
                F.smooth_l1_loss(
                    torch.cat([torch.sin(2 * pred_a), torch.cos(2 * pred_a)], -1),
                    torch.cat([torch.sin(2 * gt_a), torch.cos(2 * gt_a)], -1),
                    reduction="sum",
                )
                / num_fg
            )

            pred_obb = torch.cat([pred_c, pred_wh, pred_a], -1)
            gt_obb = slot_targets[fg_mask]
            iou = obb_iou_tensor(pred_obb, gt_obb)
            loss_iou = (1 - iou).sum() / num_fg
        else:
            loss_reg = torch.zeros(1, device=device).squeeze()
            loss_angle = torch.zeros(1, device=device).squeeze()
            loss_iou = torch.zeros(1, device=device).squeeze()

        # Confidence loss
        conf = preds["conf"].squeeze(-1)
        loss_conf = F.binary_cross_entropy(conf, fg_mask.float(), reduction="sum") / (
            B * K
        )

        total = (
            self.cls_weight * loss_cls
            + self.reg_weight * loss_reg
            + self.angle_weight * loss_angle
            + self.iou_weight * loss_iou
            + self.conf_weight * loss_conf
        )

        return {
            "total": total,
            "cls": loss_cls.detach(),
            "reg": loss_reg.detach(),
            "angle": loss_angle.detach(),
            "iou": loss_iou.detach(),
            "objectness": loss_conf.detach(),
        }
