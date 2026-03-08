"""Shared base classes for all detection architectures."""

import torch
import torch.nn as nn


class InferenceMixin:
    """Mixin providing standardized inference for all detection models.

    Requires the subclass to implement `forward(images) -> dict` with
    keys 'conf', 'cls_logits', 'centers', 'wh', 'angles'.
    """

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
    ) -> list:
        """Run inference and return per-image detection results.

        Args:
            images: (B, 3, H, W)
            conf_thresh: confidence threshold
            nms_thresh: OBB NMS threshold

        Returns:
            list of dicts, one per image, each with:
                'obbs': (M, 5) — cx, cy, w, h, angle
                'scores': (M,)
                'classes': (M,)
        """
        from utils.inference import inference_postprocess

        was_training = self.training
        self.eval()
        preds = self.forward(images)
        if was_training:
            self.train()
        return inference_postprocess(preds, images.shape[0], conf_thresh, nms_thresh)
