"""Shared inference post-processing for all detection architectures.

Handles: confidence filtering → sigmoid → per-class NMS → result assembly.
"""

import numpy as np
import torch

from utils.obb_utils import obb_nms


def inference_postprocess(
    preds: dict,
    num_images: int,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45,
) -> list:
    """Post-process model predictions into per-image detection results.

    Args:
        preds: model output dict containing 'conf', 'cls_logits',
               'centers', 'wh', 'angles'
        num_images: batch size
        conf_thresh: confidence threshold
        nms_thresh: OBB NMS threshold

    Returns:
        list of dicts per image with 'obbs', 'scores', 'classes'
    """
    results = []

    for b in range(num_images):
        conf = preds["conf"][b].squeeze(-1)  # (K,)
        cls_logits = preds["cls_logits"][b]  # (K, C)
        cls_probs = torch.sigmoid(cls_logits)  # Match training (binary focal loss)
        cls_scores, cls_ids = cls_probs.max(dim=-1)  # (K,), (K,)
        scores = conf * cls_scores  # (K,)

        mask = scores > conf_thresh
        if mask.sum() == 0:
            results.append(
                {
                    "obbs": np.zeros((0, 5), dtype=np.float32),
                    "scores": np.zeros(0, dtype=np.float32),
                    "classes": np.zeros(0, dtype=np.int64),
                }
            )
            continue

        centers = preds["centers"][b][mask]  # (M, 2)
        wh = preds["wh"][b][mask]  # (M, 2)
        angles = preds["angles"][b][mask]  # (M, 1)
        scores_f = scores[mask]
        cls_ids_f = cls_ids[mask]

        obbs = torch.cat([centers, wh, angles], dim=-1).cpu().numpy()
        scores_np = scores_f.cpu().numpy()
        cls_np = cls_ids_f.cpu().numpy()

        # Per-class NMS
        keep_all = []
        for c in np.unique(cls_np):
            c_mask = cls_np == c
            c_obbs = obbs[c_mask]
            c_scores = scores_np[c_mask]
            c_indices = np.where(c_mask)[0]
            keep = obb_nms(c_obbs, c_scores, nms_thresh)
            keep_all.extend(c_indices[keep].tolist())

        keep_all = np.array(keep_all, dtype=np.int64)
        results.append(
            {
                "obbs": obbs[keep_all],
                "scores": scores_np[keep_all],
                "classes": cls_np[keep_all],
            }
        )

    return results
