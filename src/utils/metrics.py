"""Evaluation metrics for oriented object detection."""

import numpy as np

from .obb_utils import obb_iou


def compute_ap(
    pred_obbs: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    pred_img_ids: np.ndarray,
    gt_obbs: np.ndarray,
    gt_classes: np.ndarray,
    gt_img_ids: np.ndarray,
    iou_thresh: float = 0.5,
    num_classes: int = 15,
) -> dict:
    """Compute Average Precision per class with per-image matching.

    Predictions can only match ground truth within the same image,
    preventing cross-image false positives.

    Args:
        pred_obbs: (P, 5) predicted OBBs
        pred_scores: (P,) confidence scores
        pred_classes: (P,) predicted class indices
        pred_img_ids: (P,) image index for each prediction
        gt_obbs: (G, 5) ground truth OBBs
        gt_classes: (G,) ground truth class indices
        gt_img_ids: (G,) image index for each GT
        iou_thresh: IoU threshold for TP
        num_classes: total number of classes
    """
    ap_per_class = {}

    for c in range(num_classes):
        pred_mask = pred_classes == c
        gt_mask = gt_classes == c

        p_obbs = pred_obbs[pred_mask]
        p_scores = pred_scores[pred_mask]
        p_imgs = pred_img_ids[pred_mask]
        g_obbs = gt_obbs[gt_mask]
        g_classes_c = gt_classes[gt_mask]
        g_imgs = gt_img_ids[gt_mask]

        n_gt = len(g_obbs)
        if n_gt == 0 and len(p_obbs) == 0:
            continue
        if n_gt == 0:
            ap_per_class[c] = 0.0
            continue
        if len(p_obbs) == 0:
            ap_per_class[c] = 0.0
            continue

        # Sort predictions by descending confidence
        order = np.argsort(-p_scores)
        p_obbs = p_obbs[order]
        p_scores = p_scores[order]
        p_imgs = p_imgs[order]

        tp = np.zeros(len(p_obbs))
        fp = np.zeros(len(p_obbs))

        # Track which GT boxes have been matched (per image)
        matched_gt = set()

        for i in range(len(p_obbs)):
            img_id = p_imgs[i]

            # Only consider GT from the same image
            same_img_mask = g_imgs == img_id
            if not same_img_mask.any():
                fp[i] = 1
                continue

            same_img_gt_obbs = g_obbs[same_img_mask]
            same_img_gt_indices = np.where(same_img_mask)[0]

            # Compute IoU with same-image GTs
            ious = obb_iou(p_obbs[i : i + 1], same_img_gt_obbs, exact=True).flatten()
            best_local = ious.argmax()
            best_iou = ious[best_local]
            best_gt_global = same_img_gt_indices[best_local]

            if best_iou >= iou_thresh and best_gt_global not in matched_gt:
                tp[i] = 1
                matched_gt.add(best_gt_global)
            else:
                fp[i] = 1

        # Cumulative PR curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)

        # AP via 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_recall = precision[recall >= t]
            ap += (prec_at_recall.max() if len(prec_at_recall) > 0 else 0.0) / 11.0

        ap_per_class[c] = ap

    mean_ap = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    return {"ap_per_class": ap_per_class, "map": mean_ap}


def compute_map(
    all_predictions: list,
    all_targets: list,
    iou_thresholds: list = None,
    num_classes: int = 15,
) -> dict:
    """Compute mAP across multiple IoU thresholds.

    Maintains per-image boundaries so predictions can only match
    GT within the same image.

    Args:
        all_predictions: list of dicts with 'obbs', 'scores', 'classes'
        all_targets: list of dicts with 'obbs', 'classes'
        iou_thresholds: list of IoU thresholds; defaults to [0.5]
        num_classes: total number of classes
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    if len(all_predictions) == 0:
        return {"map": 0.0, "map_per_thresh": {t: 0.0 for t in iou_thresholds}}

    # Aggregate with per-image indices
    pred_obbs_list, pred_scores_list, pred_classes_list, pred_img_list = [], [], [], []
    gt_obbs_list, gt_classes_list, gt_img_list = [], [], []

    for img_idx, (p, t) in enumerate(zip(all_predictions, all_targets)):
        if len(p["obbs"]) > 0:
            pred_obbs_list.append(p["obbs"])
            pred_scores_list.append(p["scores"])
            pred_classes_list.append(p["classes"])
            pred_img_list.append(np.full(len(p["obbs"]), img_idx, dtype=np.int64))
        if len(t["obbs"]) > 0:
            gt_obbs_list.append(t["obbs"])
            gt_classes_list.append(t["classes"])
            gt_img_list.append(np.full(len(t["obbs"]), img_idx, dtype=np.int64))

    pred_obbs = (
        np.concatenate(pred_obbs_list, axis=0) if pred_obbs_list else np.zeros((0, 5))
    )
    pred_scores = np.concatenate(pred_scores_list) if pred_scores_list else np.zeros(0)
    pred_classes = (
        np.concatenate(pred_classes_list)
        if pred_classes_list
        else np.zeros(0, dtype=np.int64)
    )
    pred_img_ids = (
        np.concatenate(pred_img_list) if pred_img_list else np.zeros(0, dtype=np.int64)
    )

    gt_obbs = np.concatenate(gt_obbs_list, axis=0) if gt_obbs_list else np.zeros((0, 5))
    gt_classes = (
        np.concatenate(gt_classes_list)
        if gt_classes_list
        else np.zeros(0, dtype=np.int64)
    )
    gt_img_ids = (
        np.concatenate(gt_img_list) if gt_img_list else np.zeros(0, dtype=np.int64)
    )

    map_per_thresh = {}
    for t in iou_thresholds:
        result = compute_ap(
            pred_obbs,
            pred_scores,
            pred_classes,
            pred_img_ids,
            gt_obbs,
            gt_classes,
            gt_img_ids,
            iou_thresh=t,
            num_classes=num_classes,
        )
        map_per_thresh[t] = result["map"]

    mean_map = np.mean(list(map_per_thresh.values())) if map_per_thresh else 0.0
    return {"map": mean_map, "map_per_thresh": map_per_thresh}
