"""Tests for AP and mAP computation in utils/metrics.py."""

import numpy as np
import pytest

from utils.metrics import compute_ap, compute_map


class TestComputeAP:
    """Verify AP calculation with known-answer scenarios."""

    def test_perfect_detector(self):
        """All predictions match GT exactly → AP ≈ 1.0."""
        gt_obbs = np.array([[100, 100, 40, 20, 0.1], [200, 200, 30, 30, 0.0]])
        gt_classes = np.array([0, 0])
        gt_imgs = np.array([0, 0])

        pred_obbs = gt_obbs.copy()
        pred_scores = np.array([0.9, 0.8])
        pred_classes = np.array([0, 0])
        pred_imgs = np.array([0, 0])

        result = compute_ap(
            pred_obbs,
            pred_scores,
            pred_classes,
            pred_imgs,
            gt_obbs,
            gt_classes,
            gt_imgs,
            iou_thresh=0.5,
            num_classes=1,
        )
        assert result["map"] > 0.95

    def test_no_predictions(self):
        """No predictions → AP = 0."""
        result = compute_ap(
            np.zeros((0, 5)),
            np.zeros(0),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.array([[100, 100, 40, 20, 0.0]]),
            np.array([0]),
            np.array([0]),
            iou_thresh=0.5,
            num_classes=1,
        )
        assert result["map"] == 0.0

    def test_all_false_positives(self):
        """Predictions far from any GT → AP = 0."""
        gt_obbs = np.array([[100, 100, 40, 20, 0.0]])
        pred_obbs = np.array([[900, 900, 40, 20, 0.0]])

        result = compute_ap(
            pred_obbs,
            np.array([0.9]),
            np.array([0]),
            np.array([0]),
            gt_obbs,
            np.array([0]),
            np.array([0]),
            iou_thresh=0.5,
            num_classes=1,
        )
        assert result["map"] == 0.0

    def test_cross_image_no_match(self):
        """Predictions and GT in different images should not match."""
        gt_obbs = np.array([[100, 100, 40, 20, 0.0]])
        pred_obbs = gt_obbs.copy()

        result = compute_ap(
            pred_obbs,
            np.array([0.9]),
            np.array([0]),
            np.array([0]),
            gt_obbs,
            np.array([0]),
            np.array([1]),  # GT in image 1, pred in image 0
            iou_thresh=0.5,
            num_classes=1,
        )
        assert result["map"] == 0.0

    def test_no_gt_no_preds(self):
        """Empty GT and empty predictions → empty result."""
        result = compute_ap(
            np.zeros((0, 5)),
            np.zeros(0),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 5)),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            iou_thresh=0.5,
            num_classes=1,
        )
        assert result["map"] == 0.0


class TestComputeMAP:
    """Verify mAP wrapper with multiple IoU thresholds."""

    def test_empty_inputs(self):
        result = compute_map([], [], num_classes=1)
        assert result["map"] == 0.0

    def test_multiple_thresholds(self):
        preds = [
            {
                "obbs": np.array([[100, 100, 40, 20, 0.0]]),
                "scores": np.array([0.9]),
                "classes": np.array([0]),
            }
        ]
        targets = [
            {"obbs": np.array([[100, 100, 40, 20, 0.0]]), "classes": np.array([0])}
        ]
        result = compute_map(preds, targets, iou_thresholds=[0.5, 0.75], num_classes=1)
        assert 0.5 in result["map_per_thresh"]
        assert 0.75 in result["map_per_thresh"]
