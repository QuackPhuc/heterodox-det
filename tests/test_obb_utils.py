"""Tests for OBB geometry utilities.

Validates: conversions, IoU properties, NMS, differentiable IoU.
"""

import math

import numpy as np
import pytest
import torch

from utils.obb_utils import (
    poly_to_obb,
    obb_to_poly,
    obb_iou,
    obb_nms,
    obb_iou_tensor,
)


# ---------------------------------------------------------------------------
# Conversion round-trip
# ---------------------------------------------------------------------------


class TestConversions:

    def test_roundtrip_single(self):
        """poly → obb → poly should recover original (within tolerance)."""
        obb = np.array([100.0, 50.0, 40.0, 20.0, 0.3])
        poly = obb_to_poly(obb)
        recovered = poly_to_obb(poly)
        np.testing.assert_allclose(recovered[:4], obb[:4], atol=1e-4)

    def test_roundtrip_batch(self):
        """Batch conversion preserves center and area (w/h may swap due to angle ambiguity)."""
        obbs = np.array(
            [
                [100, 100, 50, 30, 0.0],
                [200, 200, 40, 60, math.pi / 6],
            ],
            dtype=np.float64,
        )
        polys = obb_to_poly(obbs)
        assert polys.shape == (2, 8)
        recovered = poly_to_obb(polys)
        # Centers should be preserved exactly
        np.testing.assert_allclose(recovered[:, :2], obbs[:, :2], atol=1e-4)
        # Area (w*h) should be preserved (w/h may swap due to angle convention)
        np.testing.assert_allclose(
            recovered[:, 2] * recovered[:, 3], obbs[:, 2] * obbs[:, 3], atol=1e-4
        )

    def test_zero_angle(self):
        """Zero-angle OBB should produce axis-aligned rectangle."""
        obb = np.array([50.0, 50.0, 20.0, 10.0, 0.0])
        poly = obb_to_poly(obb)
        xs = poly[0::2]
        ys = poly[1::2]
        # Width along x, height along y
        assert abs(xs.max() - xs.min() - 20.0) < 1e-6
        assert abs(ys.max() - ys.min() - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# IoU properties
# ---------------------------------------------------------------------------


class TestOBBIoU:

    def test_self_iou_is_one(self):
        """IoU of an OBB with itself should be ~1."""
        obb = np.array([[100.0, 100.0, 40.0, 20.0, 0.2]])
        iou = obb_iou(obb, obb)
        assert iou.shape == (1, 1)
        assert iou[0, 0] > 0.95, f"Self-IoU too low: {iou[0, 0]}"

    def test_symmetry(self):
        """IoU(A, B) == IoU(B, A)."""
        a = np.array([[100, 100, 40, 20, 0.1]], dtype=np.float64)
        b = np.array([[110, 105, 30, 25, -0.2]], dtype=np.float64)
        np.testing.assert_allclose(obb_iou(a, b), obb_iou(b, a).T, atol=1e-6)

    def test_disjoint_is_zero(self):
        """Non-overlapping OBBs should have IoU ≈ 0."""
        a = np.array([[0.0, 0.0, 10.0, 10.0, 0.0]])
        b = np.array([[1000.0, 1000.0, 10.0, 10.0, 0.0]])
        iou = obb_iou(a, b)
        assert iou[0, 0] < 0.01

    def test_iou_range(self):
        """IoU must lie in [0, 1]."""
        rng = np.random.default_rng(42)
        obbs = rng.uniform([50, 50, 10, 10, -0.5], [200, 200, 80, 80, 0.5], (20, 5))
        iou = obb_iou(obbs, obbs)
        assert iou.min() >= -1e-6
        assert iou.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


class TestOBBNMS:

    def test_nms_removes_duplicates(self):
        """Identical OBBs with different scores — keep only the best."""
        obbs = np.tile([100.0, 100.0, 40.0, 20.0, 0.0], (5, 1))
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        keep = obb_nms(obbs, scores, iou_thresh=0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_nms_keeps_separated(self):
        """Well-separated OBBs should all survive NMS."""
        obbs = np.array(
            [
                [50, 50, 20, 20, 0],
                [200, 200, 20, 20, 0],
                [350, 350, 20, 20, 0],
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.8, 0.7])
        keep = obb_nms(obbs, scores, iou_thresh=0.5)
        assert len(keep) == 3

    def test_nms_sorted_by_score(self):
        """NMS should return indices in descending score order."""
        obbs = np.array([[50, 50, 20, 20, 0], [200, 200, 20, 20, 0]], dtype=np.float64)
        scores = np.array([0.3, 0.9])
        keep = obb_nms(obbs, scores, iou_thresh=0.5)
        assert keep[0] == 1, "Highest score should be first"


# ---------------------------------------------------------------------------
# Differentiable IoU (torch)
# ---------------------------------------------------------------------------


class TestOBBIoUTensor:

    def test_self_iou(self, device):
        pred = torch.tensor(
            [[100, 100, 40, 20, 0.2]], dtype=torch.float32, device=device
        )
        iou = obb_iou_tensor(pred, pred)
        assert iou.item() > 0.95

    def test_gradient_exists(self, device):
        pred = torch.tensor(
            [[100, 100, 40, 20, 0.2]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        target = torch.tensor(
            [[105, 102, 38, 22, 0.1]], dtype=torch.float32, device=device
        )
        iou = obb_iou_tensor(pred, target)
        iou.sum().backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_range(self, device):
        rng = torch.manual_seed(42)
        pred = torch.rand(50, 5, device=device) * torch.tensor(
            [200, 200, 80, 80, 1.0], device=device
        ) + torch.tensor([50, 50, 10, 10, -0.5], device=device)
        iou = obb_iou_tensor(pred, pred)
        assert iou.min() >= -1e-3
        assert iou.max() <= 1.0 + 1e-3
