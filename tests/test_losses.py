"""Tests for OTDetLoss and PeakDetLoss."""

import pytest
import torch

from losses import OTDetLoss, PeakDetLoss
from conftest import NUM_CLASSES, NUM_PROPOSALS, IMG_SIZE, BATCH_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_preds(device, keys_extra=None):
    """Synthesize minimal model-like prediction dict."""
    K = NUM_PROPOSALS
    B = BATCH_SIZE
    preds = {
        "centers": torch.randn(B, K, 2, device=device) * IMG_SIZE / 2 + IMG_SIZE / 2,
        "wh": torch.randn(B, K, 2, device=device).abs() * 20 + 5,
        "angles": torch.randn(B, K, 1, device=device) * 0.3,
        "cls_logits": torch.randn(B, K, NUM_CLASSES, device=device),
        "conf": torch.sigmoid(torch.randn(B, K, 1, device=device)),
        "mass": torch.rand(B, K, device=device),
    }
    if keys_extra:
        for k, shape in keys_extra.items():
            preds[k] = torch.randn(*shape, device=device)
    # Ensure gradients flow
    for v in preds.values():
        v.requires_grad_(True)
    return preds


# ---------------------------------------------------------------------------
# OTDetLoss
# ---------------------------------------------------------------------------


class TestOTDetLoss:

    @pytest.fixture
    def criterion(self):
        return OTDetLoss(num_classes=NUM_CLASSES)

    def test_output_keys(self, criterion, dummy_targets, device):
        preds = _make_preds(
            device,
            keys_extra={
                "objectness_map": (BATCH_SIZE, 1, 16, 16),
                "transport_plan": (BATCH_SIZE, 256, NUM_PROPOSALS),
            },
        )
        loss_dict = criterion(preds, dummy_targets, img_size=IMG_SIZE)
        for key in ("total", "cls", "reg", "angle", "iou", "objectness"):
            assert key in loss_dict, f"Missing loss key: {key}"

    def test_total_is_finite(self, criterion, dummy_targets, device):
        preds = _make_preds(
            device,
            keys_extra={
                "objectness_map": (BATCH_SIZE, 1, 16, 16),
                "transport_plan": (BATCH_SIZE, 256, NUM_PROPOSALS),
            },
        )
        loss = criterion(preds, dummy_targets, img_size=IMG_SIZE)["total"]
        assert torch.isfinite(loss), "Loss is not finite"

    def test_total_is_positive(self, criterion, dummy_targets, device):
        preds = _make_preds(
            device,
            keys_extra={
                "objectness_map": (BATCH_SIZE, 1, 16, 16),
                "transport_plan": (BATCH_SIZE, 256, NUM_PROPOSALS),
            },
        )
        loss = criterion(preds, dummy_targets, img_size=IMG_SIZE)["total"]
        assert loss > 0, "Total loss should be positive for random predictions"

    def test_zero_gt_does_not_crash(self, criterion, empty_targets, device):
        preds = _make_preds(
            device,
            keys_extra={
                "objectness_map": (BATCH_SIZE, 1, 16, 16),
                "transport_plan": (BATCH_SIZE, 256, NUM_PROPOSALS),
            },
        )
        loss = criterion(preds, empty_targets, img_size=IMG_SIZE)["total"]
        assert torch.isfinite(loss)

    def test_backward(self, criterion, dummy_targets, device):
        preds = _make_preds(
            device,
            keys_extra={
                "objectness_map": (BATCH_SIZE, 1, 16, 16),
                "transport_plan": (BATCH_SIZE, 256, NUM_PROPOSALS),
            },
        )
        loss = criterion(preds, dummy_targets, img_size=IMG_SIZE)["total"]
        loss.backward()
        assert preds["centers"].grad is not None


# ---------------------------------------------------------------------------
# PeakDetLoss
# ---------------------------------------------------------------------------


class TestPeakDetLoss:

    @pytest.fixture
    def criterion(self):
        return PeakDetLoss(num_classes=NUM_CLASSES)

    def test_output_keys(self, criterion, dummy_targets, device):
        preds = _make_preds(device)
        loss_dict = criterion(preds, dummy_targets, img_size=IMG_SIZE)
        for key in ("total", "cls", "reg", "angle", "iou", "objectness"):
            assert key in loss_dict, f"Missing loss key: {key}"

    def test_total_is_finite(self, criterion, dummy_targets, device):
        preds = _make_preds(device)
        loss = criterion(preds, dummy_targets, img_size=IMG_SIZE)["total"]
        assert torch.isfinite(loss)

    def test_zero_gt_does_not_crash(self, criterion, empty_targets, device):
        preds = _make_preds(device)
        loss = criterion(preds, empty_targets, img_size=IMG_SIZE)["total"]
        assert torch.isfinite(loss)

    def test_backward(self, criterion, dummy_targets, device):
        preds = _make_preds(device)
        loss = criterion(preds, dummy_targets, img_size=IMG_SIZE)["total"]
        loss.backward()
        assert preds["cls_logits"].grad is not None
