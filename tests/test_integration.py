"""Integration tests: model → loss end-to-end shape compatibility."""

import pytest
import torch

from losses import OTDetLoss, PeakDetLoss
from conftest import ARCHITECTURES, NUM_CLASSES, IMG_SIZE


# OTDet uses its own loss; all others use PeakDetLoss
_LOSS_MAP = {
    "otdet": lambda: OTDetLoss(num_classes=NUM_CLASSES),
}
_DEFAULT_LOSS = lambda: PeakDetLoss(num_classes=NUM_CLASSES)


@pytest.fixture(params=list(ARCHITECTURES.keys()))
def model_and_loss(request, device):
    arch = request.param
    model = ARCHITECTURES[arch](device)
    loss_fn = _LOSS_MAP.get(arch, _DEFAULT_LOSS)()
    return model, loss_fn


class TestModelLossIntegration:
    """Model output passes directly to loss without shape or key errors."""

    def test_forward_loss_finite(self, model_and_loss, dummy_images, dummy_targets):
        model, criterion = model_and_loss
        model.train()
        preds = model(dummy_images)
        loss_dict = criterion(preds, dummy_targets, img_size=IMG_SIZE)
        assert torch.isfinite(loss_dict["total"]), "Loss is not finite"
        assert loss_dict["total"] > 0, "Loss should be positive for random init"

    def test_backward_updates_params(self, model_and_loss, dummy_images, dummy_targets):
        model, criterion = model_and_loss
        model.train()
        preds = model(dummy_images)
        loss = criterion(preds, dummy_targets, img_size=IMG_SIZE)["total"]
        loss.backward()
        graded = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert graded > 0, "No parameters received gradients"

    def test_zero_gt_no_crash(self, model_and_loss, dummy_images, empty_targets):
        model, criterion = model_and_loss
        model.train()
        preds = model(dummy_images)
        loss = criterion(preds, empty_targets, img_size=IMG_SIZE)["total"]
        assert torch.isfinite(loss)
