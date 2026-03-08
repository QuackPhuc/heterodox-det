"""Integration tests: model → loss end-to-end shape compatibility."""

import pytest
import torch

from models import OTDet, WaveDetNet, ScaleNet, TopoNet, FlowNet
from losses import OTDetLoss, PeakDetLoss
from conftest import NUM_CLASSES, FEAT_CHANNELS, NUM_PROPOSALS, IMG_SIZE, BATCH_SIZE


# ---------------------------------------------------------------------------
# Model-Loss pairs
# ---------------------------------------------------------------------------

MODEL_LOSS_PAIRS = {
    "otdet": (
        lambda dev: OTDet(
            num_classes=NUM_CLASSES,
            num_slots=NUM_PROPOSALS,
            feat_channels=FEAT_CHANNELS,
            pretrained_backbone=False,
            sinkhorn_iters=5,
            img_size=IMG_SIZE,
        ).to(dev),
        lambda: OTDetLoss(num_classes=NUM_CLASSES),
    ),
    "wavedet": (
        lambda dev: WaveDetNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            num_wave_steps=4,
            img_size=IMG_SIZE,
        ).to(dev),
        lambda: PeakDetLoss(num_classes=NUM_CLASSES),
    ),
    "scalenet": (
        lambda dev: ScaleNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            num_scales=4,
            sigma_range=(0.5, 4.0),
            img_size=IMG_SIZE,
        ).to(dev),
        lambda: PeakDetLoss(num_classes=NUM_CLASSES),
    ),
    "toponet": (
        lambda dev: TopoNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            num_filtration_steps=4,
            img_size=IMG_SIZE,
        ).to(dev),
        lambda: PeakDetLoss(num_classes=NUM_CLASSES),
    ),
    "flownet": (
        lambda dev: FlowNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            ode_steps=4,
            img_size=IMG_SIZE,
        ).to(dev),
        lambda: PeakDetLoss(num_classes=NUM_CLASSES),
    ),
}


@pytest.fixture(params=list(MODEL_LOSS_PAIRS.keys()))
def model_and_loss(request, device):
    make_model, make_loss = MODEL_LOSS_PAIRS[request.param]
    return make_model(device), make_loss()


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
        total = sum(1 for p in model.parameters() if p.requires_grad)
        assert graded > 0, "No parameters received gradients"

    def test_zero_gt_no_crash(self, model_and_loss, dummy_images, empty_targets):
        model, criterion = model_and_loss
        model.train()
        preds = model(dummy_images)
        loss = criterion(preds, empty_targets, img_size=IMG_SIZE)["total"]
        assert torch.isfinite(loss)
