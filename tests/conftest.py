"""Shared fixtures and constants for the test suite."""

import sys
import os

import pytest
import torch
import numpy as np

from typing import Callable, Dict

# Add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from models import OTDet, WaveDetNet, ScaleNet, TopoNet, FlowNet, InfoGeoNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
FEAT_CHANNELS = 64
NUM_PROPOSALS = 10
IMG_SIZE = 128
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Shared model constructors (single source of truth)
# ---------------------------------------------------------------------------

ARCHITECTURES: Dict[str, Callable] = {
    "otdet": lambda dev: OTDet(
        num_classes=NUM_CLASSES,
        num_slots=NUM_PROPOSALS,
        feat_channels=FEAT_CHANNELS,
        pretrained_backbone=False,
        sinkhorn_iters=5,
        img_size=IMG_SIZE,
    ).to(dev),
    "wavedet": lambda dev: WaveDetNet(
        num_classes=NUM_CLASSES,
        feat_channels=FEAT_CHANNELS,
        num_proposals=NUM_PROPOSALS,
        num_wave_steps=4,
        img_size=IMG_SIZE,
    ).to(dev),
    "scalenet": lambda dev: ScaleNet(
        num_classes=NUM_CLASSES,
        feat_channels=FEAT_CHANNELS,
        num_proposals=NUM_PROPOSALS,
        num_scales=4,
        sigma_range=(0.5, 4.0),
        img_size=IMG_SIZE,
    ).to(dev),
    "toponet": lambda dev: TopoNet(
        num_classes=NUM_CLASSES,
        feat_channels=FEAT_CHANNELS,
        num_proposals=NUM_PROPOSALS,
        num_filtration_steps=4,
        img_size=IMG_SIZE,
    ).to(dev),
    "flownet": lambda dev: FlowNet(
        num_classes=NUM_CLASSES,
        feat_channels=FEAT_CHANNELS,
        num_proposals=NUM_PROPOSALS,
        ode_steps=4,
        img_size=IMG_SIZE,
    ).to(dev),
    "infogeonet": lambda dev: InfoGeoNet(
        num_classes=NUM_CLASSES,
        feat_channels=FEAT_CHANNELS,
        num_proposals=NUM_PROPOSALS,
        num_fisher_samples=0,
        img_size=IMG_SIZE,
    ).to(dev),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_images(device):
    """Deterministic random (B, 3, H, W) input batch."""
    gen = torch.Generator(device=device).manual_seed(123)
    return torch.randn(
        BATCH_SIZE,
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=device,
        generator=gen,
    )


@pytest.fixture
def dummy_targets():
    """Per-image GT targets with deterministic random OBBs."""
    rng = np.random.default_rng(42)
    targets = []
    for _ in range(BATCH_SIZE):
        n_gt = rng.integers(1, 4)
        cx = rng.uniform(20, IMG_SIZE - 20, n_gt)
        cy = rng.uniform(20, IMG_SIZE - 20, n_gt)
        w = rng.uniform(10, 40, n_gt)
        h = rng.uniform(10, 40, n_gt)
        a = rng.uniform(-np.pi / 4, np.pi / 4, n_gt)
        obbs = np.stack([cx, cy, w, h, a], axis=1).astype(np.float32)
        classes = rng.integers(0, NUM_CLASSES, n_gt).astype(np.int64)
        targets.append(
            {
                "obbs": torch.from_numpy(obbs),
                "classes": torch.from_numpy(classes),
            }
        )
    return targets


@pytest.fixture
def empty_targets():
    """Per-image targets with zero GT objects (edge case)."""
    return [
        {
            "obbs": torch.zeros(0, 5),
            "classes": torch.zeros(0, dtype=torch.long),
        }
        for _ in range(BATCH_SIZE)
    ]
