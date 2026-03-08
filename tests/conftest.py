"""Shared fixtures for the test suite."""

import sys
import os

import pytest
import torch
import numpy as np

# Add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
FEAT_CHANNELS = 64
NUM_PROPOSALS = 10
IMG_SIZE = 128
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_images(device):
    """Random (B, 3, H, W) input batch."""
    return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)


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
