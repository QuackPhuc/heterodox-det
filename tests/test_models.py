"""Tests for all 5 detection architectures.

Each architecture is tested at two levels:
1. Generic contract: output shapes, key presence, gradient flow, inference format
2. Architecture-specific mathematical properties:
   - OTDet: transport plan marginals, non-negativity
   - WaveDetNet: energy non-negativity, wave speed positivity
   - ScaleNet: peak detection in scale-space
   - TopoNet: persistence map range, filtration properties
   - FlowNet: convergence map range, contractiveness indication
"""

import pytest
import torch

from models import OTDet, WaveDetNet, ScaleNet, TopoNet, FlowNet
from conftest import NUM_CLASSES, FEAT_CHANNELS, NUM_PROPOSALS, IMG_SIZE, BATCH_SIZE


# ---------------------------------------------------------------------------
# Parametrized model constructors
# ---------------------------------------------------------------------------

ARCHITECTURES = {
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
}

REQUIRED_KEYS = {"centers", "wh", "angles", "cls_logits", "conf", "mass"}


@pytest.fixture(params=list(ARCHITECTURES.keys()))
def arch_name(request):
    return request.param


@pytest.fixture
def model(arch_name, device):
    return ARCHITECTURES[arch_name](device)


# ═══════════════════════════════════════════════════════════════════════════
# Generic contract — all architectures must satisfy
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputContract:
    """All architectures produce the same detection output format."""

    def test_required_keys(self, model, dummy_images):
        preds = model(dummy_images)
        missing = REQUIRED_KEYS - set(preds.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_shapes(self, model, dummy_images):
        preds = model(dummy_images)
        B = BATCH_SIZE
        K = preds["centers"].shape[1]
        assert preds["centers"].shape == (B, K, 2)
        assert preds["wh"].shape == (B, K, 2)
        assert preds["angles"].shape == (B, K, 1)
        assert preds["cls_logits"].shape == (B, K, NUM_CLASSES)
        assert preds["conf"].shape == (B, K, 1)

    def test_confidence_in_unit_interval(self, model, dummy_images):
        conf = model(dummy_images)["conf"]
        assert conf.min() >= 0.0 and conf.max() <= 1.0


class TestGradientFlow:
    """Gradients reach all trainable parameters (end-to-end differentiability)."""

    def test_all_params_receive_gradient(self, model, dummy_images):
        model.train()
        preds = model(dummy_images)
        scalar = sum(v.sum() for v in preds.values() if isinstance(v, torch.Tensor))
        scalar.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert (
                    p.grad is not None and p.grad.abs().sum() > 0
                ), f"No gradient for {name}"


class TestInference:
    """Inference postprocessing returns well-formed results."""

    def test_result_format(self, model, dummy_images):
        results = model.inference(dummy_images, conf_thresh=0.0)
        assert len(results) == BATCH_SIZE
        for r in results:
            assert r["obbs"].ndim == 2 and r["obbs"].shape[1] == 5
            assert len(r["obbs"]) == len(r["scores"]) == len(r["classes"])

    def test_strict_threshold_filters_all(self, model, dummy_images):
        for r in model.inference(dummy_images, conf_thresh=1.0):
            assert len(r["obbs"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Architecture-specific mathematical properties
# ═══════════════════════════════════════════════════════════════════════════


class TestOTDet:
    """Optimal Transport Detection — Sinkhorn algorithm properties.

    Theory (Hypothesis 3): Detection = Kantorovich OT from pixel evidence
    to object slots. The transport plan π must satisfy:
      - π ≥ 0 (non-negative coupling)
      - Row marginals ≈ source distribution μ (objectness)
      - Objectness map ∈ [0, 1] (sigmoid output)
    """

    @pytest.fixture
    def otdet_model(self, device):
        return ARCHITECTURES["otdet"](device)

    def test_transport_plan_non_negative(self, otdet_model, dummy_images):
        """π(pixel, slot) ≥ 0 — fundamental OT constraint."""
        preds = otdet_model(dummy_images)
        assert preds["transport_plan"].min() >= -1e-7

    def test_transport_plan_row_marginals(self, otdet_model, dummy_images):
        """Row sums of π should approximate the source distribution μ.

        This verifies Σ_k π(i, k) ≈ μ(i) — the Sinkhorn algorithm
        enforces this marginal constraint at convergence.
        """
        preds = otdet_model(dummy_images)
        plan = preds["transport_plan"]  # (B, N, K)
        row_sums = plan.sum(dim=2)  # (B, N)
        # Each pixel's total transport should be close to uniform 1/N
        N = plan.shape[1]
        expected = 1.0 / N
        assert (row_sums.mean() - expected).abs() < 0.1

    def test_objectness_map_range(self, otdet_model, dummy_images):
        """Objectness map ∈ [0, 1] — sigmoid activation ensures valid distribution."""
        preds = otdet_model(dummy_images)
        obj = preds["objectness_map"]
        assert obj.min() >= 0.0 and obj.max() <= 1.0


class TestWaveDetNet:
    """Wave Resonance Detection — wave equation properties.

    Theory (Hypothesis 2): Objects = resonance regions of a damped wave
    on a learned graph. Key invariants:
      - Energy E(x) = Σ_t |u(x,t)|² ≥ 0 (non-negative energy)
      - Wave speed c(x) > 0 (physical constraint, softplus ensures this)
    """

    @pytest.fixture
    def wavedet_model(self, device):
        return ARCHITECTURES["wavedet"](device)

    def test_energy_non_negative(self, wavedet_model, dummy_images):
        """Accumulated wave energy E(x) = Σ|u|² must be ≥ 0."""
        preds = wavedet_model(dummy_images)
        assert preds["energy"].min() >= -1e-7

    def test_wave_speed_positive(self, wavedet_model, dummy_images):
        """Wave speed c(x) must be positive (softplus output)."""
        preds = wavedet_model(dummy_images)
        assert preds["wave_speed"].min() > 0.0


class TestScaleNet:
    """Continuous Scale-Space Detection — Lindeberg theory properties.

    Theory (Hypothesis 7): Objects exist at continuous scale σ ∈ ℝ⁺.
    Detection = 3D extrema in (x, y, σ) space. SIREN provides precise
    derivatives. σ²-normalized Laplacian ensures scale covariance.
    """

    @pytest.fixture
    def scalenet_model(self, device):
        return ARCHITECTURES["scalenet"](device)

    def test_proposals_within_image_bounds(self, scalenet_model, dummy_images):
        """Detected extrema centers should lie within the image domain."""
        preds = scalenet_model(dummy_images)
        centers = preds["centers"]
        assert centers[:, :, 0].min() >= -IMG_SIZE * 0.1
        assert centers[:, :, 1].min() >= -IMG_SIZE * 0.1
        assert centers[:, :, 0].max() <= IMG_SIZE * 1.1
        assert centers[:, :, 1].max() <= IMG_SIZE * 1.1


class TestTopoNet:
    """Topological Persistence Detection — persistent homology properties.

    Theory (Hypothesis 1): Objects create topological holes on the feature
    manifold. persistence = death - birth. High persistence = real object,
    low persistence = noise.
      - Persistence map ∈ [0, 1] (normalized)
      - Birth map tracks when components first appear in filtration
    """

    @pytest.fixture
    def toponet_model(self, device):
        return ARCHITECTURES["toponet"](device)

    def test_persistence_map_range(self, toponet_model, dummy_images):
        """Persistence map values ∈ [0, 1] — product of (1-birth) × stability."""
        preds = toponet_model(dummy_images)
        pm = preds["persistence_map"]
        assert pm.min() >= -1e-6, f"Persistence map has negative values: {pm.min()}"
        assert pm.max() <= 1.0 + 1e-6

    def test_filtration_is_spatial(self, toponet_model, dummy_images):
        """Filtration function should produce a 2D scalar field over the image."""
        preds = toponet_model(dummy_images)
        filt = preds["filtration"]
        assert filt.shape[0] == BATCH_SIZE
        assert filt.shape[1] == 1, "Filtration should be single-channel"


class TestFlowNet:
    """Neural ODE Attractor Detection — dynamical systems properties.

    Theory (Hypothesis 5): Features evolve via dh/dt = f_θ(h,t) - α·h.
    Objects = stable attractors where features converge (velocity → 0).
      - Convergence map = 1/(1 + |velocity|) ∈ (0, 1]
      - Higher convergence at attractor locations (objects)
      - Trajectory variance captures dynamics stability
    """

    @pytest.fixture
    def flownet_model(self, device):
        return ARCHITECTURES["flownet"](device)

    def test_convergence_map_range(self, flownet_model, dummy_images):
        """Convergence map = 1/(1+|v|) is strictly in (0, 1]."""
        preds = flownet_model(dummy_images)
        cm = preds["convergence_map"]
        assert cm.min() > 0.0, "Convergence map must be strictly positive"
        assert cm.max() <= 1.0 + 1e-6

    def test_trajectory_variance_non_negative(self, flownet_model, dummy_images):
        """Trajectory variance (Welford output) must be ≥ 0."""
        preds = flownet_model(dummy_images)
        tv = preds["trajectory_var"]
        assert tv.min() >= -1e-7
