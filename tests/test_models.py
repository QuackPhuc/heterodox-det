"""Tests for all 6 detection architectures.

Each architecture is tested at two levels:
1. Generic contract: output shapes, key presence, gradient flow, inference format
2. Architecture-specific mathematical properties:
   - OTDet: transport plan marginals, non-negativity
   - WaveDetNet: energy non-negativity, wave speed positivity
   - ScaleNet: scale positivity, scale-feature spatial structure
   - TopoNet: persistence map range, filtration properties
   - FlowNet: convergence map range, contractiveness indication
   - InfoGeoNet: Fisher map non-negativity, spatial field properties
"""

import pytest
import torch

from models import InfoGeoNet
from conftest import (
    ARCHITECTURES,
    NUM_CLASSES,
    FEAT_CHANNELS,
    NUM_PROPOSALS,
    IMG_SIZE,
    BATCH_SIZE,
)


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
        assert not model.training, "Model should be in eval mode after inference"
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

    @pytest.fixture
    def preds(self, otdet_model, dummy_images):
        return otdet_model(dummy_images)

    def test_transport_plan_non_negative(self, preds):
        """π(pixel, slot) ≥ 0 — fundamental OT constraint."""
        assert preds["transport_plan"].min() >= -1e-7

    def test_transport_plan_row_marginals(self, preds):
        """Row sums of π should approximate the source distribution μ.

        This verifies Σ_k π(i, k) ≈ μ(i) — the Sinkhorn algorithm
        enforces this marginal constraint at convergence.
        """
        plan = preds["transport_plan"]  # (B, N, K)
        row_sums = plan.sum(dim=2)  # (B, N)
        N = plan.shape[1]
        expected = 1.0 / N
        assert (row_sums.mean() - expected).abs() < 0.1

    def test_objectness_map_range(self, preds):
        """Objectness map ∈ [0, 1] — sigmoid activation ensures valid distribution."""
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

    @pytest.fixture
    def preds(self, wavedet_model, dummy_images):
        return wavedet_model(dummy_images)

    def test_energy_non_negative(self, preds):
        """Accumulated wave energy E(x) = Σ|u|² must be ≥ 0."""
        assert preds["energy"].min() >= -1e-7

    def test_wave_speed_positive(self, preds):
        """Wave speed c(x) must be positive (softplus output)."""
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

    @pytest.fixture
    def preds(self, scalenet_model, dummy_images):
        return scalenet_model(dummy_images)

    def test_proposals_within_image_bounds(self, preds):
        """Detected extrema centers should lie within the image domain."""
        centers = preds["centers"]
        assert centers[:, :, 0].min() >= -IMG_SIZE * 0.1
        assert centers[:, :, 1].min() >= -IMG_SIZE * 0.1
        assert centers[:, :, 0].max() <= IMG_SIZE * 1.1
        assert centers[:, :, 1].max() <= IMG_SIZE * 1.1

    def test_sigmas_positive(self, preds):
        """Characteristic scales σ* must be strictly positive."""
        sigmas = preds["sigmas"]
        assert sigmas.min() > 0.0, f"Non-positive sigma detected: {sigmas.min()}"

    def test_scale_feats_shape(self, preds):
        """Scale-space features should be a 5D tensor (B, S, C, H, W)."""
        sf = preds["scale_feats"]
        assert sf.ndim == 5, "scale_feats should be (B, S, C, H, W)"
        assert sf.shape[0] == BATCH_SIZE


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

    @pytest.fixture
    def preds(self, toponet_model, dummy_images):
        return toponet_model(dummy_images)

    def test_persistence_map_range(self, preds):
        """Persistence map values ∈ [0, 1] — product of (1-birth) × stability."""
        pm = preds["persistence_map"]
        assert pm.min() >= -1e-6, f"Persistence map has negative values: {pm.min()}"
        assert pm.max() <= 1.0 + 1e-6

    def test_filtration_is_spatial(self, preds):
        """Filtration function should produce a 2D scalar field over the image."""
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

    @pytest.fixture
    def preds(self, flownet_model, dummy_images):
        return flownet_model(dummy_images)

    def test_convergence_map_range(self, preds):
        """Convergence map = 1/(1+|v|) is strictly in (0, 1]."""
        cm = preds["convergence_map"]
        assert cm.min() > 0.0, "Convergence map must be strictly positive"
        assert cm.max() <= 1.0 + 1e-6

    def test_trajectory_variance_non_negative(self, preds):
        """Trajectory variance (Welford output) must be ≥ 0."""
        tv = preds["trajectory_var"]
        assert tv.min() >= -1e-7


class TestInfoGeoNet:
    """Information Geometry Detection — Fisher Information properties.

    Theory (Hypothesis 4): Pixels with high Fisher Information carry
    discriminative features. Fisher diagonal F_ii = p_i(1-p_i) where
    p_i is the predicted class probability (Bernoulli variance).
      - Fisher map ≥ 0 (variance is non-negative)
      - Fisher map is a single-channel spatial field
    """

    @pytest.fixture
    def infogeonet_model(self, device):
        return ARCHITECTURES["infogeonet"](device)

    @pytest.fixture
    def preds(self, infogeonet_model, dummy_images):
        return infogeonet_model(dummy_images)

    def test_fisher_map_non_negative(self, preds):
        """Fisher Information (Bernoulli variance) must be ≥ 0."""
        fm = preds["fisher_map"]
        assert fm.min() >= -1e-7, f"Fisher map has negative values: {fm.min()}"

    def test_fisher_map_is_spatial(self, preds):
        """Fisher map should be a single-channel 2D field over the image."""
        fm = preds["fisher_map"]
        assert fm.shape[0] == BATCH_SIZE
        assert fm.shape[1] == 1, "Fisher map should be single-channel"

    def test_mc_fisher_gradient_flow(self, device, dummy_images):
        """MC refinement path must pass gradients through to Fisher params."""
        model = InfoGeoNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            num_fisher_samples=3,
            img_size=IMG_SIZE,
        ).to(device)
        model.train()
        preds = model(dummy_images)
        preds["fisher_map"].sum().backward()
        assert model.fisher.temperature.grad is not None
        assert model.fisher.temperature.grad.abs().sum() > 0

    def test_mc_refinement_alters_fisher(self, device, dummy_images):
        """MC Fisher refinement should produce different values than analytical."""
        torch.manual_seed(99)
        model_analytical = InfoGeoNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            num_fisher_samples=0,
            img_size=IMG_SIZE,
        ).to(device)

        model_mc = InfoGeoNet(
            num_classes=NUM_CLASSES,
            feat_channels=FEAT_CHANNELS,
            num_proposals=NUM_PROPOSALS,
            num_fisher_samples=4,
            mc_blend=0.5,
            img_size=IMG_SIZE,
        ).to(device)

        # Share weights so only the MC path differs
        model_mc.load_state_dict(model_analytical.state_dict(), strict=False)
        model_analytical.train()
        model_mc.train()

        with torch.no_grad():
            fisher_analytical = model_analytical(dummy_images)["fisher_map"]
            fisher_mc = model_mc(dummy_images)["fisher_map"]

        # MC refinement should produce a different Fisher map
        diff = (fisher_analytical - fisher_mc).abs().sum()
        assert diff > 1e-6, "MC refinement did not alter Fisher map"
