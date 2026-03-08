"""Tests for the Sinkhorn OT module."""

import math

import pytest
import torch

from models.otdet.ot_module import SinkhornOT, ObjectSlots


class TestSinkhornOT:

    @pytest.fixture
    def sinkhorn(self):
        return SinkhornOT(num_iters=20, eps=0.05, cost_type="l2")

    def test_output_shape(self, sinkhorn, device):
        B, N, K, D = 2, 64, 10, 32
        pf = torch.randn(B, N, D, device=device)
        sf = torch.randn(B, K, D, device=device)
        plan = sinkhorn(pf, sf)
        assert plan.shape == (B, N, K)

    def test_plan_is_non_negative(self, sinkhorn, device):
        pf = torch.randn(2, 64, 32, device=device)
        sf = torch.randn(2, 10, 32, device=device)
        plan = sinkhorn(pf, sf)
        assert plan.min() >= -1e-7

    def test_row_marginal(self, sinkhorn, device):
        """Row sums of π should approximate the source distribution μ."""
        B, N, K, D = 1, 32, 8, 16
        pf = torch.randn(B, N, D, device=device)
        sf = torch.randn(B, K, D, device=device)
        # Uniform source
        plan = sinkhorn(pf, sf)
        row_sums = plan.sum(dim=2)  # (B, N)
        # Should be approximately uniform: 1/N each
        expected = 1.0 / N
        assert (row_sums - expected).abs().max() < 0.1

    def test_gradient_flow(self, sinkhorn, device):
        pf = torch.randn(2, 32, 16, device=device, requires_grad=True)
        sf = torch.randn(2, 8, 16, device=device, requires_grad=True)
        plan = sinkhorn(pf, sf)
        plan.sum().backward()
        assert pf.grad is not None
        assert sf.grad is not None

    def test_cosine_cost(self, device):
        sinkhorn_cos = SinkhornOT(num_iters=10, eps=0.1, cost_type="cosine")
        pf = torch.randn(2, 32, 16, device=device)
        sf = torch.randn(2, 8, 16, device=device)
        plan = sinkhorn_cos(pf, sf)
        assert plan.shape == (2, 32, 8)
        assert plan.min() >= -1e-7


class TestObjectSlots:

    def test_shape(self):
        slots = ObjectSlots(num_slots=20, feat_dim=64)
        out = slots(batch_size=4)
        assert out.shape == (4, 20, 64)

    def test_shared_across_batch(self):
        """All batch entries should be identical (broadcasting)."""
        slots = ObjectSlots(num_slots=10, feat_dim=32)
        out = slots(batch_size=3)
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[1], out[2])
