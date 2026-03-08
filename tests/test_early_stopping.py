"""Unit tests for the EarlyStopping utility."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from utils.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestEarlyStoppingMin:
    """Default mode='min' (lower is better, e.g. loss)."""

    def test_no_trigger_while_improving(self):
        es = EarlyStopping(patience=3)
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in losses:
            assert not es(loss)

    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # best = 1.0
        es(1.1)  # counter = 1
        es(1.2)  # counter = 2
        assert es(1.3)  # counter = 3 → stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # best = 1.0
        es(1.1)  # counter = 1
        es(1.2)  # counter = 2
        es(0.9)  # improvement → counter = 0
        assert not es(1.0)  # counter = 1 (not yet)
        assert not es(1.0)  # counter = 2
        assert es(1.0)  # counter = 3 → stop

    def test_min_delta_threshold(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)  # best = 1.0
        # 0.95 is lower, but not by min_delta=0.1
        es(0.95)  # counter = 1
        es(0.92)  # counter = 2
        assert es(0.91)  # counter = 3 → stop

    def test_min_delta_allows_real_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)  # best = 1.0
        assert not es(0.89)  # 1.0 - 0.89 > 0.1 → improvement


class TestEarlyStoppingMax:
    """mode='max' (higher is better, e.g. mAP)."""

    def test_no_trigger_while_improving(self):
        es = EarlyStopping(patience=3, mode="max")
        scores = [0.5, 0.6, 0.7, 0.8]
        for s in scores:
            assert not es(s)

    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=2, mode="max")
        es(0.8)  # best = 0.8
        es(0.7)  # counter = 1
        assert es(0.6)  # counter = 2 → stop

    def test_min_delta_with_max_mode(self):
        es = EarlyStopping(patience=2, min_delta=0.05, mode="max")
        es(0.8)  # best = 0.8
        es(0.83)  # 0.83 < 0.8 + 0.05 → counter = 1
        assert es(0.84)  # counter = 2 → stop


# ---------------------------------------------------------------------------
# Checkpoint serialisation
# ---------------------------------------------------------------------------


class TestStateDict:

    def test_round_trip(self):
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")
        es(1.0)
        es(1.1)
        es(1.2)

        state = es.state_dict()
        es2 = EarlyStopping()
        es2.load_state_dict(state)

        assert es2.patience == 5
        assert es2.min_delta == 0.01
        assert es2.mode == "min"
        assert es2.best == 1.0
        assert es2.counter == 2
        assert not es2.stopped

    def test_resumed_instance_continues_counting(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # counter = 1
        es(1.2)  # counter = 2

        es2 = EarlyStopping()
        es2.load_state_dict(es.state_dict())

        assert es2(1.3)  # counter = 3 → stop


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_patience_one(self):
        es = EarlyStopping(patience=1)
        es(1.0)
        assert es(1.0)  # no improvement → stop immediately

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="invalid")

    def test_stays_stopped(self):
        """Once stopped, calling again still returns True."""
        es = EarlyStopping(patience=1)
        es(1.0)
        es(1.0)  # stopped
        assert es(0.5)  # even an improvement keeps stopped=True

    def test_repr(self):
        es = EarlyStopping(patience=5)
        assert "EarlyStopping" in repr(es)
        assert "patience=5" in repr(es)

    def test_repr_fresh_instance_with_inf(self):
        es = EarlyStopping(patience=5)
        r = repr(es)
        assert "inf" in r

    def test_nan_metric_does_not_corrupt_state(self):
        es = EarlyStopping(patience=5)
        es(1.0)
        es(float("nan"))
        assert es.best == 1.0
        assert es.counter == 1

    def test_nan_corrupted_best_recovered_on_load(self):
        """Loading a checkpoint with best=NaN resets to mode-appropriate inf."""
        es = EarlyStopping(patience=3)
        state = es.state_dict()
        state["best"] = float("nan")

        es2 = EarlyStopping()
        es2.load_state_dict(state)
        assert es2.best == float("inf")

    def test_nan_corrupted_best_recovered_max_mode(self):
        """Max-mode NaN recovery resets to -inf."""
        es = EarlyStopping(patience=3, mode="max")
        state = es.state_dict()
        state["best"] = float("nan")

        es2 = EarlyStopping()
        es2.load_state_dict(state)
        assert es2.best == float("-inf")

    def test_consecutive_nans_exhaust_patience(self):
        """Multiple NaN metrics count toward patience."""
        es = EarlyStopping(patience=2)
        es(1.0)
        es(float("nan"))  # counter = 1
        assert es(float("nan"))  # counter = 2 → stop
