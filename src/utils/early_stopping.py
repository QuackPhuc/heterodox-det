"""Early stopping utility for training loops.

Monitors a tracked metric and signals when training should stop
if no improvement is observed for a given number of checks.
"""

import math


class EarlyStopping:
    """Tracks a metric and signals when to stop training.

    Args:
        patience: Number of checks without improvement before stopping.
        min_delta: Minimum absolute change to qualify as an improvement.
        mode: ``"min"`` treats lower as better (e.g. loss),
              ``"max"`` treats higher as better (e.g. mAP).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode

        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.stopped: bool = False

    def _is_improvement(self, current: float) -> bool:
        """Return True if *current* is better than *best* by at least min_delta."""
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def __call__(self, metric: float) -> bool:
        """Update state with the latest metric value.

        Returns:
            ``True`` if training should stop (patience exhausted).
        """
        # NaN metrics are treated as non-improvements to prevent state corruption
        if math.isnan(metric):
            self.counter += 1
        elif self._is_improvement(metric):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True

        return self.stopped

    # ------------------------------------------------------------------
    # Checkpoint serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best": self.best,
            "counter": self.counter,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: dict) -> None:
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]
        self.best = state["best"]
        self.counter = state["counter"]
        self.stopped = state["stopped"]
        # Recover from NaN-corrupted checkpoint state
        if math.isnan(self.best):
            self.best = float("inf") if self.mode == "min" else float("-inf")

    def __repr__(self) -> str:
        best_str = "inf" if math.isinf(self.best) else f"{self.best:.6g}"
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"mode='{self.mode}', counter={self.counter}/{self.patience}, "
            f"best={best_str})"
        )
