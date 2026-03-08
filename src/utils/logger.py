"""Experiment tracking abstraction for TensorBoard and WandB.

Provides a unified logging interface that automatically detects
available backends. If neither is installed, all calls are no-ops
so training works identically without tracking dependencies.

Usage:
    logger = ExperimentLogger(backend="wandb", project="my-project", config=cfg)
    logger.log_scalars({"train/loss": 0.5, "train/lr": 1e-3}, step=epoch)
    logger.finish()
"""

import warnings

# Probe for available backends at import time
_HAS_WANDB = False

try:
    import wandb as _wandb

    _HAS_WANDB = True
except ImportError:
    _wandb = None


class ExperimentLogger:
    """Unified experiment logger supporting WandB and TensorBoard.

    Args:
        backend: 'wandb', 'tensorboard', 'both', or 'none'
        project: WandB project name (ignored for tensorboard-only)
        name: run name (auto-generated if None)
        config: config dict to log (hyperparameters, architecture, etc.)
        save_dir: directory for TensorBoard event files
        tags: optional list of tags for WandB runs
    """

    def __init__(
        self,
        backend: str = "none",
        project: str = "ionized-meteorite",
        name: str = None,
        config: dict = None,
        save_dir: str = None,
        tags: list = None,
    ):
        self._use_wandb = False
        self._use_tb = False
        self._wandb_run = None
        self._tb_writer = None

        if backend in ("wandb", "both"):
            if _HAS_WANDB:
                self._wandb_run = _wandb.init(
                    project=project,
                    name=name,
                    config=config or {},
                    tags=tags,
                    reinit=True,
                )
                self._use_wandb = True
            else:
                warnings.warn(
                    "WandB requested but not installed. "
                    "Install with: pip install wandb",
                    stacklevel=2,
                )

        if backend in ("tensorboard", "both"):
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = save_dir or "runs/tb_logs"
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                self._use_tb = True
            except (ImportError, Exception) as e:
                warnings.warn(
                    f"TensorBoard requested but unavailable: {e}. "
                    "Install with: pip install tensorboard",
                    stacklevel=2,
                )

    @property
    def active(self) -> bool:
        """Whether any logging backend is active."""
        return self._use_wandb or self._use_tb

    def log_scalars(self, metrics: dict, step: int):
        """Log a dictionary of scalar metrics at the given step.

        Args:
            metrics: dict of metric_name → value (e.g. {"train/loss": 0.5})
            step: global step (typically epoch number)
        """
        if self._use_wandb:
            _wandb.log(metrics, step=step)

        if self._use_tb:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, global_step=step)

    def log_config(self, config: dict):
        """Log or update run configuration.

        Args:
            config: additional config dict to merge
        """
        if self._use_wandb and self._wandb_run is not None:
            self._wandb_run.config.update(config, allow_val_change=True)

    def finish(self):
        """Finalize logging and flush all buffers."""
        if self._use_wandb and self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

        if self._use_tb and self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass
