"""Version-safe checkpoint loading utilities.

PyTorch >= 2.6 defaults to `weights_only=True` which restricts
deserialization to tensor-safe types. Older checkpoints or those
containing non-tensor metadata (config dicts, string fields) may
fail to load. This module provides a safe wrapper with automatic
fallback and clear warnings.
"""

import warnings
import torch


def safe_load_checkpoint(
    path: str,
    map_location=None,
) -> dict:
    """Load a checkpoint with version-safe `weights_only` handling.

    Strategy:
      1. Try `weights_only=True` (secure, PyTorch >= 2.0).
      2. On failure, retry with `weights_only=False` and warn.
      3. For PyTorch < 2.0 (no `weights_only` kwarg), load directly.

    Args:
        path: path to checkpoint `.pt` file
        map_location: device mapping (e.g., 'cpu', 'cuda:0')

    Returns:
        Loaded checkpoint dict.
    """
    # PyTorch >= 2.0 supports weights_only
    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            warnings.warn(
                f"Could not load '{path}' with weights_only=True. "
                f"Falling back to weights_only=False. "
                f"Only load checkpoints from trusted sources.",
                stacklevel=2,
            )
            return torch.load(path, map_location=map_location, weights_only=False)

    # Fallback for older PyTorch versions
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # weights_only kwarg not supported
        return torch.load(path, map_location=map_location)
