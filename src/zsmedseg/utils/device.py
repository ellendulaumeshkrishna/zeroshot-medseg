"""Device selection and reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA when available, else CPU; anything else is passed to torch."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch (the pipeline itself has no random steps)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
