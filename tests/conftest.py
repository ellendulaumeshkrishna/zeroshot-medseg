"""Shared fixtures. All tests run on CPU without model weights."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def toy_dataset(tmp_path: Path) -> Path:
    """Three 40x50 images; two with a square lesion mask, one empty."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(3):
        image = np.full((40, 50, 3), 30 * i, dtype=np.uint8)
        mask = np.zeros((40, 50), dtype=np.uint8)
        if i < 2:
            mask[10:20, 15:30] = 1
        cv2.imwrite(str(tmp_path / "images" / f"case_{i}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"case_{i}_mask.png"), mask)
    return tmp_path
