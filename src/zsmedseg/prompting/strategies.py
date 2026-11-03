"""Turn the saliency map into point prompts for MedSAM2."""

from __future__ import annotations

import numpy as np

Point = tuple[int, int]  # (x, y) in original-image pixels


class MultiPeakPrompts:
    """The ``num_points`` highest, spatially separated saliency peaks.

    Repeatedly take the global maximum, then blank a square of half-side
    ``suppression_frac * min(H, W)`` around it so the next peak lies elsewhere. Several
    peaks act as backups when the hottest one sits on the wrong structure.
    """

    def __init__(self, num_points: int = 5, suppression_frac: float = 0.10) -> None:
        if num_points < 1:
            raise ValueError("num_points must be >= 1")
        self.num_points = num_points
        self.suppression_frac = suppression_frac

    def __call__(self, saliency: np.ndarray) -> list[Point]:
        work = saliency.astype(np.float32, copy=True)
        height, width = work.shape
        radius = max(3, int(self.suppression_frac * min(height, width)))
        points: list[Point] = []
        for _ in range(self.num_points):
            y, x = np.unravel_index(int(work.argmax()), work.shape)
            if work[y, x] <= 0:
                break
            points.append((int(x), int(y)))
            work[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = -1.0
        return points
