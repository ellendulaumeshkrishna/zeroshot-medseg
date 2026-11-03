"""Region masks for CorrCLIP's scope reconstruction.

MedSAM2's automatic mask generator splits the image into object-like regions. CorrCLIP
then lets a patch average features only with patches of its own region, so a lesion's
features are not mixed with the tissue around it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.medsam2 import AutomaticRegionGenerator


class RegionGenerator:
    """Returns an ``H x W`` integer map of region ids (0 = not covered by any region)."""

    def __init__(
        self,
        sam_model: Any,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.5,
        box_nms_thresh: float = 0.5,
    ) -> None:
        self.generator = AutomaticRegionGenerator(
            sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
        )

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray | None:
        return self.generator(image_rgb)
