"""MedSAM2: the model that draws precise outlines ("exactly what shape is it?").

Two uses in this project, sharing one set of weights:

* :class:`PointPromptSegmenter` - given a point, return 3 candidate masks with MedSAM2's
  own quality estimate (predicted IoU) for each.
* :class:`AutomaticRegionGenerator` - prompt a regular grid of points over the whole image
  and paint the resulting masks into a region map (used by CorrCLIP's scope reconstruction).

Requires the ``sam2`` package from the MedSAM2 repository
(https://github.com/bowang-lab/MedSAM2).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Where to look for a plain `git clone https://github.com/bowang-lab/MedSAM2` when the
# `sam2` package is not installed: $ZSMEDSEG_MEDSAM2_DIR, then ./MedSAM2.
MEDSAM2_DIR_ENV = "ZSMEDSEG_MEDSAM2_DIR"


def ensure_sam2_importable() -> None:
    """Make MedSAM2's ``sam2`` package importable, or explain how to get it."""
    if importlib.util.find_spec("sam2") is not None:
        return
    for candidate in (os.environ.get(MEDSAM2_DIR_ENV), "MedSAM2"):
        if candidate and (Path(candidate) / "sam2" / "__init__.py").is_file():
            sys.path.insert(0, str(Path(candidate).resolve()))
            logger.info("Using MedSAM2 code from %s", Path(candidate).resolve())
            return
    raise ImportError(
        "The `sam2` package from MedSAM2 is required. Either\n"
        "  git clone https://github.com/bowang-lab/MedSAM2   (inside this repository), or\n"
        f"  set {MEDSAM2_DIR_ENV}=/path/to/MedSAM2, or\n"
        "  SAM2_BUILD_CUDA=0 pip install --no-deps -e /path/to/MedSAM2"
    )


def build_medsam2(config_path: str | Path, checkpoint_path: str | Path, device: torch.device | str) -> torch.nn.Module:
    """Instantiate MedSAM2 from a Hydra-style model YAML and load its checkpoint.

    The YAML is read directly (not via Hydra's search path), so it can live anywhere.
    No post-processing overrides are applied (``apply_postprocessing=False`` upstream).
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ensure_sam2_importable()
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    for path, what in ((config_path, "MedSAM2 config"), (checkpoint_path, "MedSAM2 checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{what} not found at {path}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"MedSAM2 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    logger.info("Loaded MedSAM2 from %s", checkpoint_path)
    return model.to(device).eval()


@dataclass
class MaskCandidate:
    """One mask proposed by MedSAM2 for one point prompt."""

    mask: np.ndarray  # H x W bool, original image resolution
    score: float  # MedSAM2's predicted IoU of this mask
    low_res_logits: np.ndarray  # h x w float logits before thresholding (MedSAM2's raw output)
    prompt_index: int  # which point produced it
    mask_index: int  # which of the 3 multimask outputs it is


class PointPromptSegmenter:
    """Point-prompted MedSAM2 on a fixed image."""

    def __init__(self, model: torch.nn.Module) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Encode the full-resolution RGB image once; prompts reuse the embedding."""
        self.predictor.set_image(image_rgb)

    @torch.inference_mode()
    def segment_point(self, x: int, y: int, prompt_index: int = 0) -> list[MaskCandidate]:
        """Return MedSAM2's 3 candidate masks for a single foreground point ``(x, y)``."""
        masks, scores, low_res = self.predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return [
            MaskCandidate(
                mask=masks[i] > 0,
                score=float(scores[i]),
                low_res_logits=low_res[i],
                prompt_index=prompt_index,
                mask_index=i,
            )
            for i in range(len(scores))
        ]


class AutomaticRegionGenerator:
    """MedSAM2's automatic mask generator, flattened into one integer region map."""

    def __init__(
        self,
        model: torch.nn.Module,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.5,
        box_nms_thresh: float = 0.5,
    ) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self.generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            multimask_output=True,
        )

    @torch.inference_mode()
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return an ``H x W`` int map: 0 = not covered by any mask, 1..N = regions.

        Masks are painted from smallest to largest area, so where masks overlap the larger
        one wins (this matches the reference implementation).
        """
        regions = np.zeros(image_rgb.shape[:2], dtype=np.int64)
        annotations = sorted(self.generator.generate(image_rgb), key=lambda a: a["area"])
        for region_id, ann in enumerate(annotations, start=1):
            regions[ann["segmentation"]] = region_id
        return regions
