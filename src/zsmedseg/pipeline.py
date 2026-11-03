"""The full zero-shot segmentation pipeline.

In one sentence: **CLIP points, MedSAM2 draws.**

    1. regions    = MedSAM2 automatic masks(image)       # object-like regions
    2. saliency   = CorrCLIP + BMC-CLIP(image, regions)   # where is the target? (blurry)
    3. points     = 5 highest, well-separated peaks of saliency
    4. candidates = MedSAM2(image, each point)            # 3 masks per point = 15
    5. mask       = candidate with the highest predicted IoU

Nothing is trained. The only task-specific input is the text describing the target and
its background (``dataset.target_prompts`` / ``dataset.background_prompts``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .config import Config
from .models import (
    CorrCLIPVisualEncoder,
    DinoFeatureExtractor,
    MaskCandidate,
    PointPromptSegmenter,
    build_medsam2,
    encode_classes,
    load_bmc_clip,
)
from .prompting import MultiPeakPrompts, Point
from .saliency import RegionGenerator, SaliencyEstimator
from .selection import MaskSelector
from .utils.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Everything the pipeline produced for one image (useful for debugging and figures)."""

    mask: np.ndarray  # H x W bool, the final prediction
    saliency: np.ndarray  # H x W float32, target probability
    points: list[Point] = field(default_factory=list)
    candidates: list[MaskCandidate] = field(default_factory=list)
    kept: list[MaskCandidate] = field(default_factory=list)  # the candidate(s) forming the output
    chosen: MaskCandidate | None = None  # kept candidates merged into one
    regions: np.ndarray | None = None  # H x W int region ids

    @property
    def score(self) -> float:
        """Predicted IoU of the chosen mask (0 when nothing was chosen)."""
        return self.chosen.score if self.chosen is not None else 0.0


class ZeroShotSegmenter:
    """Text-prompted, training-free segmentation of one target class."""

    def __init__(
        self,
        region_generator: RegionGenerator,
        saliency_estimator: SaliencyEstimator,
        prompter: MultiPeakPrompts,
        point_segmenter: PointPromptSegmenter,
        selector: MaskSelector,
        encode_text: Callable[[list[str], list[str]], torch.Tensor] | None = None,
    ) -> None:
        self.region_generator = region_generator
        self.saliency_estimator = saliency_estimator
        self.prompter = prompter
        self.point_segmenter = point_segmenter
        self.selector = selector
        self.encode_text = encode_text  # (background, target) phrases -> class embeddings

    @classmethod
    def from_config(cls, cfg: Config) -> ZeroShotSegmenter:
        """Load the frozen models once and build every stage described by ``cfg``."""
        device = resolve_device(cfg.runtime.device)
        logger.info("Building pipeline on %s", device)

        # BMC-CLIP: one embedding per class from the text prompts, and the image encoder.
        clip_model, tokenizer = load_bmc_clip(cfg.weights.bmc_clip, cfg.weights.clip_arch, device)

        def encode_text(background: list[str], target: list[str]) -> torch.Tensor:
            return encode_classes(
                clip_model, tokenizer, [background, target], template_set=cfg.saliency.template_set, device=device
            )

        class_embeddings = encode_text(cfg.dataset.background_prompts, cfg.dataset.target_prompts)
        # CorrCLIP replaces CLIP's last attention using DINO similarities and the regions.
        encoder = CorrCLIPVisualEncoder(
            clip_model.visual,
            tau=cfg.corrclip.tau,
            alpha_spatial=cfg.corrclip.alpha_spatial,
            beta_semantic=cfg.corrclip.beta_semantic,
            merge_eps=cfg.corrclip.merge_eps,
            num_spatial_layers=cfg.corrclip.num_spatial_layers,
        )
        dino = DinoFeatureExtractor(cfg.weights.dino_repo, cfg.weights.dino_model).to(device)

        # MedSAM2 is used twice: automatic regions (step 1) and point prompts (step 4).
        sam_model = build_medsam2(cfg.weights.medsam2_config, cfg.weights.medsam2, device)
        regions = cfg.saliency.region_masks

        return cls(
            region_generator=RegionGenerator(
                sam_model,
                points_per_side=regions.points_per_side,
                pred_iou_thresh=regions.pred_iou_thresh,
                stability_score_thresh=regions.stability_score_thresh,
                box_nms_thresh=regions.box_nms_thresh,
            ),
            saliency_estimator=SaliencyEstimator(encoder, dino, class_embeddings, cfg.saliency, device),
            prompter=MultiPeakPrompts(cfg.prompting.num_points, cfg.prompting.suppression_frac),
            point_segmenter=PointPromptSegmenter(sam_model),
            selector=MaskSelector(cfg.selection.num_objects, cfg.selection.max_overlap),
            encode_text=encode_text,
        )

    def set_prompts(self, background: list[str], target: list[str]) -> None:
        """Change what to segment: re-embed the class phrases (the models stay loaded)."""
        if self.encode_text is None:
            raise RuntimeError("This segmenter was built without a text encoder")
        embeddings = self.encode_text(background, target)
        self.saliency_estimator.class_embeddings = embeddings.to(self.saliency_estimator.device)

    @torch.inference_mode()
    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Segment the target in one ``H x W x 3`` uint8 BGR image."""
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 1-2. Where is the target? (well localised but blurry probability map)
        regions = self.region_generator(image_rgb)
        saliency = self.saliency_estimator(image_bgr, regions).target

        # 3. Where to click?
        points = self.prompter(saliency)

        # 4. MedSAM2 draws three candidate outlines at each point.
        candidates: list[MaskCandidate] = []
        if points:
            self.point_segmenter.set_image(image_rgb)
            for index, (x, y) in enumerate(points):
                candidates.extend(self.point_segmenter.segment_point(x, y, prompt_index=index))

        # 5. Keep the outline MedSAM2 is most confident about.
        kept = self.selector.choose(candidates)
        chosen = self.selector.merge(kept)
        mask = np.zeros((height, width), dtype=bool)
        if chosen is not None:
            mask = chosen.mask
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0

        return SegmentationResult(
            mask=mask,
            saliency=saliency,
            points=points,
            candidates=candidates,
            kept=kept,
            chosen=chosen,
            regions=regions,
        )
