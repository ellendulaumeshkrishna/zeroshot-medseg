"""Text-guided saliency: a per-pixel probability that the pixel belongs to the target.

Pipeline for one image::

    BGR image --rescale (short side 336)--> CLIP-normalised image
             --sliding 336x336 windows--> per window: DINO features + CorrCLIP features
             --cosine with class embeddings--> per-window class logits
             --stitch (average overlaps), resize to original size--> logits K x H x W
             --softmax(logit_scale * logits)--> probabilities; channel 1 = target

The map localises the target well but is blurry and too large, so it is only used to
place point prompts for MedSAM2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..config import SaliencyConfig
from ..data.preprocessing import prepare_clip_input
from ..models.corrclip import PAD_REGION_ID, CorrCLIPVisualEncoder
from ..models.dino import DinoFeatureExtractor

TARGET_CLASS = 1  # class 0 = background, class 1 = target


@dataclass
class SaliencyResult:
    probabilities: torch.Tensor  # K x H x W, softmax over classes, original resolution
    target: np.ndarray  # H x W float32, probability of the target class


def centred_padding(height: int, width: int, multiple: int) -> tuple[int, int, int, int]:
    """(left, right, top, bottom) padding that makes both sides divisible by ``multiple``."""
    left = right = top = bottom = 0
    if width % multiple:
        extra = multiple - width % multiple
        left, right = extra // 2, extra - extra // 2
    if height % multiple:
        extra = multiple - height % multiple
        top, bottom = extra // 2, extra - extra // 2
    return left, right, top, bottom


def sliding_windows(height: int, width: int, size: int, stride: int):
    """Yield ``(y1, x1, y2, x2)`` windows covering the image; the last ones are shifted inward."""
    rows = max(height - size + stride - 1, 0) // stride + 1
    cols = max(width - size + stride - 1, 0) // stride + 1
    for r in range(rows):
        for c in range(cols):
            y2, x2 = min(r * stride + size, height), min(c * stride + size, width)
            yield max(y2 - size, 0), max(x2 - size, 0), y2, x2


class SaliencyEstimator:
    """Computes the target probability map of an image."""

    def __init__(
        self,
        encoder: CorrCLIPVisualEncoder,
        dino: DinoFeatureExtractor,
        class_embeddings: torch.Tensor,
        cfg: SaliencyConfig,
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.dino = dino
        self.class_embeddings = class_embeddings.to(device)
        self.cfg = cfg
        self.device = device

    @torch.inference_mode()
    def _window_logits(self, crop: torch.Tensor, crop_regions: torch.Tensor) -> torch.Tensor:
        """Class logits (``1 x K x h x w``) for one CLIP-normalised window."""
        dino_feats, feat_shape = self.dino(crop)
        feats = self.encoder(crop.float(), dino_feats, feat_shape, crop_regions)
        feats = F.normalize(feats, dim=-1)
        logits = feats @ self.class_embeddings.T  # 1 x (h*w) x K
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], *feat_shape)
        return F.interpolate(logits, size=crop.shape[-2:], mode="bilinear")

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray, regions: np.ndarray | None = None) -> SaliencyResult:
        cfg = self.cfg
        out_h, out_w = image_bgr.shape[:2]
        image = prepare_clip_input(image_bgr, cfg.short_side, cfg.max_long_side).to(self.device)
        _, _, height, width = image.shape

        # Region ids at the rescaled resolution (zeros = a single region).
        if regions is None:
            region_map = torch.zeros((1, 1, height, width), dtype=torch.int32, device=self.device)
        else:
            region_map = torch.from_numpy(regions).to(self.device)[None, None].float()
            region_map = F.interpolate(region_map, size=(height, width), mode="nearest").int()

        num_classes = self.class_embeddings.shape[0]
        logit_sum = image.new_zeros((1, num_classes, height, width))
        counts = image.new_zeros((1, 1, height, width))
        for y1, x1, y2, x2 in sliding_windows(height, width, cfg.window_size, cfg.window_stride):
            crop = image[:, :, y1:y2, x1:x2]
            crop_regions = region_map[:, :, y1:y2, x1:x2]
            h, w = crop.shape[2:]
            pad = centred_padding(h, w, cfg.pad_multiple)
            if any(pad):
                crop = F.pad(crop, pad)
                crop_regions = F.pad(crop_regions, pad, value=PAD_REGION_ID)
            logits = self._window_logits(crop, crop_regions)
            if any(pad):
                logits = logits[:, :, pad[2] : pad[2] + h, pad[0] : pad[0] + w]
            logit_sum += F.pad(logits, (x1, width - x2, y1, height - y2))
            counts[:, :, y1:y2, x1:x2] += 1

        logits = F.interpolate(logit_sum / counts, size=(out_h, out_w), mode="bilinear")[0]
        probabilities = (logits * cfg.logit_scale).softmax(dim=0)
        target = probabilities[TARGET_CLASS].float().cpu().numpy()
        return SaliencyResult(probabilities=probabilities, target=target)
