"""Image preprocessing for the CLIP branch.

These functions reproduce, bit for bit, what the original mmsegmentation pipeline did
(``Resize(scale=(2048, 336), keep_ratio=True)`` followed by ``SegDataPreProcessor`` with
``bgr_to_rgb=True``), so results match the numbers reported for this project.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# OpenAI-CLIP normalisation constants on the 0-255 scale (RGB order).
CLIP_MEAN_255 = np.array([122.771, 116.746, 104.094], dtype=np.float32)
CLIP_STD_255 = np.array([68.501, 66.632, 70.323], dtype=np.float32)

# The same constants on the 0-1 scale, and ImageNet's (used by DINO).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rescaled_size(height: int, width: int, max_long: int, max_short: int) -> tuple[int, int]:
    """Output (height, width) when fitting an image inside ``max_long x max_short``.

    The aspect ratio is kept and the scale is the largest one for which the long side is
    at most ``max_long`` and the short side at most ``max_short`` (mmcv ``rescale_size``).
    """
    scale = min(max_long / max(height, width), max_short / min(height, width))
    return int(height * scale + 0.5), int(width * scale + 0.5)


def prepare_clip_input(
    image_bgr: np.ndarray,
    short_side: int = 336,
    max_long_side: int = 2048,
) -> torch.Tensor:
    """Resize and normalise a BGR uint8 image into a ``1 x 3 x H x W`` CLIP tensor."""
    h, w = image_bgr.shape[:2]
    new_h, new_w = rescaled_size(h, w, max_long_side, short_side)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = resized[..., ::-1].astype(np.float32)
    normalised = (rgb - CLIP_MEAN_255) / CLIP_STD_255
    return torch.from_numpy(np.ascontiguousarray(normalised.transpose(2, 0, 1)))[None]


def clip_to_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Re-normalise CLIP-normalised images (``B x 3 x H x W``) with ImageNet statistics."""
    clip_mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    clip_std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    in_mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    in_std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images * clip_std + clip_mean - in_mean) / in_std
