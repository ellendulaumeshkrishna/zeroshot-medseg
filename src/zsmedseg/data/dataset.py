"""A dataset is a folder of images and, optionally, a folder of binary masks.

Expected layout (names configurable in the dataset YAML)::

    <root>/
      images/   case_001.png  case_002.jpg ...
      masks/    case_001_mask.png  case_002_mask.png ...   (optional)

Masks are single-channel images; a pixel is foreground when its value is greater than
``mask_threshold`` (0 for {0,1} or {0,255} masks, 127 for JPEG-compressed masks).
Images without a mask folder can still be segmented (``predict``), just not evaluated.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import DatasetConfig


@dataclass(frozen=True)
class Sample:
    """One image and its (optional) ground-truth mask on disk."""

    name: str
    image_path: Path
    mask_path: Path | None


def read_image_bgr(path: Path) -> np.ndarray:
    """Read an image as an ``H x W x 3`` uint8 BGR array (OpenCV convention)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def read_binary_mask(path: Path, threshold: int = 0) -> np.ndarray:
    """Read a mask as an ``H x W`` boolean array (``value > threshold``)."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > threshold


class SegmentationFolderDataset:
    """Pairs every image in ``<root>/<images_dir>`` with its mask (if a mask folder exists)."""

    def __init__(self, cfg: DatasetConfig, limit: int | None = None) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.image_dir = self.root / cfg.images_dir
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")

        self.mask_dir: Path | None = None
        if cfg.masks_dir:
            candidate = self.root / cfg.masks_dir
            if candidate.is_dir():
                self.mask_dir = candidate

        self.samples = self._discover()
        if limit is not None:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise RuntimeError(f"No images with extensions {cfg.image_extensions} in {self.image_dir}")

    def _discover(self) -> list[Sample]:
        extensions = {e.lower() for e in self.cfg.image_extensions}
        samples = []
        for path in sorted(self.image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            mask_path = None
            if self.mask_dir is not None:
                mask_path = self.mask_dir / self.cfg.mask_pattern.format(stem=path.stem)
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"No mask for {path.name}: expected {mask_path}. Check 'mask_pattern' in the dataset config."
                    )
            samples.append(Sample(name=path.name, image_path=path, mask_path=mask_path))
        return samples

    @property
    def has_masks(self) -> bool:
        return self.mask_dir is not None

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_image_bgr(sample.image_path)

    def load_mask(self, sample: Sample) -> np.ndarray | None:
        if sample.mask_path is None:
            return None
        return read_binary_mask(sample.mask_path, self.cfg.mask_threshold)
