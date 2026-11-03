from pathlib import Path

import numpy as np
import pytest
import torch

from zsmedseg.config import DatasetConfig
from zsmedseg.data import SegmentationFolderDataset, clip_to_imagenet, prepare_clip_input, rescaled_size
from zsmedseg.data.preprocessing import CLIP_MEAN, IMAGENET_MEAN


def test_rescaled_size_matches_mmcv_rule():
    # BUSI 'malignant (88).png' is 775 x 591 (W x H) -> short side 336.
    assert rescaled_size(591, 775, 2048, 336) == (336, 441)
    # portrait images keep their orientation
    assert rescaled_size(800, 400, 2048, 336) == (672, 336)


def test_prepare_clip_input_shape_and_normalisation():
    image = np.full((100, 200, 3), 128, dtype=np.uint8)
    tensor = prepare_clip_input(image, short_side=50)
    assert tensor.shape == (1, 3, 50, 100)
    assert abs(float(tensor.mean())) < 1.0


def test_clip_to_imagenet_maps_mean_colour():
    clip_mean_image = torch.zeros(1, 3, 4, 4)  # CLIP-normalised value 0 == CLIP mean colour
    out = clip_to_imagenet(clip_mean_image)
    expected = (torch.tensor(CLIP_MEAN) - torch.tensor(IMAGENET_MEAN)) / torch.tensor((0.229, 0.224, 0.225))
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-6)


def test_dataset_pairs_images_and_masks(toy_dataset: Path):
    ds = SegmentationFolderDataset(DatasetConfig(root=str(toy_dataset)))
    assert len(ds) == 3 and ds.has_masks
    sample = ds[0]
    assert sample.mask_path.name == "case_0_mask.png"
    mask = ds.load_mask(sample)
    assert mask.dtype == bool and mask.sum() == 10 * 15


def test_dataset_without_masks(toy_dataset: Path):
    ds = SegmentationFolderDataset(DatasetConfig(root=str(toy_dataset), masks_dir=None))
    assert not ds.has_masks and ds.load_mask(ds[0]) is None


def test_missing_mask_is_reported(toy_dataset: Path):
    (toy_dataset / "masks" / "case_1_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="case_1"):
        SegmentationFolderDataset(DatasetConfig(root=str(toy_dataset)))


def test_limit(toy_dataset: Path):
    assert len(SegmentationFolderDataset(DatasetConfig(root=str(toy_dataset)), limit=2)) == 2
