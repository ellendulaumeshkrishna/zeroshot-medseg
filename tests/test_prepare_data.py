import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np

from zsmedseg.tools.prepare_data import (
    prepare_busi,
    prepare_camus,
    prepare_generic,
    prepare_isic2018,
    prepare_medclipsamv2,
    prepare_refuge2,
    read_nifti,
)


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)


def _read_mask(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _busi_with_two_masks(src: Path) -> None:
    _write(src / "benign" / "benign (1).png", np.zeros((8, 8, 3), np.uint8))
    m1 = np.zeros((8, 8), np.uint8)
    m1[0:2, 0:2] = 255
    m2 = np.zeros((8, 8), np.uint8)
    m2[5:7, 5:7] = 255
    _write(src / "benign" / "benign (1)_mask.png", m1)
    _write(src / "benign" / "benign (1)_mask_1.png", m2)
    # a similarly named image must not steal the masks above
    _write(src / "benign" / "benign (10).png", np.zeros((8, 8, 3), np.uint8))
    _write(src / "benign" / "benign (10)_mask.png", np.zeros((8, 8), np.uint8))


def test_busi_uses_the_first_mask_by_default(tmp_path: Path):
    _busi_with_two_masks(tmp_path / "raw")
    assert prepare_busi(tmp_path / "raw", tmp_path / "out") == 2
    mask = _read_mask(tmp_path / "out" / "masks" / "benign (1)_mask.png")
    assert set(np.unique(mask)) == {0, 1} and mask.sum() == 4


def test_busi_can_merge_all_masks(tmp_path: Path):
    _busi_with_two_masks(tmp_path / "raw")
    prepare_busi(tmp_path / "raw", tmp_path / "out", merge_extra_masks=True)
    assert _read_mask(tmp_path / "out" / "masks" / "benign (1)_mask.png").sum() == 8


def test_isic_reads_the_zips_and_caps_the_size(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = np.full((1500, 2000, 3), 128, np.uint8)
    gt = np.zeros((1500, 2000), np.uint8)
    gt[500:1000, 500:1500] = 255
    with zipfile.ZipFile(raw / "ISIC2018_Task1-2_Test_Input.zip", "w") as zf:
        zf.writestr("ISIC2018_Task1-2_Test_Input/ISIC_0000001.jpg", cv2.imencode(".jpg", image)[1].tobytes())
        zf.writestr("ISIC2018_Task1-2_Test_Input/LICENSE.txt", "not an image")
    with zipfile.ZipFile(raw / "ISIC2018_Task1_Test_GroundTruth.zip", "w") as zf:
        name = "ISIC2018_Task1_Test_GroundTruth/ISIC_0000001_segmentation.png"
        zf.writestr(name, cv2.imencode(".png", gt)[1].tobytes())

    assert prepare_isic2018(raw, tmp_path / "out") == 1
    out_image = cv2.imread(str(tmp_path / "out" / "images" / "ISIC_0000001.png"))
    out_mask = _read_mask(tmp_path / "out" / "masks" / "ISIC_0000001_mask.png")
    assert out_image.shape[:2] == out_mask.shape == (768, 1024)
    assert set(np.unique(out_mask)) == {0, 1}


def test_refuge2_keeps_the_test_split_and_cup_plus_rim(tmp_path: Path):
    raw = tmp_path / "raw" / "REFUGE2"
    mask = np.full((6, 6), 255, np.uint8)
    mask[1:5, 1:5] = 128  # rim
    mask[2:4, 2:4] = 0  # cup
    for split in ("test", "train"):
        _write(raw / split / "images" / f"{split}1.jpg", np.zeros((6, 6, 3), np.uint8))
        _write(raw / split / "mask" / f"{split}1.bmp", mask)
    assert prepare_refuge2(tmp_path / "raw", tmp_path / "out") == 1
    assert _read_mask(tmp_path / "out" / "masks" / "test1_mask.png").sum() == 16  # the whole disc


def _write_nifti(path: Path, data: np.ndarray) -> None:
    """Minimal uncompressed NIfTI-1 file (only the fields read_nifti uses)."""
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    dims = [data.ndim, *data.shape] + [1] * (7 - data.ndim)
    struct.pack_into("<8h", header, 40, *dims)
    struct.pack_into("<h", header, 70, 2)  # uint8
    struct.pack_into("<f", header, 108, 352.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + b"\0" * 4 + data.astype(np.uint8).tobytes(order="F"))


def test_camus_keeps_test_patients_and_label_1(tmp_path: Path):
    raw = tmp_path / "raw"
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    labels = np.zeros((4, 5), np.uint8)
    labels[1:3, 1:4] = 1  # left-ventricle cavity
    labels[0, :] = 2  # myocardium: not the target
    for patient in ("patient0451", "patient0100"):  # 100 is a training patient
        _write_nifti(raw / patient / f"{patient}_2CH_ED.nii", image)
        _write_nifti(raw / patient / f"{patient}_2CH_ED_gt.nii", labels)
    assert np.array_equal(read_nifti(raw / "patient0451" / "patient0451_2CH_ED.nii"), image)

    assert prepare_camus(raw, tmp_path / "out") == 1
    mask = _read_mask(tmp_path / "out" / "masks" / "patient0451_2CH_ED_mask.png")
    assert mask.shape == (5, 4) and mask.sum() == 6  # transposed to image rows/cols


def test_medclipsamv2_bundle_is_unpacked_without_the_data_prefix(tmp_path: Path):
    bundle = tmp_path / "data.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("data/lung_Xray/test_images/a.png", b"x")
        zf.writestr("data/lung_Xray/test_masks/a.png", b"y")
    assert prepare_medclipsamv2(bundle, tmp_path / "out") == 1
    assert (tmp_path / "out" / "lung_Xray" / "test_masks" / "a.png").is_file()


def test_generic_with_custom_mask_names(tmp_path: Path):
    _write(tmp_path / "img" / "a.jpg", np.zeros((6, 6, 3), np.uint8))
    m = np.zeros((6, 6), np.uint8)
    m[:3] = 200
    _write(tmp_path / "gt" / "a_seg.png", m)
    out = tmp_path / "out"
    assert prepare_generic(tmp_path / "img", tmp_path / "gt", out, threshold=127, mask_regex="{stem}_seg") == 1
    assert (out / "images" / "a.png").is_file()
    assert _read_mask(out / "masks" / "a_mask.png").sum() == 18


def test_generic_images_only(tmp_path: Path):
    _write(tmp_path / "img" / "x.png", np.zeros((4, 4, 3), np.uint8))
    assert prepare_generic(tmp_path / "img", None, tmp_path / "out") == 1
    assert not (tmp_path / "out" / "masks").exists()
