"""Convert public datasets (or any image/mask folders) into the layout the pipeline expects::

    <out>/images/<stem>.png
    <out>/masks/<stem>_mask.png      uint8, 0 = background, 1 = target

Supported sources (``python -m zsmedseg download-data <name>`` fetches the raw files):

* ``busi``         - Dataset_BUSI_with_GT: ``benign/``, ``malignant/``, ``normal/`` folders with
  ``<name>.png`` + ``<name>_mask.png`` (16 images also have ``<name>_mask_1.png``, ...; the
  reported results use the first mask only, see ``prepare_busi``).
* ``kvasir``       - Kvasir-SEG: ``images/*.jpg`` + ``masks/*.jpg``; JPEG masks binarised at > 127,
  images stored losslessly as PNG.
* ``isic2018``     - ISIC 2018 Task 1 official test set (two zip files). Masks binarised at > 127;
  images larger than 1024 px are shrunk so the longest side is 1024 (the full-size photos,
  up to 6621 x 4411, do not fit in GPU memory).
* ``refuge2``      - REFUGE2 test split. Mask values: 0 = cup, 128 = rim, 255 = background,
  so the optic disc (cup + rim) is ``value < 255``.
* ``camus``        - CAMUS echocardiography, test patients 451-500, 2- and 4-chamber views at
  end-diastole and end-systole. NIfTI -> PNG; target = label 1 (left-ventricle cavity).
* ``medclipsamv2`` - the split bundle of MedCLIP-SAMv2 (``data.zip``): breast ultrasound (UDIAT),
  lung X-ray, brain MRI and lung CT, already in ``test_images/`` + ``test_masks/`` form.
* ``generic``      - any ``images/`` + ``masks/`` folders whose files share a stem (or follow
  a pattern), binarised at a chosen threshold.
"""

from __future__ import annotations

import gzip
import logging
import re
import struct
import zipfile
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _write_pair(image: np.ndarray, mask: np.ndarray, out: Path, stem: str) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / f"{stem}.png"), image)
    cv2.imwrite(str(out / "masks" / f"{stem}_mask.png"), mask.astype(np.uint8))


def _read_gray(path: Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(path)
    return data[..., 0] if data.ndim == 3 else data


def prepare_busi(source: Path, out: Path, threshold: int = 127, merge_extra_masks: bool = False) -> int:
    """Dataset_BUSI_with_GT -> standard layout. Returns the number of images written.

    16 images have extra masks (``<name>_mask_1.png``, ...) for further lesions. The reported
    results use only ``<name>_mask.png``, which is the default; ``merge_extra_masks=True``
    takes the union of all of them instead.
    """
    count = 0
    for class_dir in ("benign", "malignant", "normal"):
        folder = source / class_dir
        if not folder.is_dir():
            logger.warning("Skipping missing folder %s", folder)
            continue
        for image_path in sorted(folder.glob("*.png")):
            if "_mask" in image_path.stem:
                continue
            pattern = "_mask*.png" if merge_extra_masks else "_mask.png"
            masks = sorted(folder.glob(f"{glob_escape(image_path.stem)}{pattern}"))
            if not masks:
                raise FileNotFoundError(f"No mask for {image_path}")
            merged = np.zeros(_read_gray(masks[0]).shape, dtype=bool)
            for m in masks:
                merged |= _read_gray(m) > threshold
            _write_pair(cv2.imread(str(image_path)), merged, out, image_path.stem)
            count += 1
    return count


def prepare_kvasir(source: Path, out: Path, threshold: int = 127) -> int:
    """Kvasir-SEG -> standard layout (PNG images, masks binarised at ``threshold``)."""
    count = 0
    for image_path in sorted((source / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = _find_by_stem(source / "masks", image_path.stem)
        _write_pair(cv2.imread(str(image_path)), _read_gray(mask_path) > threshold, out, image_path.stem)
        count += 1
    return count


def prepare_isic2018(source: Path, out: Path, threshold: int = 127, max_side: int = 1024) -> int:
    """ISIC 2018 Task 1 test zips -> standard layout, longest side capped at ``max_side``."""
    images = zipfile.ZipFile(_one(source, "ISIC2018_Task1-2_Test_Input.zip"))
    truths = zipfile.ZipFile(_one(source, "ISIC2018_Task1_Test_GroundTruth.zip"))
    suffix = "_segmentation.png"
    gt_names = {Path(n).name.replace(suffix, ""): n for n in truths.namelist() if n.endswith(suffix)}
    count = 0
    for name in sorted(images.namelist()):
        if not name.lower().endswith(".jpg"):
            continue
        stem = Path(name).stem
        image = cv2.imdecode(np.frombuffer(images.read(name), np.uint8), cv2.IMREAD_COLOR)
        gt = cv2.imdecode(np.frombuffer(truths.read(gt_names[stem]), np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = (gt > threshold).astype(np.uint8)
        height, width = image.shape[:2]
        scale = max_side / max(height, width)
        if scale < 1:
            size = (int(round(width * scale)), int(round(height * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        _write_pair(image, mask, out, stem)
        count += 1
    return count


def prepare_refuge2(source: Path, out: Path, split: str = "test") -> int:
    """REFUGE2 images (``*.jpg``) + masks (``*.bmp``) of one split, matched by file stem."""
    in_split = lambda p: split in p.relative_to(source).parts  # noqa: E731
    masks = {p.stem: p for p in source.rglob("*.bmp") if in_split(p)}
    count = 0
    for image_path in sorted(p for p in source.rglob("*.jpg") if in_split(p)):
        mask = _read_gray(masks[image_path.stem])
        _write_pair(cv2.imread(str(image_path), cv2.IMREAD_COLOR), mask < 255, out, image_path.stem)
        count += 1
    return count


CAMUS_TEST = re.compile(r"patient0*(\d+)_(2CH|4CH)_(ED|ES)_gt\.nii(\.gz)?$")


def prepare_camus(source: Path, out: Path, first_patient: int = 451, last_patient: int = 500) -> int:
    """CAMUS ``<patient>_<view>_<phase>.nii`` + ``..._gt.nii`` -> PNG pairs (target = label 1).

    Only the test patients (451-500) and the 2-/4-chamber end-diastole/end-systole frames.
    """
    count = 0
    for gt_path in sorted(source.rglob("*_gt.nii*")):
        match = CAMUS_TEST.search(gt_path.name)
        if not match or not first_patient <= int(match.group(1)) <= last_patient:
            continue
        image_path = gt_path.with_name(gt_path.name.replace("_gt.nii", ".nii"))
        image = np.squeeze(read_nifti(image_path)).T  # NIfTI (x, y) -> image rows/cols
        gt = np.squeeze(read_nifti(gt_path)).T
        image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        stem = image_path.name.split(".nii")[0]
        _write_pair(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), gt == 1, out, stem)
        count += 1
    return count


def prepare_medclipsamv2(source: Path, out: Path) -> int:
    """Unpack MedCLIP-SAMv2's ``data.zip`` into ``out/<dataset>/{test,val}_{images,masks}``.

    The training split is skipped: nothing is trained here, and it is two thirds of the bundle.
    """
    archive = zipfile.ZipFile(_one(source, "data.zip") if source.is_dir() else source)
    count = 0
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if member.is_dir() or ".." in parts or len(parts) < 3 or parts[-2].startswith("train_"):
            continue
        target = out.joinpath(*parts[1:])  # drop the leading "data/"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(member))
        count += "_images" in parts[-2]
    return count


def read_nifti(path: Path) -> np.ndarray:
    """Minimal reader for a single-file NIfTI-1 image, ``.nii`` or ``.nii.gz`` (enough for CAMUS)."""
    with (gzip.open(path, "rb") if path.name.endswith(".gz") else open(path, "rb")) as fh:
        header = fh.read(348)
        endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
        dims = struct.unpack(endian + "8h", header[40:56])
        dtype_code = struct.unpack(endian + "h", header[70:72])[0]
        offset = int(struct.unpack(endian + "f", header[108:112])[0])
        dtype = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}[dtype_code]
        fh.seek(offset)
        shape = dims[1 : 1 + dims[0]]
        data = np.frombuffer(fh.read(), dtype=endian + dtype, count=int(np.prod(shape)))
    return data.reshape(shape, order="F")


def prepare_generic(
    images: Path,
    masks: Path | None,
    out: Path,
    threshold: int = 0,
    mask_regex: str | None = None,
) -> int:
    """Any image folder (+ optional mask folder) -> standard layout.

    Masks are matched by identical stem, or by ``mask_regex`` with a ``{stem}`` placeholder
    (e.g. ``"{stem}_seg"``). Without masks only images are written (prediction-only data).
    """
    count = 0
    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path))
        if masks is None:
            (out / "images").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "images" / f"{image_path.stem}.png"), image)
        else:
            stem = mask_regex.format(stem=re.escape(image_path.stem)) if mask_regex else re.escape(image_path.stem)
            mask_path = _find_by_regex(masks, stem)
            _write_pair(image, _read_gray(mask_path) > threshold, out, image_path.stem)
        count += 1
    return count


def _one(folder: Path, filename: str) -> Path:
    """Find ``filename`` directly in ``folder`` or anywhere below it."""
    matches = [folder / filename] if (folder / filename).is_file() else sorted(folder.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {folder}")
    return matches[0]


def glob_escape(text: str) -> str:
    """Escape glob metacharacters (BUSI names contain parentheses and brackets are possible)."""
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def _find_by_stem(folder: Path, stem: str) -> Path:
    return _find_by_regex(folder, re.escape(stem))


def _find_by_regex(folder: Path, pattern: str) -> Path:
    regex = re.compile(rf"^{pattern}$")
    matches: Iterable[Path] = [
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and regex.match(p.stem)
    ]
    matches = sorted(matches)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mask matching '{pattern}' in {folder}, found {len(matches)}")
    return matches[0]
