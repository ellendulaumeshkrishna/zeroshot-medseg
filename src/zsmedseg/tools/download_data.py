"""Download the raw files of every dataset used in this project.

    python -m zsmedseg download-data <name>          # -> raw/<name>/
    python -m zsmedseg prepare-data  <name>          # raw/<name>/ -> data/<name>/

| name           | source                                                          | size   |
|----------------|-----------------------------------------------------------------|--------|
| ``busi``         | Kaggle ``aryashah2k/breast-ultrasound-images-dataset``             | 0.27 GB|
| ``kvasir``       | https://datasets.simula.no/downloads/kvasir-seg.zip             | 0.05 GB|
| ``isic2018``     | ISIC archive on S3, Task 1 test input + ground truth            | 2.4 GB |
| ``refuge2``      | Kaggle ``victorlemosml/refuge2``, test split only                  | 1.4 GB |
| ``camus``        | Kaggle ``mykcs01/camus-public-202404``, test patients 451-500       | 0.45 GB|
| ``medclipsamv2`` | MedCLIP-SAMv2 split bundle on Google Drive (UDIAT breast US,    | 2.5 GB |
|                | lung X-ray, brain MRI, lung CT)                                  |        |

Kaggle sources need the ``kaggle`` package and an API token (``~/.kaggle/kaggle.json``).
On a Kaggle notebook you can instead attach the dataset with "Add Input" and point
``prepare-data --source`` at ``/kaggle/input/...``. Google Drive needs ``gdown``.
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ISIC_URLS = [
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip",
]
BUSI_KAGGLE = "aryashah2k/breast-ultrasound-images-dataset"
REFUGE2_KAGGLE = "victorlemosml/refuge2"
CAMUS_KAGGLE = "mykcs01/camus-public-202404"
CAMUS_FILES = ("_2CH_ED", "_2CH_ES", "_4CH_ED", "_4CH_ES")  # also matches the *_gt files
MEDCLIPSAMV2_GDRIVE_ID = "1uYtyg3rClE-XXPNuEz7s6gYq2p48Z08p"

DATASETS = ("busi", "kvasir", "isic2018", "refuge2", "camus", "medclipsamv2")


def download_data(name: str, out_dir: str | Path = "raw") -> Path:
    """Download dataset ``name`` into ``out_dir/name`` and return that folder."""
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    if name == "busi":
        _kaggle_dataset(BUSI_KAGGLE, out)
    elif name == "kvasir":
        _unzip(_url(KVASIR_URL, out), out)
    elif name == "isic2018":
        for url in ISIC_URLS:
            _url(url, out)  # kept zipped; prepare-data reads the zips directly
    elif name == "refuge2":
        _kaggle_files(REFUGE2_KAGGLE, out, lambda f: f.startswith("REFUGE2/test/"))
    elif name == "camus":
        test_patients = tuple(f"patient{p:04d}" for p in range(451, 501))
        _kaggle_files(
            CAMUS_KAGGLE, out, lambda f: any(p in f for p in test_patients) and any(s in f for s in CAMUS_FILES)
        )
    elif name == "medclipsamv2":
        _gdrive(MEDCLIPSAMV2_GDRIVE_ID, out / "data.zip")
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    logger.info("Raw %s files are in %s. Next: python -m zsmedseg prepare-data %s --source %s", name, out, name, out)
    return out


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------


def _url(url: str, out: Path) -> Path:
    target = out / url.rsplit("/", 1)[-1]
    if target.is_file():
        logger.info("%s already downloaded", target.name)
        return target
    logger.info("Downloading %s ...", url)
    partial = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    if target.suffix == ".zip" and not zipfile.is_zipfile(partial):  # e.g. a "file has moved" page
        partial.unlink()
        raise RuntimeError(f"{url} did not return a zip file; the dataset may have moved")
    partial.rename(target)
    return target


def _unzip(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if ".." in Path(member).parts or Path(member).is_absolute():
                raise ValueError(f"Unsafe path in {archive}: {member}")
        zf.extractall(out)


def _kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as err:
        raise ImportError("pip install kaggle, and put your API token in ~/.kaggle/kaggle.json") from err
    api = KaggleApi()
    api.authenticate()
    return api


def _kaggle_dataset(ref: str, out: Path) -> None:
    logger.info("Downloading Kaggle dataset %s ...", ref)
    _kaggle_api().dataset_download_files(ref, path=str(out), unzip=True, quiet=False)


def _kaggle_files(ref: str, out: Path, keep) -> None:
    """Download only the files of ``ref`` whose name satisfies ``keep`` (one request each)."""
    api = _kaggle_api()
    names, token = [], None
    while True:
        page = api.dataset_list_files(ref, page_token=token, page_size=1000)
        names += [f.name for f in page.files if keep(f.name)]
        token = getattr(page, "next_page_token", None)
        if not token:
            break
    logger.info("Downloading %d files from Kaggle dataset %s ...", len(names), ref)
    for i, name in enumerate(names, 1):
        dest = out / Path(name).parent
        if (dest / Path(name).name).is_file():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_file(ref, name, path=str(dest), quiet=True)
        zipped = dest / (Path(name).name + ".zip")  # Kaggle zips large single files
        if zipped.is_file():
            _unzip(zipped, dest)
            zipped.unlink()
        if i % 100 == 0:
            logger.info("  %d / %d", i, len(names))


def _gdrive(file_id: str, target: Path) -> None:
    if target.is_file():
        logger.info("%s already downloaded", target.name)
        return
    try:
        import gdown
    except ImportError as err:
        raise ImportError("pip install gdown (needed for the Google Drive download)") from err
    gdown.download(id=file_id, output=str(target), quiet=False)
