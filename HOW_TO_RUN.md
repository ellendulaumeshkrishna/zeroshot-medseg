# How to run and check this project

Each step lists the command to type and the output you should see. Pick one place to run it:

* [A. Your own computer](#a-your-own-computer) (Windows, Linux or macOS, NVIDIA GPU with 4 GB or more)
* [B. Google Colab](#b-google-colab) (free GPU, nothing to install locally)
* [C. Kaggle](#c-kaggle) (free GPU, datasets can be attached without downloading)

Then: [get the data](#2-get-a-dataset), [check that it works](#3-check-that-it-works),
[reproduce the results](#4-reproduce-the-results), [segment your own images](#5-segment-your-own-images),
[open the website](#6-the-demo-website). If something fails, see [troubleshooting](#7-troubleshooting).

What you need in all cases: Python 3.10 or newer, about 6 GB of disk for the model weights,
and internet access the first time (weights, DINO, datasets).

---

## 1. Install

### A. Your own computer

Use **Python 3.10 or 3.11** (the exact library versions below have no builds for 3.12).

```bash
git clone <this repository> zeroshot-medseg
cd zeroshot-medseg
python -m venv .venv
```

Activate the environment. Windows **cmd**: `.venv\Scripts\activate`.
Windows **PowerShell**: `.venv\Scripts\Activate.ps1`. Linux/macOS: `source .venv/bin/activate`.

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-exact.txt
pip install --no-deps -e .
pip install fastapi uvicorn python-multipart kaggle gdown pytest "surface-distance @ git+https://github.com/google-deepmind/surface-distance.git"
git clone https://github.com/bowang-lab/MedSAM2.git
```

Why each line:

| Line | Purpose |
|---|---|
| `torch==2.1.2 ...` | PyTorch with CUDA 12.1 (works with any recent NVIDIA driver). |
| `requirements-exact.txt` | open_clip, OpenCV, Hydra (needed by MedSAM2) and the rest, at the exact versions that produced the reported numbers. |
| `--no-deps -e .` | Installs this package in editable mode without touching your PyTorch. |
| `pip install fastapi ...` | Optional: the website (`fastapi uvicorn python-multipart`), dataset downloads (`kaggle gdown`), tests (`pytest`), the NSD metric (`surface-distance`). |
| `git clone ... MedSAM2` | MedSAM2's `sam2` package. Kept inside the repository; it is found automatically, nothing is compiled. |

**Check:**

```bash
python -m zsmedseg --help
```

Expected: a list of commands (`download-weights`, `download-data`, `prepare-data`,
`evaluate`, `predict`, `serve`, `show-config`).

**Exact versus recent library versions.** With these pinned versions you get exactly the
numbers in this guide. Newer PyTorch / NumPy (for example `pip install torch` and
`requirements.txt`, as on Colab) work too, but floating-point differences change some
predictions slightly: on the first 20 Kvasir images, Dice 83.93 instead of 83.11.

### B. Google Colab

Open `notebooks/quickstart.ipynb` in Colab (*File > Upload notebook*), choose
*Runtime > Change runtime type > T4 GPU*, and run the cells from top to bottom. Or paste this
into a code cell:

```python
%cd /content
!git clone <this repository> zeroshot-medseg          # or upload the folder as a zip and unzip it
%cd /content/zeroshot-medseg
!pip install -q -r requirements.txt
!pip install -q --no-deps -e .
!pip install -q fastapi uvicorn python-multipart kaggle gdown "surface-distance @ git+https://github.com/google-deepmind/surface-distance.git"
!git clone -q https://github.com/bowang-lab/MedSAM2.git
!python -m zsmedseg --help
```

PyTorch is already installed on Colab, so it is not reinstalled. Colab uses Python 3.12 and
recent library versions, so expect numbers that differ slightly from the reported ones (see
*Exact versus recent library versions* above). On Kaggle (Python 3.11) you can install the
exact versions: `!pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121`
and `!pip install -q -r requirements-exact.txt` instead of `requirements.txt`.

For the Kaggle-hosted datasets (BUSI, REFUGE2, CAMUS) Colab needs your Kaggle API token:
on kaggle.com go to *Settings > API > Create New Token*, which downloads `kaggle.json`; then

```python
from google.colab import files; files.upload()          # choose kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

### C. Kaggle

Create a notebook, then in the right-hand panel set *Accelerator* to **GPU T4 x2** (or P100)
and switch **Internet on** (needs a phone-verified account). Run the same cell as for Colab
with `/kaggle/working` instead of `/content`.

On Kaggle you do not need to download the Kaggle-hosted datasets: click *Add Input*, search
for the dataset and attach it; it appears read-only under `/kaggle/input/`. See the table in
the next section for the names to search.

---

## 2. Get a dataset

First the model weights (once; BMC-CLIP is 5.1 GB, MedSAM2 156 MB; DINO downloads itself
on first use):

```bash
python -m zsmedseg download-weights
```

Expected: `weights/BMC_CLIP_CF.pt` and `weights/MedSAM2_latest.pt`.

Every dataset is then two commands: `download-data` puts the original files in `raw/<name>/`,
and `prepare-data` converts them to `data/<name>/images/` + `data/<name>/masks/`, which is
where the config files look.

| Config (`configs/datasets/…`) | Commands | Size | Images | Notes |
|---|---|---|---|---|
| `kvasir.yaml` | `download-data kvasir` then `prepare-data kvasir` | 46 MB | 1000 | public URL |
| `isic2018.yaml` | `download-data isic2018` then `prepare-data isic2018` | 2.4 GB | 1000 | public URL; images shrunk to 1024 px longest side |
| `busi.yaml` | `download-data busi` then `prepare-data busi` | 270 MB | 780 | Kaggle `aryashah2k/breast-ultrasound-images-dataset` |
| `refuge2.yaml` | `download-data refuge2` then `prepare-data refuge2` | 1.4 GB | 400 | Kaggle `victorlemosml/refuge2`, test split |
| `camus.yaml` | `download-data camus` then `prepare-data camus` | 450 MB | 200 | Kaggle `mykcs01/camus-public-202404`, patients 451-500 |
| `udiat.yaml`, `lung_xray.yaml`, `brain_mri.yaml`, `lung_ct.yaml` | `download-data medclipsamv2` then `prepare-data medclipsamv2` | 2.5 GB | 113, 957, 600, 1800 | one Google Drive bundle (from the MedCLIP-SAMv2 authors) for all four |

(Prefix every command with `python -m zsmedseg`.)

Kaggle sources need a Kaggle token (`~/.kaggle/kaggle.json`, see Colab above; on Windows it is
`C:\Users\<you>\.kaggle\kaggle.json`). **On a Kaggle notebook**, attach the dataset instead and
point `prepare-data` at it, for example:

```bash
python -m zsmedseg prepare-data busi    --source /kaggle/input/breast-ultrasound-images-dataset
python -m zsmedseg prepare-data refuge2 --source /kaggle/input/refuge2
python -m zsmedseg prepare-data camus   --source /kaggle/input/camus-public-202404
```

`prepare-data` keeps only the test split, so attaching the full dataset is fine.

**Check:** `prepare-data` ends with `Wrote <N> images to data/<name>` where N is the number in
the table (for `medclipsamv2`: 8904, the test and validation images of all four datasets).

**Your own dataset:** put images and masks in two folders and run
`python -m zsmedseg prepare-data generic --images <img folder> --masks <mask folder> --out data/mine`,
copy `configs/datasets/_template.yaml` to `configs/datasets/mine.yaml`, and write the target
and background phrases in it.

---

## 3. Check that it works

**Unit tests** (CPU only, no weights, a few seconds):

```bash
pytest
```

Expected: `.. passed` with no failures.

**Quick run on 20 images** (about a minute after the models have loaded):

```bash
python -m zsmedseg evaluate -c configs/datasets/kvasir.yaml --set runtime.limit=20
```

Expected output, ending with:

```
Images: 20 (20 with polyp, 0 without)
Dice (per image, mean over polyp images): 83.11
...
Reports written to outputs/kvasir/<date_time>
```

The first 20 images of each dataset give these exact values with the pinned library
versions. A clearly different number means the weights or the data preparation differ; a
small difference usually means newer library versions (see section 1):

| Config | Dice on the first 20 images |
|---|---|
| `busi` | 81.43 |
| `kvasir` | 83.11 |
| `isic2018` | 87.16 |
| `refuge2` | 73.95 |
| `udiat` | 83.19 |
| `lung_xray` | 94.59 |
| `brain_mri` | 47.19 |
| `lung_ct` | 2.53 |
| `camus` | 6.89 |

To *see* what happened, add `--set runtime.save_overlays=true`: every image gets a panel
(image | target map with the 5 points | prediction | ground truth) in
`outputs/<dataset>/<date_time>/overlays/`.

---

## 4. Reproduce the results

One dataset (full test set):

```bash
python -m zsmedseg evaluate -c configs/datasets/udiat.yaml
```

All datasets that are prepared, one table at the end:

```bash
python scripts/evaluate_all.py
```

Expected values (mean per-image Dice; NSD at 2 px where the MedCLIP-SAMv2 protocol reports it):

| Config | Test images | Dice | NSD |
|---|---|---|---|
| `udiat` | 113 | 82.19 | 85.95 |
| `lung_xray` | 957 | 94.59 | 97.11 |
| `isic2018` | 1000 | 90.40 | |
| `kvasir` | 1000 | 88.39 | |
| `refuge2` | 400 | 81.57 | |
| `busi` | 780 | 76.96 | |
| `brain_mri` | 600 | 49.54 | 52.98 |
| `lung_ct` | 1800 | 12.88 | 13.97 |
| `camus` | 200 | 2.88 | |

The last three are the known limitations (see the README). Speed: about 1.5 s per image on a
4 GB RTX 3050 laptop GPU, faster on a Colab/Kaggle T4.

Each run writes `outputs/<dataset>/<date_time>/`:

| File | Content |
|---|---|
| `metrics.json` | per-image Dice / IoU means (and NSD), dataset-aggregate IoU / Dice |
| `per_image.csv` | for every image: intersection, predicted and true area, Dice, IoU, MedSAM2 score |
| `config.yaml` | the exact settings of the run |
| `masks/`, `overlays/` | only with `--set runtime.save_masks=true` / `--set runtime.save_overlays=true` |

Settings can be changed without editing files: `--set prompting.num_points=3`,
`--set runtime.device=cpu`, and so on. `python -m zsmedseg show-config -c <config>` prints
every setting; their meaning is written next to each one in `src/zsmedseg/config.py`.

---

## 5. Segment your own images

No ground truth needed. Describe the target and its surroundings in words:

```bash
python -m zsmedseg predict --input path/to/image_or_folder \
    --target "skin lesion" "melanoma" --background "healthy skin" "background"
```

Or reuse a dataset's prompts: `python -m zsmedseg predict --input scan.png -c configs/datasets/busi.yaml`.

Expected: `outputs/predictions/masks/` (0/255 PNG masks), `saliency/` (target maps) and
`overlays/` (image | target map with points | prediction).

From Python:

```python
import cv2
from zsmedseg import ZeroShotSegmenter, load_config

segmenter = ZeroShotSegmenter.from_config(load_config(["configs/datasets/kvasir.yaml"]))
result = segmenter.segment(cv2.imread("polyp.png"))   # BGR image as read by OpenCV
result.mask          # H x W bool, the segmentation
result.saliency      # H x W float, the target map
result.points        # the 5 point prompts (x, y)
result.candidates    # all 15 MedSAM2 masks with their predicted IoU
segmenter.set_prompts(["healthy skin", "background"], ["skin lesion"])   # switch task, models stay loaded
```

---

## 6. The demo website

```bash
python -m zsmedseg serve
```

Open **http://127.0.0.1:8000**. The page opens at once; the models load in the background
(about a minute; the badge in the top right turns green when ready).

* Drop an image, or click one of the six test images.
* Choose the task (fills in the target and background phrases) or type your own description.
* Optionally add the expert mask to get the Dice score (the test images bring theirs).
* *Segment* shows the target map with the five points, all fifteen MedSAM2 candidates with
  their predicted IoU (the kept one is framed), and the result, which you can download.

Weights stored elsewhere: `python -m zsmedseg serve --set weights.bmc_clip=D:/models/BMC_CLIP_CF.pt --set weights.medsam2=D:/models/MedSAM2_latest.pt`.
On Colab or Kaggle the website is not reachable from your browser without a tunnel; use the
notebook cells there instead.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: The sam2 package from MedSAM2 is required` | Run `git clone https://github.com/bowang-lab/MedSAM2.git` inside the repository folder, or set `ZSMEDSEG_MEDSAM2_DIR` to your clone. |
| `No module named 'efficient_track_anything'` | You installed Meta's SAM 2 instead of MedSAM2. Use the MedSAM2 repository, which contains both packages. |
| `BMC-CLIP checkpoint not found` | Run `download-weights`, or pass `--set weights.bmc_clip=<path>`. |
| `Raw data not found at raw/<name>` | Run `download-data <name>` first, or give `prepare-data` a `--source`. |
| `401 Unauthorized` from Kaggle | `kaggle.json` is missing or in the wrong folder (see section 2). |
| `No mask for <image>` | The dataset's `mask_pattern` does not match your mask file names (e.g. use `"{stem}.png"`). |
| `CUDA out of memory` | Close other programs using the GPU. Very large images: shrink them (the website does this automatically above 1024 px). |
| DINO download fails | `torch.hub` needs internet on first use (on Kaggle, switch Internet on). |
| Very slow | Check that the log says `Building pipeline on cuda`. On CPU each image takes minutes. |
| Website says "Model loading failed" | The message shows the reason, usually a wrong weights path (see section 6). |
