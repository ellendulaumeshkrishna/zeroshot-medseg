# AGENTS.md: the whole project on one page

Read this first. It is written for an AI assistant (or a new developer) joining the project:
everything needed to work on it correctly, without reading the rest of the repository first.
Details live in `README.md` (overview), `HOW_TO_RUN.md` (commands and expected output) and
`docs/CODE_WALKTHROUGH.md` (code file by file).

---

## 1. What the project is

**Task.** Segment a structure (tumour, polyp, skin lesion, optic disc, lungs, ...) in a 2D
medical image, given only the image and a short **text description** of the target.

**Constraint.** **Zero-shot and training-free**: no model is trained or fine-tuned, and no
labelled image of the target dataset is used. Three public pretrained models are combined
and kept frozen. The only task-specific input is text (target phrases + background phrases).

**Why.** The best segmentation models (U-Net, nnU-Net) are supervised: they need thousands
of expert-labelled images and must be retrained for every new task.

**Origin.** B.Tech project at IIT (BHU) Varanasi by Umesh Krishna and Ganesh Bugatha,
supervised by Dr. Indra Deep Mastan.

**One-line summary.** *CLIP points, MedSAM2 draws*: BMC-CLIP (made dense by CorrCLIP) finds
**where** the target is; MedSAM2 draws **its outline**.

---

## 2. The method (this is the only method; there are no variants)

For one image, `src/zsmedseg/pipeline.py: ZeroShotSegmenter.segment()`:

| Step | What | Code | Key settings |
|---|---|---|---|
| 0 (once) | Each class phrase is put into 12 medical templates, embedded by BMC-CLIP's text encoder, averaged over templates then synonyms → one unit vector for background, one for target | `models/text_encoder.py` | templates `biomedical` |
| 1 | MedSAM2 automatic mask generator splits the image into regions | `saliency/regions.py`, `models/medsam2.py` | 16x16 point grid, thresholds 0.5 |
| 2 | Target map: image rescaled to short side 336, 336x336 sliding windows (stride 112); per window DINO ViT-B/8 patch similarity + CorrCLIP rebuilds BMC-CLIP's last attention (a patch averages only similar patches **of its own region**); cosine with class vectors; `softmax(40 · cos)` → P(target) per pixel | `saliency/estimator.py`, `models/corrclip.py`, `models/dino.py` | λ = 40, CorrCLIP τ = 4, spatial 1.0, semantic 0.5, merge eps 0.2 |
| 3 | 5 highest, well-separated peaks of the map: take max, blank a square of half-side 0.1·min(H,W), repeat | `prompting/strategies.py` | K = 5, 0.10 |
| 4 | MedSAM2 point prompt per peak → 3 masks each with a predicted IoU → 15 candidates | `models/medsam2.py: PointPromptSegmenter` | `multimask_output=True` |
| 5 | Keep the candidate with the highest predicted IoU. For targets in two parts (both lungs) also keep the next most confident mask from another point overlapping ≤ 10 %, and take the union | `selection/rules.py: MaskSelector` | `num_objects` 1 (2 for lungs) |

Models: **BMC-CLIP** ViT-L/14 (`BIOMEDICA/BMC_CLIP_CF`, 5.1 GB), **MedSAM2** Hiera-tiny 512 px
(`wanglab/MedSAM2`, `MedSAM2_latest.pt`, 156 MB), **DINO** ViT-B/8 (torch.hub, automatic).

Settings origin: λ and the CorrCLIP values are CorrCLIP defaults; K = 5, the 0.10 square and
the 16x16 grid were chosen once on BUSI, then frozen for every other dataset. All defaults are
in `src/zsmedseg/config.py`; dataset YAMLs set only paths, prompts and (lungs) `num_objects`.

Formulas used in slides/report ↔ code: (1) class embedding → `encode_classes`; (2) CorrCLIP
attention `A = softmax(τ·sim_DINO)` masked to same region, `f = A·v` → `_scope_attention` +
`torch.bmm(weights, v)`; (3) `P = softmax(λ·cos)` → end of `SaliencyEstimator.__call__`;
(4) peak picking → `MultiPeakPrompts`; (5) argmax predicted IoU → `MaskSelector.choose`;
(6) Dice = 2|M∩G|/(|M|+|G|) → `ImageRecord.dice`.

---

## 3. Results (the numbers to quote)

Metric: **mean per-image Dice** over images that contain the target, full test sets, same
settings everywhere.

| Config | Dataset | Test images | Dice | NSD (2 px) | Published zero-shot best |
|---|---|---|---|---|---|
| `udiat` | UDIAT breast ultrasound | 113 | **82.19** | 85.95 | MedCLIP-SAMv2 77.76 (supervised nnU-Net 82.47) |
| `lung_xray` | COVID-QU-Ex chest X-ray, both lungs | 957 | **94.59** | 97.11 | MedCLIP-SAMv2 75.79 |
| `isic2018` | ISIC 2018 skin lesion | 1000 | **90.40** | | none |
| `kvasir` | Kvasir-SEG polyp | 1000 | **88.39** | | none |
| `refuge2` | REFUGE2 optic disc | 400 | **81.57** | | none |
| `busi` | BUSI breast lesion | 780 | **76.96** | | none |
| `brain_mri` | brain tumour MRI | 600 | 49.54 | 52.98 | limitation |
| `lung_ct` | chest CT lungs | 1800 | 12.88 | 13.97 | limitation |
| `camus` | CAMUS left ventricle (echo) | 200 | 2.88 | | limitation |

`udiat`, `lung_xray`, `brain_mri`, `lung_ct` follow the MedCLIP-SAMv2 protocol
(`configs/datasets/_medclipsamv2_protocol.yaml`: masks > 200, prediction resized to mask size,
NSD at 2 px), so they are directly comparable with the published numbers.
Quick-check values (first 20 images) are in `HOW_TO_RUN.md` section 3.

Limitations (why the last three fail): the target looks like its surroundings (brain MRI →
CLIP cannot point to it); large textured organs (lung CT, heart) → one point makes MedSAM2
outline only a small part; all-or-nothing when no point lands on the target; always predicts
a mask, even on target-free images.

---

## 4. Repository map

```
src/zsmedseg/pipeline.py        the method (read first)
src/zsmedseg/config.py          all settings with defaults and comments; unknown keys raise
src/zsmedseg/cli.py             python -m zsmedseg <download-weights|download-data|prepare-data|evaluate|predict|serve|show-config>
src/zsmedseg/models/            clip_backbone, text_encoder, dino, corrclip, medsam2
src/zsmedseg/saliency/          estimator (target map), regions (MedSAM2 automatic)
src/zsmedseg/prompting/         5 peaks;  selection/  candidate choice
src/zsmedseg/evaluation/        metrics (Dice/IoU/NSD), runner (dataset loop, reports)
src/zsmedseg/tools/             download_weights, download_data, prepare_data (all 9 datasets)
src/zsmedseg/webapp/            demo website: FastAPI server.py + static/ (plain HTML/CSS/JS, 6 example images)
configs/datasets/*.yaml         one per dataset: root, mask pattern, prompts
scripts/evaluate_all.py         all prepared datasets → one table (models loaded once)
notebooks/quickstart.ipynb      Colab/Kaggle, choose DATASET in one cell
tests/                          CPU-only, no weights (37 tests)
```

Data layout produced by `prepare-data`: `data/<name>/images/<stem>.png` +
`data/<name>/masks/<stem>_mask.png` (0/1). The MedCLIP-SAMv2 bundle keeps its own
`data/medclipsamv2/<dataset>/{test,val}_{images,masks}`. `data/`, `raw/`, `weights/`,
`outputs/`, `MedSAM2/` are git-ignored.

---

## 5. Running and verifying

```bash
python -m zsmedseg download-weights
python -m zsmedseg download-data kvasir && python -m zsmedseg prepare-data kvasir
python -m zsmedseg evaluate -c configs/datasets/kvasir.yaml --set runtime.limit=20   # must print Dice 83.11
python -m zsmedseg evaluate -c configs/datasets/kvasir.yaml                          # 88.39
python scripts/evaluate_all.py --limit 20                                            # all datasets, quick
python -m zsmedseg serve                                                             # http://127.0.0.1:8000
pytest                                                                               # 37 passed
```

Override any setting with `--set key=value` (e.g. `weights.bmc_clip=/path`,
`runtime.save_overlays=true`). Outputs: `outputs/<dataset>/<timestamp>/{metrics.json,per_image.csv,config.yaml}`.

**Exact reproduction needs the pinned versions** (`requirements-exact.txt`: Python 3.10/3.11,
torch 2.1.2 + cu121, numpy 1.23.5, scipy 1.15.3, scikit-learn 1.7.0). Newer PyTorch/NumPy
change results slightly (first 20 Kvasir: 83.93 instead of 83.11). Colab is Python 3.12, so
close but not exact; Kaggle (3.11) can be exact. ~1.5 s/image on a 4 GB RTX 3050.

Python API: `ZeroShotSegmenter.from_config(load_config([...]))`, `.segment(bgr) → result`
(`mask, saliency, points, candidates, kept, chosen, regions, score`), `.set_prompts(bg, target)`
switches task without reloading models.

---

## 6. Facts that are easy to get wrong

* **BUSI:** 16 images have extra mask files (`_mask_1.png`, ...). Reported results use **only
  `<name>_mask.png`** (`prepare_busi` default); `merge_extra_masks=True` is the alternative.
* **ISIC 2018** files moved (Oct 2026) to `https://isic-archive.s3.amazonaws.com/challenges/2018/`.
  Images are shrunk to 1024 px longest side (full size does not fit 4 GB GPUs).
* **REFUGE2** mask values: 0 cup, 128 rim, 255 background → optic disc = `< 255`.
* **CAMUS:** test patients 451–500, 2CH/4CH at ED/ES, target label 1, NIfTI transposed to rows/cols.
* **Lung X-ray:** images 299 px, masks 256 px → prediction resized to the mask before scoring.
* **MedCLIP-SAMv2 bundle:** Google Drive id `1uYtyg3rClE-XXPNuEz7s6gYq2p48Z08p`; only test and
  val splits are unpacked.
* The re-implementation reproduces the original research code image by image; edge-case
  behaviour in `models/corrclip.py` (padded-border region id, tie-breaking, "smallest region
  id = unsegmented") is deliberate. Do not "clean it up".
* `models/corrclip.py: _block_forward` exists because open_clip would turn a boolean attention
  mask into a float bias.

---

## 7. Rules for working on this repository

1. **Only the final method exists.** Never add, mention or offer a "Version 1/2", a baseline
   (plain CorrCLIP/BMC-CLIP thresholding), a presence gate, or alternative selection rules —
   not in code, configs, docs or UI. The authors were explicit about this.
2. **Report per-image Dice** (and NSD where the protocol has it). Do not report misses
   (Dice < 10 counts), standard deviations or variance in user-facing material. The three
   failing datasets appear only as limitations; the website shows the six good ones.
3. **Never tune a setting on the set you report.** Use the validation split
   (`--set dataset.images_dir=val_images --set dataset.masks_dir=val_masks`) for any check.
4. **Full test sets only** for reported numbers; `runtime.limit` is for quick checks.
5. **A change to the pipeline must keep the quick-check values** in `HOW_TO_RUN.md` section 3
   (with the pinned versions). Run `python scripts/evaluate_all.py --limit 20` and `pytest`.
6. Keep the code plain: one class per stage, no plug-in registries. Match the surrounding
   comment style. Line length 120 (`ruff check src tests scripts`). Text files use LF.
7. Do not commit weights, data, outputs, `MedSAM2/`, or any credential (`kaggle.json`, `.env`).
8. Cite the works in README "Acknowledgements" when writing about the method.

---

## 8. Glossary

**Target map / saliency**: per-pixel probability of the target from BMC-CLIP + CorrCLIP.
**Regions**: MedSAM2 automatic masks that confine CorrCLIP's patch averaging ("scope
reconstruction"). **Candidates**: the 15 MedSAM2 masks (5 points × 3). **Predicted IoU**:
MedSAM2's own quality estimate of a mask, used to choose. **NSD**: normalised surface Dice,
share of boundary within 2 px. **Zero-shot**: no image or label of the test task used for
training; **training-free**: no weights updated at all.
