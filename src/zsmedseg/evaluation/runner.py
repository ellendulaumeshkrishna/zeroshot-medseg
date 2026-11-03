"""Run the pipeline over a dataset and write reports.

Outputs, in ``<output_dir>/<dataset name>/<timestamp>/``:

* ``metrics.json``   - the summary from :class:`SegmentationMeter`;
* ``per_image.csv``  - counts, Dice, IoU and the chosen mask's score for every image;
* ``config.yaml``    - the exact configuration used;
* ``masks/``         - predicted masks as 0/255 PNGs (``runtime.save_masks``);
* ``overlays/``      - image | saliency + points | prediction | ground truth panels
  (``runtime.save_overlays``).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..data import Sample, SegmentationFolderDataset
from ..pipeline import ZeroShotSegmenter
from ..utils.visualization import make_panel
from .metrics import SegmentationMeter, format_summary

logger = logging.getLogger(__name__)


def _prefetch(
    dataset: SegmentationFolderDataset, workers: int
) -> Iterator[tuple[Sample, np.ndarray, np.ndarray | None]]:
    """Yield ``(sample, image, mask)`` while the next images are read in background threads."""

    def load(sample: Sample):
        return sample, dataset.load_image(sample), dataset.load_mask(sample)

    if workers <= 0:
        yield from map(load, dataset)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(load, dataset)


def make_run_dir(cfg: Config) -> Path:
    run_dir = Path(cfg.runtime.output_dir) / cfg.dataset.name / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_dataset(
    segmenter: ZeroShotSegmenter,
    dataset: SegmentationFolderDataset,
    cfg: Config,
    run_dir: Path,
) -> dict[str, float]:
    """Segment every image; evaluate when masks exist. Returns the metric summary."""
    (run_dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    mask_dir = run_dir / "masks"
    overlay_dir = run_dir / "overlays"
    if cfg.runtime.save_masks:
        mask_dir.mkdir(exist_ok=True)
    if cfg.runtime.save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    meter = SegmentationMeter(nsd_tolerance=cfg.evaluation.nsd_tolerance)
    scores: dict[str, float] = {}
    num_resized = 0
    progress = tqdm(_prefetch(dataset, cfg.runtime.num_workers), total=len(dataset), desc=cfg.dataset.name)
    for sample, image, gt in progress:
        result = segmenter.segment(image)
        scores[sample.name] = result.score
        stem = Path(sample.name).stem
        if cfg.runtime.save_masks:
            cv2.imwrite(str(mask_dir / f"{stem}.png"), result.mask.astype(np.uint8) * 255)
        if cfg.runtime.save_overlays:
            panel = make_panel(image, result.saliency, result.points, result.mask, gt)
            cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), panel)
        if gt is not None:
            prediction = result.mask
            if prediction.shape != gt.shape and cfg.evaluation.resize_prediction_to_mask:
                prediction = (
                    cv2.resize(prediction.astype(np.uint8), gt.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                )
                num_resized += 1
            meter.update(sample.name, prediction, gt)
            summary = meter.summary()
            progress.set_postfix(dice=f"{summary['per_image_dice_mean']:.2f}")

    if not dataset.has_masks:
        logger.info("No masks folder: predictions saved, nothing to evaluate.")
        return {}

    summary = meter.summary()
    summary["predictions_resized_to_mask"] = num_resized
    if num_resized:
        logger.info("%d predictions were resized to their mask size before scoring.", num_resized)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(run_dir / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        rows = meter.rows()
        writer = csv.DictWriter(fh, fieldnames=[*rows[0].keys(), "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "score": scores[row["name"]]})
    logger.info("Results for %s:\n%s", cfg.dataset.name, format_summary(summary, cfg.dataset.target_name))
    logger.info("Reports written to %s", run_dir)
    return summary
