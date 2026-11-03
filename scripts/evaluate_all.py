"""Evaluate every dataset whose data is prepared and print one results table.

    python scripts/evaluate_all.py                       # all prepared datasets, full test sets
    python scripts/evaluate_all.py --limit 20            # quick check: first 20 images of each
    python scripts/evaluate_all.py --only udiat kvasir   # a subset
    python scripts/evaluate_all.py --set weights.bmc_clip=D:/models/BMC_CLIP_CF.pt

Datasets whose folder does not exist are skipped (prepare them with `prepare-data` first).
The models are loaded once and reused; only the text prompts change between datasets.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zsmedseg.config import load_config
from zsmedseg.data import SegmentationFolderDataset
from zsmedseg.evaluation import make_run_dir, run_dataset
from zsmedseg.pipeline import ZeroShotSegmenter
from zsmedseg.selection import MaskSelector
from zsmedseg.utils.logging_utils import setup_logging

DATASETS = ["udiat", "lung_xray", "isic2018", "kvasir", "refuge2", "busi", "brain_mri", "lung_ct", "camus"]
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--limit", type=int, help="Only the first N images of each dataset (quick check).")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    overrides = list(args.overrides) + ([f"runtime.limit={args.limit}"] if args.limit else [])
    configs = {name: load_config([CONFIGS / f"{name}.yaml"], overrides) for name in args.only}
    ready = {n: c for n, c in configs.items() if (Path(c.dataset.root) / c.dataset.images_dir).is_dir()}
    for name in configs.keys() - ready.keys():
        logging.warning("Skipping %s: %s not found (run prepare-data first).", name, configs[name].dataset.root)
    if not ready:
        return

    segmenter = ZeroShotSegmenter.from_config(next(iter(ready.values())))  # models are shared by all datasets
    rows = []
    for name, cfg in ready.items():
        segmenter.set_prompts(cfg.dataset.background_prompts, cfg.dataset.target_prompts)
        segmenter.selector = MaskSelector(cfg.selection.num_objects, cfg.selection.max_overlap)
        dataset = SegmentationFolderDataset(cfg.dataset, limit=cfg.runtime.limit)
        summary = run_dataset(segmenter, dataset, cfg, make_run_dir(cfg))
        rows.append((name, summary["num_images"], summary["per_image_dice_mean"], summary.get("per_image_nsd_mean")))

    print("\n| Dataset | Images | Dice | NSD |\n|---|---|---|---|")
    for name, n, dice, nsd in rows:
        print(f"| {name} | {n} | {dice:.2f} | {'' if nsd is None else f'{nsd:.2f}'} |")


if __name__ == "__main__":
    main()
