"""Command-line interface: ``python -m zsmedseg <command> ...`` (or ``zsmedseg <command>``).

Commands
--------
download-weights  fetch BMC-CLIP and MedSAM2 from the Hugging Face Hub
download-data     fetch the raw files of a dataset           -> raw/<name>/
prepare-data      convert raw files to images/ + masks/      -> data/<name>/
evaluate          segment a dataset and report Dice / IoU (and NSD where configured)
predict           segment an image or a folder of images (no masks needed)
serve             start the demo website on http://localhost:8000
show-config       print the fully merged configuration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("zsmedseg")

# Where prepare-data writes each dataset by default (the configs read from these folders).
DATA_DIRS = {
    "busi": "data/busi",
    "kvasir": "data/kvasir",
    "isic2018": "data/isic2018",
    "refuge2": "data/refuge2",
    "camus": "data/camus",
    "medclipsamv2": "data/medclipsamv2",
}


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="YAML",
        help="Dataset config file, e.g. configs/datasets/busi.yaml.",
    )
    parser.add_argument(
        "-s",
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override any setting, e.g. --set runtime.limit=20 (repeatable).",
    )


def _load(args: argparse.Namespace):
    from .config import load_config

    return load_config(args.config, args.overrides)


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .data import SegmentationFolderDataset
    from .evaluation import make_run_dir, run_dataset
    from .pipeline import ZeroShotSegmenter

    cfg = _load(args)
    dataset = SegmentationFolderDataset(cfg.dataset, limit=cfg.runtime.limit)
    logger.info("Dataset '%s': %d images (masks: %s)", cfg.dataset.name, len(dataset), dataset.has_masks)
    if cfg.runtime.limit:
        logger.warning("runtime.limit=%d: subset numbers are for quick checks only.", cfg.runtime.limit)
    segmenter = ZeroShotSegmenter.from_config(cfg)
    run_dataset(segmenter, dataset, cfg, make_run_dir(cfg))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from .config import load_config
    from .data import read_image_bgr
    from .pipeline import ZeroShotSegmenter
    from .utils.visualization import colorize_saliency, make_panel

    overrides = list(args.overrides)
    if args.target:
        overrides.append(f"dataset.target_prompts={json.dumps(args.target)}")
    if args.background:
        overrides.append(f"dataset.background_prompts={json.dumps(args.background)}")
    cfg = load_config(args.config, overrides)

    source = Path(args.input)
    extensions = {e.lower() for e in cfg.dataset.image_extensions}
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in extensions) if source.is_dir() else [source]
    if cfg.runtime.limit:
        images = images[: cfg.runtime.limit]
    if not images:
        logger.error("No images found at %s", source)
        return 1

    out = Path(args.output_dir)
    for sub in ("masks", "overlays", "saliency"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    segmenter = ZeroShotSegmenter.from_config(cfg)
    for path in images:
        image = read_image_bgr(path)
        result = segmenter.segment(image)
        cv2.imwrite(str(out / "masks" / f"{path.stem}.png"), result.mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "saliency" / f"{path.stem}.png"), colorize_saliency(result.saliency))
        cv2.imwrite(
            str(out / "overlays" / f"{path.stem}.jpg"), make_panel(image, result.saliency, result.points, result.mask)
        )
        logger.info("%s: mask area %d px, predicted IoU %.3f", path.name, int(result.mask.sum()), result.score)
    logger.info("Wrote %d predictions to %s", len(images), out)
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    from .tools.download_data import download_data

    download_data(args.name, args.out)
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from .tools import prepare_data as prep

    name = args.source_type
    if name == "generic":
        masks = Path(args.masks) if args.masks else None
        n = prep.prepare_generic(Path(args.images), masks, Path(args.out), args.threshold, args.mask_regex)
        logger.info("Wrote %d images to %s", n, args.out)
        return 0

    source = Path(args.source or f"raw/{name}")
    out = Path(args.out or DATA_DIRS[name])
    if not source.exists():
        sys.exit(f"Raw data not found at {source}. Run: python -m zsmedseg download-data {name}  (or pass --source)")
    converters = {
        "busi": lambda: prep.prepare_busi(_busi_root(source), out),
        "kvasir": lambda: prep.prepare_kvasir(_kvasir_root(source), out),
        "isic2018": lambda: prep.prepare_isic2018(source, out),
        "refuge2": lambda: prep.prepare_refuge2(source, out),
        "camus": lambda: prep.prepare_camus(source, out),
        "medclipsamv2": lambda: prep.prepare_medclipsamv2(source, out),
    }
    n = converters[name]()
    logger.info("Wrote %d images to %s", n, out)
    return 0


def _busi_root(source: Path) -> Path:
    """The folder that contains benign/ malignant/ normal/ (the Kaggle copy nests it)."""
    if (source / "benign").is_dir():
        return source
    found = sorted(p.parent for p in source.rglob("benign") if p.is_dir())
    if not found:
        sys.exit(f"No benign/ folder below {source}")
    return found[0]


def _kvasir_root(source: Path) -> Path:
    """The folder that contains images/ and masks/ (the zip nests it in Kvasir-SEG/)."""
    if (source / "images").is_dir():
        return source
    found = sorted(p.parent for p in source.rglob("images") if p.is_dir() and (p.parent / "masks").is_dir())
    if not found:
        sys.exit(f"No images/ + masks/ folders below {source}")
    return found[0]


def cmd_download_weights(args: argparse.Namespace) -> int:
    from .tools.download_weights import download_weights

    for name, path in download_weights(args.out, tuple(args.only or ("bmc_clip", "medsam2"))).items():
        logger.info("%s -> %s", name, path)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .webapp import run_server
    except ImportError as err:
        sys.exit(f"The website needs extra packages ({err.name}): pip install fastapi uvicorn python-multipart")

    run_server(host=args.host, port=args.port, overrides=args.overrides)
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    print(_load(args).to_yaml())
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zsmedseg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-weights", help="Download BMC-CLIP and MedSAM2.")
    p.add_argument("--out", default="weights")
    p.add_argument("--only", nargs="+", choices=["bmc_clip", "medsam2"])
    p.set_defaults(func=cmd_download_weights)

    p = sub.add_parser("download-data", help="Download the raw files of a dataset.")
    p.add_argument("name", choices=list(DATA_DIRS))
    p.add_argument("--out", default="raw", help="Parent folder; files go to <out>/<name>/.")
    p.set_defaults(func=cmd_download_data)

    p = sub.add_parser("prepare-data", help="Convert raw files to images/ + masks/.")
    p.add_argument("source_type", choices=[*DATA_DIRS, "generic"])
    p.add_argument("--source", help="Raw files (default raw/<name>, as written by download-data).")
    p.add_argument("--out", help="Output dataset root (default: the folder its config reads).")
    p.add_argument("--images", help="(generic) image folder.")
    p.add_argument("--masks", help="(generic) mask folder; omit for prediction-only data.")
    p.add_argument("--mask-regex", help='(generic) mask stem pattern, e.g. "{stem}_seg".')
    p.add_argument("--threshold", type=int, default=127, help="(generic) mask pixel > threshold = target.")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("evaluate", help="Segment a dataset and report metrics.")
    _add_config_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="Segment an image or a folder of images.")
    p.add_argument("--input", required=True, help="Image file or folder.")
    p.add_argument("--output-dir", default="outputs/predictions")
    p.add_argument("--target", nargs="+", help='Target phrase(s), e.g. --target "polyp" "colorectal polyp".')
    p.add_argument("--background", nargs="+", help="Background phrase(s).")
    _add_config_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("serve", help="Start the demo website.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="Override settings, e.g. --set weights.bmc_clip=/path/BMC_CLIP_CF.pt")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("show-config", help="Print the merged configuration.")
    _add_config_args(p)
    p.set_defaults(func=cmd_show_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .utils.logging_utils import setup_logging

    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.command == "prepare-data" and args.source_type == "generic" and not (args.images and args.out):
        sys.exit("prepare-data generic: --images and --out are required")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
