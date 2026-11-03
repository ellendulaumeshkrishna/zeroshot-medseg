"""Typed configuration for the whole pipeline.

Configuration lives in YAML files under ``configs/`` and is validated into the dataclasses
below, so a typo in a key fails loudly instead of being silently ignored.

Three layers are merged, later ones winning:

1. the dataclass defaults in this file (these reproduce the reported results);
2. one or more YAML files (a file may list parents under ``inherit:``);
3. command-line overrides such as ``--set prompting.num_points=3``.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PathLike = str | Path


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


@dataclass
class WeightsConfig:
    """Where the frozen pretrained models come from. Nothing is trained in this project."""

    # BMC-CLIP ViT-L/14 checkpoint (https://huggingface.co/BIOMEDICA/BMC_CLIP_CF).
    bmc_clip: str = "weights/BMC_CLIP_CF.pt"
    # open_clip architecture name matching the checkpoint.
    clip_arch: str = "ViT-L-14"
    # MedSAM2 checkpoint (https://huggingface.co/wanglab/MedSAM2, file MedSAM2_latest.pt).
    medsam2: str = "weights/MedSAM2_latest.pt"
    # MedSAM2 model definition (Hiera-tiny, 512 px input), shipped in this repository.
    medsam2_config: str = "configs/sam2/sam2.1_hiera_t512.yaml"
    # DINO is fetched through torch.hub on first use.
    dino_repo: str = "facebookresearch/dino:main"
    dino_model: str = "dino_vitb8"


@dataclass
class RegionMasksConfig:
    """MedSAM2's automatic mask generator splits the image into regions; CorrCLIP then lets a
    patch attend only to patches of its own region ("scope reconstruction")."""

    # A grid of points_per_side x points_per_side prompts covers the image.
    points_per_side: int = 16
    pred_iou_thresh: float = 0.5
    stability_score_thresh: float = 0.5
    box_nms_thresh: float = 0.5


@dataclass
class SaliencyConfig:
    """How the lesion probability map (saliency) is computed with CorrCLIP + BMC-CLIP."""

    # The image is rescaled so its short side is ``short_side`` (long side capped).
    short_side: int = 336
    max_long_side: int = 2048
    # Sliding-window inference over the rescaled image.
    window_size: int = 336
    window_stride: int = 112
    # Crops are zero-padded to a multiple of this (lcm of the CLIP and DINO patch sizes).
    pad_multiple: int = 56
    # Temperature applied to the cosine logits before the softmax over classes.
    logit_scale: float = 40.0
    # Prompt templates used to embed each class phrase (see models/text_encoder.py).
    template_set: str = "biomedical"
    region_masks: RegionMasksConfig = field(default_factory=RegionMasksConfig)


@dataclass
class CorrCLIPConfig:
    """Hyper-parameters of the CorrCLIP feature reconstruction (paper defaults)."""

    # Softmax temperature on DINO patch similarities.
    tau: float = 4.0
    # Weights of the spatial and semantic refinement branches.
    alpha_spatial: float = 1.0
    beta_semantic: float = 0.5
    # DBSCAN eps for merging look-alike regions (on 1 - cosine similarity).
    merge_eps: float = 0.2
    # Number of intermediate layers averaged by the spatial branch; null = paper default.
    num_spatial_layers: int | None = None


@dataclass
class PromptingConfig:
    """Turns the saliency map into point prompts for MedSAM2 (the highest, well-separated peaks)."""

    num_points: int = 5
    # Half-side of the square blanked around each picked peak, as a fraction of min(H, W).
    suppression_frac: float = 0.10


@dataclass
class SelectionConfig:
    """Chooses the output among all MedSAM2 candidates (num_points x 3 masks).

    The candidate with the highest predicted IoU is kept. For a target made of separate
    parts (both lungs) set ``num_objects: 2``: the next most confident mask that does not
    overlap the first is added, and the output is their union.
    """

    num_objects: int = 1
    # A further object is accepted only if at most this fraction of it overlaps the
    # objects already chosen (i.e. it must be a different, essentially disjoint structure).
    max_overlap: float = 0.1


@dataclass
class DatasetConfig:
    """A dataset is a folder of images plus (optionally) a folder of binary masks."""

    name: str = "dataset"
    root: str = "data/dataset"
    images_dir: str = "images"
    masks_dir: str | None = "masks"
    image_extensions: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])
    # Mask file name for an image named ``<stem>.<ext>``; ``{stem}`` is substituted.
    mask_pattern: str = "{stem}_mask.png"
    # A mask pixel is foreground when its value is greater than this.
    mask_threshold: int = 0
    # Human-readable name of the target class (used in reports only).
    target_name: str = "lesion"
    # Text describing each class. Synonyms of one class are averaged into one embedding.
    background_prompts: list[str] = field(default_factory=lambda: ["background"])
    target_prompts: list[str] = field(default_factory=lambda: ["lesion"])


@dataclass
class RuntimeConfig:
    """Execution settings that never change the results."""

    device: str = "auto"
    output_dir: str = "outputs"
    # Evaluate only the first N images (smoke tests). Never report numbers from a subset.
    limit: int | None = None
    save_masks: bool = False
    save_overlays: bool = False
    num_workers: int = 2


@dataclass
class EvaluationConfig:
    """How predictions are scored (never changes the predictions themselves)."""

    # Per-image Normalized Surface Dice at this tolerance in pixels (null = not computed).
    # MedCLIP-SAMv2 / MedSAM report NSD at 2 px.
    nsd_tolerance: float | None = None
    # If a mask's size differs from its image, resize the prediction to the mask with
    # nearest-neighbour before scoring (the MedCLIP-SAMv2 protocol). Otherwise sizes must match.
    resize_prediction_to_mask: bool = False


@dataclass
class Config:
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    corrclip: CorrCLIPConfig = field(default_factory=CorrCLIPConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path, _seen: set | None = None) -> dict[str, Any]:
    """Read a YAML file, resolving its ``inherit:`` parents (relative to the file)."""
    path = path.resolve()
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise ValueError(f"Circular 'inherit' chain at {path}")
    _seen.add(path)

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parents = data.pop("inherit", []) or []
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(merged, _read_yaml(path.parent / parent, _seen))
    return _deep_merge(merged, data)


def _parse_override(item: str) -> dict[str, Any]:
    """``"a.b.c=value"`` -> ``{"a": {"b": {"c": value}}}``; the value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"Override must look like key.sub=value, got {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    node: dict[str, Any] = {}
    cursor = node
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return node


def _build_dataclass(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    """Recursively build ``cls`` from ``data``, rejecting unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config key(s) under '{prefix or 'root'}': {sorted(unknown)}")
    kwargs = {}
    defaults = cls()
    for name in known & set(data):
        default_value = getattr(defaults, name)
        if dataclasses.is_dataclass(default_value):
            kwargs[name] = _build_dataclass(type(default_value), data[name] or {}, f"{prefix}{name}.")
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def load_config(
    paths: Sequence[PathLike] = (),
    overrides: Sequence[str] = (),
) -> Config:
    """Build a :class:`Config` from YAML files and ``key=value`` overrides.

    Args:
        paths: YAML files merged left to right (usually one dataset file).
        overrides: dotted ``key=value`` strings applied last.
    """
    merged: dict[str, Any] = {}
    for p in paths:
        merged = _deep_merge(merged, _read_yaml(Path(p)))
    for item in overrides:
        merged = _deep_merge(merged, _parse_override(item))
    return _build_dataclass(Config, merged)
