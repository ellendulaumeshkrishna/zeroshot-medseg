"""Segmentation metrics for one target class.

Two families are reported because papers use both:

* **Per-image** Dice / IoU, averaged over the images that contain the target. This is the
  convention of medical-segmentation papers and the headline number of this project.
* **Dataset-aggregate** IoU / Dice: intersections and unions are summed over *all* pixels
  of *all* images before dividing. Large targets weigh more, and images without a target
  contribute their false positives.

Optionally the per-image **Normalized Surface Dice (NSD)**: the share of both boundaries that
lies within ``tolerance`` pixels of the other boundary. It is computed exactly as in the
MedSAM / MedCLIP-SAMv2 evaluation code: DeepMind's ``surface-distance`` package applied to
the 2D masks as single-slice volumes (``mask[..., None]``) with spacing ``[1, 1, 1]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def normalized_surface_dice(prediction: np.ndarray, ground_truth: np.ndarray, tolerance: float) -> float:
    """Per-image NSD in percent (MedCLIP-SAMv2 protocol; needs ``surface-distance``)."""
    if not ground_truth.any():
        return 100.0 if not prediction.any() else 0.0
    try:
        import surface_distance
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "NSD needs DeepMind's surface-distance package: "
            "pip install git+https://github.com/google-deepmind/surface-distance.git"
        ) from err
    distances = surface_distance.compute_surface_distances(ground_truth[..., None], prediction[..., None], [1, 1, 1])
    return 100.0 * float(surface_distance.compute_surface_dice_at_tolerance(distances, tolerance))


@dataclass
class ImageRecord:
    name: str
    intersection: int
    pred_area: int
    gt_area: int
    num_pixels: int
    nsd: float | None = None  # percent; only when NSD is enabled

    @property
    def dice(self) -> float:
        denom = self.pred_area + self.gt_area
        return 100.0 * 2 * self.intersection / denom if denom else 100.0

    @property
    def iou(self) -> float:
        union = self.pred_area + self.gt_area - self.intersection
        return 100.0 * self.intersection / union if union else 100.0


class SegmentationMeter:
    """Accumulates per-image counts and summarises them.

    Args:
        nsd_tolerance: if set, also compute per-image NSD at this tolerance (pixels).
    """

    def __init__(self, nsd_tolerance: float | None = None) -> None:
        self.records: list[ImageRecord] = []
        self.nsd_tolerance = nsd_tolerance

    def update(self, name: str, prediction: np.ndarray, ground_truth: np.ndarray) -> ImageRecord:
        if prediction.shape != ground_truth.shape:
            raise ValueError(f"{name}: prediction {prediction.shape} vs mask {ground_truth.shape}")
        nsd = None
        if self.nsd_tolerance is not None:
            nsd = normalized_surface_dice(prediction, ground_truth, self.nsd_tolerance)
        record = ImageRecord(
            name=name,
            intersection=int(np.logical_and(prediction, ground_truth).sum()),
            pred_area=int(prediction.sum()),
            gt_area=int(ground_truth.sum()),
            num_pixels=int(prediction.size),
            nsd=nsd,
        )
        self.records.append(record)
        return record

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {}
        inter = np.array([r.intersection for r in self.records], dtype=np.float64)
        pred = np.array([r.pred_area for r in self.records], dtype=np.float64)
        gt = np.array([r.gt_area for r in self.records], dtype=np.float64)
        total = np.array([r.num_pixels for r in self.records], dtype=np.float64)

        def iou_dice(i: float, p: float, g: float) -> tuple:
            union = p + g - i
            return (100 * i / union if union else 0.0, 100 * 2 * i / (p + g) if p + g else 0.0)

        target_iou, target_dice = iou_dice(inter.sum(), pred.sum(), gt.sum())
        # Background = complement of both masks, as mmseg's 2-class IoUMetric computes it.
        bg_inter = (total - pred - gt + inter).sum()
        bg_iou, bg_dice = iou_dice(bg_inter, (total - pred).sum(), (total - gt).sum())

        has_target = gt > 0
        dice = np.array([r.dice for r in self.records])[has_target]
        iou = np.array([r.iou for r in self.records])[has_target]
        empty_with_mask = int(((~has_target) & (pred > 0)).sum())

        nsd_stats = {}
        if self.nsd_tolerance is not None:
            nsd = np.array([r.nsd for r in self.records], dtype=np.float64)[has_target]
            nsd_stats = {
                "nsd_tolerance": self.nsd_tolerance,
                "per_image_nsd_mean": float(nsd.mean()) if nsd.size else float("nan"),
            }

        return {
            **nsd_stats,
            "num_images": len(self.records),
            "num_target_images": int(has_target.sum()),
            "num_empty_images": int((~has_target).sum()),
            "target_iou": target_iou,
            "target_dice": target_dice,
            "background_iou": bg_iou,
            "background_dice": bg_dice,
            "mean_iou": (target_iou + bg_iou) / 2,
            "mean_dice": (target_dice + bg_dice) / 2,
            "per_image_dice_mean": float(dice.mean()) if dice.size else float("nan"),
            "per_image_dice_median": float(np.median(dice)) if dice.size else float("nan"),
            "per_image_iou_mean": float(iou.mean()) if iou.size else float("nan"),
            "empty_images_with_mask": empty_with_mask,
        }

    def rows(self) -> list[dict[str, float]]:
        """Per-image rows for a CSV report."""
        return [dict(asdict(r), dice=r.dice, iou=r.iou) for r in self.records]


def format_summary(summary: dict[str, float], target_name: str = "target") -> str:
    """Human-readable multi-line report."""
    if not summary:
        return "No images evaluated."
    lines = [
        f"Images: {summary['num_images']} ({summary['num_target_images']} with {target_name}, "
        f"{summary['num_empty_images']} without)",
        f"Dice (per image, mean over {target_name} images): {summary['per_image_dice_mean']:.2f}",
        f"IoU  (per image, mean over {target_name} images): {summary['per_image_iou_mean']:.2f}",
        f"{target_name} IoU / Dice (dataset aggregate): {summary['target_iou']:.2f} / {summary['target_dice']:.2f}",
    ]
    if "per_image_nsd_mean" in summary:
        lines.insert(
            3,
            f"NSD  (per image, tolerance {summary['nsd_tolerance']:g} px):        {summary['per_image_nsd_mean']:.2f}",
        )
    if summary["num_empty_images"]:
        lines.append(
            f"Images without {target_name} that received a mask: "
            f"{summary['empty_images_with_mask']} / {summary['num_empty_images']}"
        )
    return "\n".join(lines)
