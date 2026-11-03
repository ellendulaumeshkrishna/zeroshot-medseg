"""Small drawing helpers for qualitative inspection of predictions."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Color = tuple[int, int, int]  # BGR


def overlay_mask(
    image_bgr: np.ndarray, mask: np.ndarray, color: Color = (40, 40, 220), alpha: float = 0.45
) -> np.ndarray:
    """Tint ``mask`` on the image and draw its outline."""
    out = image_bgr.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, max(1, image_bgr.shape[1] // 300))
    return out


def colorize_saliency(saliency: np.ndarray) -> np.ndarray:
    """Map a [0, 1] probability map to a BGR heat map (viridis)."""
    values = np.clip(saliency * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)


def draw_points(image_bgr: np.ndarray, points: Sequence[tuple[int, int]], color: Color = (0, 200, 255)) -> np.ndarray:
    """Draw numbered prompt points."""
    out = image_bgr.copy()
    radius = max(4, image_bgr.shape[1] // 90)
    for i, (x, y) in enumerate(points, start=1):
        cv2.circle(out, (int(x), int(y)), radius + 2, (255, 255, 255), -1)
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
        cv2.putText(
            out,
            str(i),
            (int(x) + radius + 2, int(y) - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def make_panel(
    image_bgr: np.ndarray,
    saliency: np.ndarray,
    points: Sequence[tuple[int, int]],
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> np.ndarray:
    """Side-by-side panel: image | saliency + points | prediction | ground truth (optional)."""
    tiles = [
        image_bgr,
        draw_points(cv2.addWeighted(image_bgr, 0.4, colorize_saliency(saliency), 0.6, 0), points),
        overlay_mask(image_bgr, prediction),
    ]
    if ground_truth is not None:
        if ground_truth.shape != image_bgr.shape[:2]:  # some datasets store masks at another size
            size = (image_bgr.shape[1], image_bgr.shape[0])
            ground_truth = cv2.resize(ground_truth.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        tiles.append(overlay_mask(image_bgr, ground_truth, color=(60, 180, 60)))
    return np.concatenate(tiles, axis=1)
