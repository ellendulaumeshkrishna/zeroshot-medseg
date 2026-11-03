"""Demo website: upload a medical image, describe the target in words, see every step.

    python -m zsmedseg serve            # then open http://127.0.0.1:8000

The models are loaded once, in a background thread, when the server starts. Each request
re-embeds the text prompts (cheap) and runs the same pipeline as ``evaluate``.

Endpoints
---------
GET  /               the page (static/index.html)
GET  /api/status     {"ready": bool, "device": str, "error": str | null}
GET  /api/presets    the datasets we evaluated on: prompts, label, example image
POST /api/segment    multipart form: image, [mask], target, background, num_objects
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config, load_config
from ..pipeline import ZeroShotSegmenter
from ..selection import MaskSelector
from ..utils.visualization import colorize_saliency, overlay_mask

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_SIDE = 1024  # larger uploads are shrunk (as prepare-data does for ISIC) to fit GPU memory
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

# The datasets of the project, in the order shown on the page. ``example`` is a test image
# whose Dice equals the median Dice of its dataset (not hand-picked).
PRESETS = [
    {"key": "busi", "label": "Breast lesion, ultrasound (BUSI)", "example": True},
    {"key": "udiat", "label": "Breast tumour, ultrasound (UDIAT)", "example": True},
    {"key": "kvasir", "label": "Polyp, colonoscopy (Kvasir-SEG)", "example": True},
    {"key": "isic2018", "label": "Skin lesion, dermoscopy (ISIC 2018)", "example": True},
    {"key": "refuge2", "label": "Optic disc, fundus photo (REFUGE2)", "example": True},
    {"key": "lung_xray", "label": "Both lungs, chest X-ray", "example": True},
    {"key": "brain_mri", "label": "Brain tumour, MRI", "example": False},
    {"key": "lung_ct", "label": "Both lungs, chest CT", "example": False},
    {"key": "camus", "label": "Left ventricle, echocardiography (CAMUS)", "example": False},
]


def _configs_dir() -> Path:
    """configs/datasets of the repository (works from the repo root or an editable install)."""
    for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
        if (root / "configs" / "datasets").is_dir():
            return root / "configs" / "datasets"
    raise FileNotFoundError("configs/datasets not found; start the server from the repository root")


def load_presets() -> list[dict]:
    presets = []
    for p in PRESETS:
        cfg = load_config([_configs_dir() / f"{p['key']}.yaml"])
        presets.append(
            {
                **p,
                "target": cfg.dataset.target_prompts,
                "background": cfg.dataset.background_prompts,
                "num_objects": cfg.selection.num_objects,
                "target_name": cfg.dataset.target_name,
                "example": f"/static/examples/{p['key']}.png" if p["example"] else None,
                "example_mask": f"/static/examples/{p['key']}_mask.png" if p["example"] else None,
            }
        )
    return presets


# ------------------------------------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------------------------------------


def _decode(data: bytes, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if image is None:
        raise ValueError("The file is not an image OpenCV can read (use PNG, JPEG, BMP or TIFF).")
    return image


def _to_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _to_jpg(image: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _thumb(image: np.ndarray, side: int = 220) -> np.ndarray:
    scale = side / max(image.shape[:2])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image


def _draw_points(image: np.ndarray, points, kept_points: set[int]) -> np.ndarray:
    """Numbered prompt points; the point(s) whose mask was kept are drawn larger, in gold."""
    out = image.copy()
    radius = max(5, max(image.shape[:2]) // 70)
    thickness = max(1, radius // 4)
    for i, (x, y) in enumerate(points):
        kept = i in kept_points
        r = int(radius * (1.35 if kept else 1))
        cv2.circle(out, (x, y), r + thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), r, (0, 190, 255) if kept else (40, 40, 40), -1, cv2.LINE_AA)
        scale = r / 14
        (tw, th), _ = cv2.getTextSize(str(i + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(out, str(i + 1), (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0) if kept else (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _contour(image: np.ndarray, mask: np.ndarray, color, width: int) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, width, cv2.LINE_AA)
    return out


def _parse_phrases(text: str) -> list[str]:
    phrases = [p.strip() for p in text.replace("\n", ",").split(",")]
    return [p for p in phrases if p]


# ------------------------------------------------------------------------------------------
# Model holder
# ------------------------------------------------------------------------------------------


class ModelHolder:
    """Loads the pipeline once (in the background) and serialises GPU use."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.segmenter: ZeroShotSegmenter | None = None
        self.error: str | None = None
        self.device = "loading"
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            t0 = time.time()
            self.segmenter = ZeroShotSegmenter.from_config(self.cfg)
            self.device = str(self.segmenter.saliency_estimator.device)
            logger.info("Models ready on %s after %.0f s. Open the page in your browser.", self.device,
                        time.time() - t0)
        except Exception as err:  # shown on the page instead of a silent crash
            logger.exception("Could not load the models")
            self.error = f"{type(err).__name__}: {err}"

    def segment(self, image: np.ndarray, target: list[str], background: list[str], num_objects: int):
        if self.segmenter is None:
            raise RuntimeError(self.error or "The models are still loading, try again in a moment.")
        with self.lock:
            self.segmenter.set_prompts(background, target)
            self.segmenter.selector = MaskSelector(num_objects, self.cfg.selection.max_overlap)
            if self.device.startswith("cuda"):
                import torch

                torch.cuda.synchronize()
            t0 = time.time()
            result = self.segmenter.segment(image)
            return result, time.time() - t0


def build_response(image: np.ndarray, result, seconds: float, gt: np.ndarray | None, resized_from) -> dict:
    """Everything the page shows, as data URLs and numbers."""
    height, width = image.shape[:2]
    line = max(2, max(height, width) // 250)
    kept_ids = {(k.prompt_index, k.mask_index) for k in result.kept}

    heat = cv2.addWeighted(image, 0.45, colorize_saliency(result.saliency), 0.55, 0)
    heat = _draw_points(heat, result.points, {k.prompt_index for k in result.kept})

    final = overlay_mask(image, result.mask, color=(60, 60, 230), alpha=0.42)
    metrics: dict = {"predicted_iou": round(result.score, 3), "seconds": round(seconds, 1)}
    if gt is not None:
        # Scored at the mask's own resolution (the prediction is resized to it, as in the
        # MedCLIP-SAMv2 protocol); drawn at the image's resolution.
        pred = result.mask
        if gt.shape != pred.shape:
            pred = cv2.resize(pred.astype(np.uint8), gt.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        shown_gt = cv2.resize(gt.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        final_gt = _contour(final, shown_gt, (70, 200, 70), line)
        inter = int(np.logical_and(pred, gt).sum())
        total = int(pred.sum() + gt.sum())
        union = total - inter
        metrics["dice"] = round(100 * 2 * inter / total, 2) if total else 100.0
        metrics["iou"] = round(100 * inter / union, 2) if union else 100.0
    else:
        final_gt = None

    candidates = []
    small = _thumb(image)
    for c in result.candidates:
        small_mask = cv2.resize(c.mask.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        tile = overlay_mask(small, small_mask, color=(60, 60, 230))
        candidates.append({
            "image": _to_jpg(tile, 85),
            "score": round(c.score, 3),
            "point": c.prompt_index + 1,
            "mask": c.mask_index + 1,
            "kept": (c.prompt_index, c.mask_index) in kept_ids,
        })

    return {
        "size": [width, height],
        "resized_from": resized_from,
        "points": [[int(x), int(y)] for x, y in result.points],
        "metrics": metrics,
        "images": {
            "input": _to_jpg(image, 92),
            "saliency": _to_jpg(heat, 90),
            "final": _to_jpg(final, 92),
            "final_with_gt": _to_jpg(final_gt, 92) if final_gt is not None else None,
            "mask": _to_png(result.mask.astype(np.uint8) * 255),
        },
        "candidates": candidates,
        "num_kept": len(result.kept),
    }


# ------------------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------------------


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="zsmedseg demo", docs_url=None, redoc_url=None)
    models = ModelHolder(cfg)
    presets = load_presets()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {"ready": models.segmenter is not None, "device": models.device, "error": models.error}

    @app.get("/api/presets")
    def get_presets():
        return presets

    @app.post("/api/segment")
    async def segment(
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None),
        target: str = Form(...),
        background: str = Form("background"),
        num_objects: int = Form(1),
    ):
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 30 MB.")
        target_phrases, background_phrases = _parse_phrases(target), _parse_phrases(background)
        if not target_phrases:
            raise HTTPException(400, "Describe the target in at least one word.")
        if num_objects not in (1, 2):
            raise HTTPException(400, "num_objects must be 1 or 2.")
        try:
            bgr = _decode(data, cv2.IMREAD_COLOR)
            gt = None
            if mask is not None and mask.filename:
                raw = _decode(await mask.read(), cv2.IMREAD_GRAYSCALE)
                gt = raw > (127 if raw.max() > 1 else 0)  # 0/255 or 0/1 masks
        except ValueError as err:
            raise HTTPException(400, str(err)) from err

        resized_from = None
        scale = MAX_SIDE / max(bgr.shape[:2])
        if scale < 1:
            resized_from = [bgr.shape[1], bgr.shape[0]]
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        try:
            result, seconds = models.segment(bgr, target_phrases, background_phrases or ["background"], num_objects)
        except RuntimeError as err:
            raise HTTPException(503, str(err)) from err
        logger.info("Segmented %s (%dx%d) for %s in %.1f s", image.filename, bgr.shape[1], bgr.shape[0],
                    target_phrases, seconds)
        return build_response(bgr, result, seconds, gt, resized_from)

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, overrides: list[str] | None = None) -> None:
    cfg = load_config([], overrides or [])
    # On Windows, asyncio logs a harmless ConnectionResetError whenever the browser closes a
    # connection; hide it so real problems stay visible.
    logging.getLogger("asyncio").addFilter(
        lambda record: not (record.exc_info and isinstance(record.exc_info[1], ConnectionResetError))
    )
    logger.info("Starting the demo at http://%s:%d  (models load in the background, about a minute)", host, port)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
