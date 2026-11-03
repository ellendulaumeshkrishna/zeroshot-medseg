"""zsmedseg - zero-shot, training-free, text-prompted medical image segmentation.

BMC-CLIP (made dense by CorrCLIP) finds *where* the described target is; MedSAM2 draws
*its exact outline*. Nothing is trained: the only input besides the image is text.

Quick use::

    from zsmedseg import ZeroShotSegmenter, load_config
    cfg = load_config(["configs/datasets/busi.yaml"])
    segmenter = ZeroShotSegmenter.from_config(cfg)
    result = segmenter.segment(image_bgr)      # result.mask, result.saliency, ...
"""

from .config import Config, load_config

__version__ = "1.0.0"
__all__ = ["Config", "ZeroShotSegmenter", "SegmentationResult", "load_config", "__version__"]


def __getattr__(name: str):
    # Import the heavy pipeline (torch, models) lazily so `import zsmedseg` stays light.
    if name in ("ZeroShotSegmenter", "SegmentationResult"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'zsmedseg' has no attribute {name!r}")
