"""Loading BMC-CLIP, the biomedical CLIP model that links image patches to words."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _extract_state_dict(checkpoint: Any) -> dict:
    """Return the model weights from a raw or training checkpoint.

    BMC-CLIP is distributed as an open_clip *training* checkpoint (weights under
    ``state_dict`` with a ``module.`` prefix from multi-GPU training, plus optimizer state).
    Plain state dicts are accepted as well.
    """
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}


def load_bmc_clip(
    checkpoint_path: str | Path,
    arch: str = "ViT-L-14",
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, Any]:
    """Build an open_clip model of architecture ``arch`` and load BMC-CLIP weights into it.

    Returns:
        ``(model, tokenizer)``; the model is frozen and in eval mode.
    """
    import open_clip  # imported lazily so that light-weight tools work without it

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"BMC-CLIP checkpoint not found at {checkpoint_path}. "
            "Run `python -m zsmedseg download-weights` or set weights.bmc_clip."
        )

    model = open_clip.create_model(arch, pretrained=None, precision="fp32")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(_extract_state_dict(checkpoint), strict=False)
    del checkpoint
    if missing or unexpected:
        raise RuntimeError(
            f"BMC-CLIP checkpoint does not match '{arch}': "
            f"{len(missing)} missing keys (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]})."
        )
    logger.info("Loaded BMC-CLIP (%s) from %s", arch, checkpoint_path)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, open_clip.get_tokenizer(arch)
