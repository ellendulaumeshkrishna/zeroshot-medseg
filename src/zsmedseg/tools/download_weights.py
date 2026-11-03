"""Download the two checkpoints from the Hugging Face Hub.

* BMC-CLIP ViT-L/14 (concept-filtered): ``BIOMEDICA/BMC_CLIP_CF`` / ``BMC_CLIP_CF.pt`` (5.1 GB)
* MedSAM2 (Hiera-tiny, 512 px):        ``wanglab/MedSAM2`` / ``MedSAM2_latest.pt`` (156 MB)

DINO ViT-B/8 is fetched automatically by ``torch.hub`` on first use.
The MedSAM2 file used for the reported results has SHA-256
``c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINTS = {
    "bmc_clip": ("BIOMEDICA/BMC_CLIP_CF", "BMC_CLIP_CF.pt"),
    "medsam2": ("wanglab/MedSAM2", "MedSAM2_latest.pt"),
}


def download_weights(out_dir: str | Path = "weights", which: tuple = ("bmc_clip", "medsam2")) -> dict:
    """Download the requested checkpoints into ``out_dir``; returns ``{name: path}``."""
    from huggingface_hub import hf_hub_download

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in which:
        repo_id, filename = CHECKPOINTS[name]
        target = out / filename
        if target.is_file():
            logger.info("%s already present at %s", filename, target)
        else:
            logger.info("Downloading %s from %s ...", filename, repo_id)
            # local_dir writes the file directly into out/ (no second copy in the HF cache).
            target = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(out)))
        paths[name] = target
    return paths
