"""Text-guided saliency (CorrCLIP + BMC-CLIP) and the MedSAM2 region masks it uses."""

from .estimator import SaliencyEstimator, SaliencyResult, centred_padding, sliding_windows
from .regions import RegionGenerator

__all__ = ["RegionGenerator", "SaliencyEstimator", "SaliencyResult", "centred_padding", "sliding_windows"]
