"""Frozen pretrained models: BMC-CLIP, DINO, CorrCLIP's dense encoder and MedSAM2."""

from .clip_backbone import load_bmc_clip
from .corrclip import CorrCLIPVisualEncoder
from .dino import DinoFeatureExtractor
from .medsam2 import AutomaticRegionGenerator, MaskCandidate, PointPromptSegmenter, build_medsam2
from .text_encoder import TEMPLATE_SETS, encode_classes

__all__ = [
    "AutomaticRegionGenerator",
    "CorrCLIPVisualEncoder",
    "DinoFeatureExtractor",
    "MaskCandidate",
    "PointPromptSegmenter",
    "TEMPLATE_SETS",
    "build_medsam2",
    "encode_classes",
    "load_bmc_clip",
]
