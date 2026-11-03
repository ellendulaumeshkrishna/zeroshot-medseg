"""DINO patch features: "which parts of the image look alike?".

CorrCLIP uses self-supervised DINO to decide which patches should share information. The
feature for each patch is ``normalize(q + k)`` taken from the query/key projection of DINO's
last attention block (captured with a forward hook).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..data.preprocessing import clip_to_imagenet


class DinoFeatureExtractor(nn.Module):
    """Wraps a torch.hub DINO ViT and exposes per-patch similarity features."""

    def __init__(self, repo: str = "facebookresearch/dino:main", model: str = "dino_vitb8") -> None:
        super().__init__()
        self.backbone = torch.hub.load(repo, model)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.patch_size: int = self.backbone.patch_embed.patch_size
        self._qkv: torch.Tensor | None = None
        self.backbone.blocks[-1].attn.qkv.register_forward_hook(self._capture_qkv)

    def _capture_qkv(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self._qkv = output

    @torch.inference_mode()
    def forward(self, clip_images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Compute patch features.

        Args:
            clip_images: ``B x 3 x H x W`` images normalised with CLIP statistics
                (H and W divisible by the DINO patch size).

        Returns:
            ``(features, (h, w))`` where ``features`` is ``B x (h*w) x C`` and L2-normalised.
        """
        images = clip_to_imagenet(clip_images.float())
        self._qkv = None
        tokens = self.backbone.get_intermediate_layers(images, n=1)[-1]
        batch, num_tokens = tokens.shape[:2]
        qkv = self._qkv.reshape(batch, num_tokens, 3, -1).permute(2, 0, 1, 3)
        feats = (qkv[0] + qkv[1])[:, 1:]  # drop the CLS token
        grid = (images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size)
        return F.normalize(feats, dim=-1), grid
