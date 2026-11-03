"""CorrCLIP dense image encoder: clean per-patch CLIP features.

Plain CLIP is trained on whole images, so its per-patch features are noisy. CorrCLIP
(Zhang et al., ICCV 2025, arXiv:2411.10086) rebuilds the *last* attention layer of CLIP's
vision transformer so that each patch averages the features of patches that (a) look alike
according to DINO and (b) lie in the same region:

* **Value reconstruction** - attention weights come from DINO patch similarity, not CLIP's
  own query/key products.
* **Scope reconstruction** - similarities between patches of different regions are masked
  out (the regions come from MedSAM2's automatic mask generator).
* **Mask merging** - regions whose mean DINO features are similar are merged (DBSCAN).
* **Feature refinement** - a *spatial branch* (the same reconstruction applied to the mean
  of intermediate layers) and a *semantic branch* (one summary token per region) are added.

This module re-implements that forward pass on top of an unmodified ``open_clip``
VisionTransformer. It follows the reference implementation operation by operation
(including its edge-case behaviour) so that results are reproduced exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Region id written into zero-padded borders of a crop (matches the reference code).
PAD_REGION_ID = 10000
# Merged regions are relabelled to MERGED_REGION_BASE + cluster index.
MERGED_REGION_BASE = 10000


def default_num_spatial_layers(num_layers: int) -> int:
    """Number of intermediate layers averaged by the spatial branch (paper defaults)."""
    return {12: 5, 24: 12, 32: 28}.get(num_layers, num_layers - 4)


def _block_forward(block: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Run an open_clip ``ResidualAttentionBlock`` on ``L x N x D`` tokens.

    ``attn_mask`` is boolean with ``True`` meaning "may NOT attend". It is passed to the
    attention layer unchanged; open_clip's own ``attention()`` would cast it to float, which
    turns a blocking mask into a small additive bias.
    """
    h = block.ln_1(x)
    attn_out = block.attn(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
    x = x + block.ls_1(attn_out)
    return x + block.ls_2(block.mlp(block.ln_2(x)))


class CorrCLIPVisualEncoder(nn.Module):
    """Dense CorrCLIP forward over a frozen open_clip ``VisionTransformer``."""

    def __init__(
        self,
        visual: nn.Module,
        tau: float = 4.0,
        alpha_spatial: float = 1.0,
        beta_semantic: float = 0.5,
        merge_eps: float = 0.2,
        num_spatial_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.patch_size: tuple[int, int] = tuple(visual.conv1.kernel_size)  # type: ignore[assignment]
        self.tau = tau
        self.alpha_spatial = alpha_spatial
        self.beta_semantic = beta_semantic
        self.merge_eps = merge_eps
        num_layers = len(visual.transformer.resblocks)
        self.num_spatial_layers = num_spatial_layers or default_num_spatial_layers(num_layers)

    # ---------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------

    def _positional_embedding(self, num_tokens: int, height: int, width: int) -> torch.Tensor:
        """Bicubically resize the positional embedding to the crop's patch grid."""
        pos = self.visual.positional_embedding
        n_orig = pos.shape[0] - 1
        if num_tokens == pos.shape[0]:
            return pos
        side = int(math.sqrt(n_orig))
        rows = height // self.patch_size[0] + 0.1
        cols = width // self.patch_size[1] + 0.1
        grid = pos[1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, scale_factor=(rows / side, cols / side), mode="bicubic")
        assert int(rows) == grid.shape[-2] and int(cols) == grid.shape[-1]
        grid = grid.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
        return torch.cat([pos[[0]].unsqueeze(0), grid], dim=1)

    def _merge_regions(self, regions: torch.Tensor, dino_feats: torch.Tensor) -> torch.Tensor:
        """Merge regions with similar mean DINO features (in place on a copy).

        Args:
            regions: ``B x h x w`` integer region ids at the DINO grid resolution.
            dino_feats: ``B x (h*w) x C`` normalised DINO features.
        """
        from sklearn.cluster import DBSCAN

        regions = regions.clone()
        batch, h, w = regions.shape
        feats = dino_feats.reshape(batch, h, w, -1)
        for b in range(batch):
            values = torch.unique(regions[b])
            if values.shape[0] <= 2:  # nothing to merge with one real region
                continue
            region_masks = [regions[b] == v for v in values]
            means = torch.stack([feats[b][m].mean(dim=0) for m in region_masks])
            means = F.normalize(means, dim=1)[1:]  # region 0 (smallest id) = unsegmented
            sim = means @ means.T
            dbscan = DBSCAN(metric="precomputed", eps=self.merge_eps, min_samples=1)
            labels = dbscan.fit_predict((sim.max() - sim).cpu().numpy())
            for idx, label in enumerate(labels.tolist(), start=1):
                if label != -1:
                    regions[b][region_masks[idx]] = label + MERGED_REGION_BASE
        return regions

    def _scope_attention(self, similarity: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        """Mask DINO similarities to within-region pairs and turn them into weights.

        Patches of the same region may attend to each other. Patches in the leftover area
        (the smallest region id, "unsegmented") may attend to any patch whose similarity is
        above the mean.
        """
        similarity = similarity.clone()
        for b in range(similarity.shape[0]):
            region_masks = [(regions[b] == v).reshape(-1) for v in torch.unique(regions[b])]
            allowed = torch.zeros_like(similarity[b], dtype=torch.bool)
            for m in region_masks[1:]:
                allowed |= m[:, None] & m[None, :]
            rest = region_masks[0]
            allowed |= (rest[None, :] | rest[:, None]) & (similarity[b] > similarity[b].mean())
            similarity[b][~allowed] = float("-inf")
        return (similarity * self.tau).softmax(dim=-1)

    @staticmethod
    def _semantic_tokens_mask(
        regions: torch.Tensor, num_regular_tokens: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention mask that lets region token ``i`` see only CLS + patches of region ``i``.

        Returns:
            ``(attn_mask, region_masks)``; ``attn_mask`` is boolean (``True`` = blocked) over
            ``[region tokens, CLS, patches]`` and ``region_masks`` is ``M x h x w``.
        """
        region_masks = torch.stack([regions == v for v in torch.unique(regions)], dim=0)
        m = region_masks.shape[0]
        total = m + num_regular_tokens
        allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
        allowed[m:, m:] = True  # ordinary tokens attend to each other as usual
        allowed[:m, m] = True  # region tokens see CLS
        allowed[:m, m + 1 :] = region_masks.reshape(m, -1)  # ... and their own patches
        return ~allowed, region_masks

    # ---------------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        dino_feats: torch.Tensor,
        feat_shape: tuple[int, int],
        regions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-patch CLIP embeddings on the DINO grid.

        Args:
            image: ``1 x 3 x H x W`` CLIP-normalised crop.
            dino_feats: ``1 x (h*w) x C`` normalised DINO features of the same crop.
            feat_shape: DINO grid ``(h, w)``; the output is produced at this resolution.
            regions: ``1 x 1 x H x W`` integer region ids for the crop.

        Returns:
            ``1 x (h*w) x D`` embeddings in CLIP's joint image-text space.
        """
        visual = self.visual
        batch, _, height, width = image.shape
        if batch != 1:
            raise ValueError("CorrCLIPVisualEncoder processes one crop at a time")
        token_size = (height // self.patch_size[0], width // self.patch_size[1])

        # --- Standard ViT stem ---------------------------------------------------------
        x = visual.conv1(image)
        x = x.reshape(batch, x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.view(1, 1, -1).expand(batch, -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + self._positional_embedding(x.shape[1], height, width).to(x.dtype)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (tokens, batch, dim)

        # --- Value + scope reconstruction: attention weights from DINO ----------------
        similarity = torch.einsum("bmc,bnc->bmn", dino_feats, dino_feats)
        grid_regions = F.interpolate(regions.float(), size=feat_shape, mode="nearest").int().squeeze(1)
        grid_regions = self._merge_regions(grid_regions, dino_feats)
        attn_weights = self._scope_attention(similarity, grid_regions)

        # --- Semantic branch: one extra token per region -------------------------------
        token_regions = (
            F.interpolate(grid_regions.float().unsqueeze(1), size=token_size, mode="nearest").int().squeeze(1)[0]
        )
        attn_mask, region_masks = self._semantic_tokens_mask(token_regions, x.shape[0], x.device)
        num_region_tokens = region_masks.shape[0]
        x = torch.cat([x[:1].repeat(num_region_tokens, 1, 1), x], dim=0)

        # --- All but the last block, collecting features for the spatial branch -------
        blocks = visual.transformer.resblocks
        num_layers = len(blocks)
        first_spatial = num_layers - 2 - self.num_spatial_layers
        spatial_feats = []
        for idx, block in enumerate(blocks[:-1]):
            x = _block_forward(block, x, attn_mask)
            if first_spatial <= idx <= num_layers - 3:
                spatial_feats.append(x[num_region_tokens:])

        # --- Last block: rebuilt attention (main + spatial), semantic tokens ----------
        last = blocks[-1]
        main_and_spatial = torch.cat([x[num_region_tokens:], torch.stack(spatial_feats).mean(0)], dim=1)
        dim = last.attn.embed_dim
        v_weight = last.attn.in_proj_weight.narrow(0, 2 * dim, dim)
        v_bias = last.attn.in_proj_bias.narrow(0, 2 * dim, dim)
        v = F.linear(last.ln_1(main_and_spatial), v_weight, v_bias)  # values only
        v = v.permute(1, 2, 0)[:, :, 1:].unflatten(dim=2, sizes=token_size)  # drop CLS
        v = F.interpolate(v, size=feat_shape, mode="bilinear", align_corners=False)
        v = v.flatten(2, 3).transpose(1, 2)
        weights = attn_weights.expand(batch * 2, -1, -1)
        main_and_spatial = last.attn.out_proj(torch.bmm(weights, v).transpose(0, 1).contiguous())

        region_tokens = _block_forward(last, x, attn_mask)[:num_region_tokens, 0]
        grid_masks = F.interpolate(
            region_masks.unsqueeze(1).to(region_tokens.dtype), size=feat_shape, mode="nearest"
        ).flatten(1, -1)
        semantic = (grid_masks.T @ region_tokens).unsqueeze(1)  # each patch gets its region token

        x = torch.cat([main_and_spatial, semantic], dim=1).permute(1, 0, 2)
        x = visual.ln_post(x)
        main, spatial, semantic = x.chunk(3, dim=0)
        x = main + spatial * self.alpha_spatial + semantic * self.beta_semantic
        return x @ visual.proj
