"""Pick the output mask among all MedSAM2 candidates."""

from __future__ import annotations

import numpy as np

from ..models.medsam2 import MaskCandidate


class MaskSelector:
    """Keep the candidate with the highest predicted IoU (MedSAM2's own confidence).

    For targets made of separate parts (both lungs) use ``num_objects=2``: candidates are
    visited from the highest predicted IoU down and a further one is added if it comes from
    a point not used yet and at most ``max_overlap`` of its area lies inside the masks
    already kept. The result's mask is the union and its score the mean of the kept scores.
    Ties go to the earliest candidate (first point, first mask).
    """

    def __init__(self, num_objects: int = 1, max_overlap: float = 0.1) -> None:
        if num_objects < 1:
            raise ValueError("num_objects must be >= 1")
        self.num_objects = num_objects
        self.max_overlap = max_overlap

    def __call__(self, candidates: list[MaskCandidate]) -> MaskCandidate | None:
        return self.merge(self.choose(candidates))

    def choose(self, candidates: list[MaskCandidate]) -> list[MaskCandidate]:
        """The kept candidates, most confident first (one, or up to ``num_objects``)."""
        if not candidates:
            return []
        # Stable sort keeps the earliest candidate first among equal scores.
        order = sorted(range(len(candidates)), key=lambda i: -candidates[i].score)
        kept = [candidates[order[0]]]
        union = kept[0].mask.copy()
        for i in order[1:]:
            if len(kept) == self.num_objects:
                break
            cand = candidates[i]
            area = int(cand.mask.sum())
            if area == 0 or cand.prompt_index in {k.prompt_index for k in kept}:
                continue
            if np.logical_and(cand.mask, union).sum() / area <= self.max_overlap:
                kept.append(cand)
                union |= cand.mask
        return kept

    @staticmethod
    def merge(kept: list[MaskCandidate]) -> MaskCandidate | None:
        """One candidate whose mask is the union of ``kept`` and whose score is their mean."""
        if not kept:
            return None
        union = kept[0].mask.copy()
        for cand in kept[1:]:
            union |= cand.mask
        first = kept[0]
        return MaskCandidate(
            mask=union,
            score=float(np.mean([k.score for k in kept])),
            low_res_logits=first.low_res_logits,
            prompt_index=first.prompt_index,
            mask_index=first.mask_index,
        )
