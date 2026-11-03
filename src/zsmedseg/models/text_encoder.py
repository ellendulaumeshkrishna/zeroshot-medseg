"""Turning class descriptions into embeddings ("what are we looking for?").

Each class is described by one or more phrases (synonyms). Every phrase is inserted into a
set of prompt templates ("an ultrasound image of {}."), each sentence is embedded by CLIP's
text encoder, and the embeddings are averaged twice: over templates, then over synonyms.
The result is exactly one unit vector per class.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

# Template sets. Add a new entry here to make it selectable via `saliency.template_set`.
TEMPLATE_SETS: dict[str, list[Callable[[str], str]]] = {
    "biomedical": [
        lambda c: f"a medical image showing {c}.",
        lambda c: f"an ultrasound image of {c}.",
        lambda c: f"a biomedical image of {c}.",
        lambda c: f"a clinical image showing {c}.",
        lambda c: f"a radiological image of {c}.",
        lambda c: f"a histological image of {c}.",
        lambda c: f"a microscopy image of {c}.",
        lambda c: f"an MRI scan showing {c}.",
        lambda c: f"a CT scan of {c}.",
        lambda c: f"an X-ray showing {c}.",
        lambda c: f"{c} in a medical scan.",
        lambda c: f"a diagnosis image of {c}.",
    ],
    # Uses each phrase verbatim, for prompts that are already full sentences.
    "identity": [lambda c: c],
}


@torch.inference_mode()
def encode_classes(
    clip_model: torch.nn.Module,
    tokenizer: Callable,
    class_prompts: Sequence[Sequence[str]],
    template_set: str = "biomedical",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Embed every class.

    Args:
        clip_model: an open_clip model.
        tokenizer: the matching open_clip tokenizer.
        class_prompts: ``class_prompts[k]`` lists the synonyms describing class ``k``.
            By convention class 0 is the background and class 1 is the target.
        template_set: key of :data:`TEMPLATE_SETS`.

    Returns:
        ``K x D`` tensor of L2-normalised class embeddings.
    """
    if template_set not in TEMPLATE_SETS:
        raise KeyError(f"Unknown template set '{template_set}'. Available: {sorted(TEMPLATE_SETS)}")
    templates = TEMPLATE_SETS[template_set]

    class_embeddings = []
    for synonyms in class_prompts:
        if not synonyms:
            raise ValueError("Every class needs at least one prompt phrase")
        phrase_embeddings = []
        for phrase in synonyms:
            tokens = tokenizer([template(phrase.strip()) for template in templates]).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0)  # average over templates
            phrase_embeddings.append(emb / emb.norm())
        pooled = torch.stack(phrase_embeddings).mean(dim=0)  # average over synonyms
        class_embeddings.append(pooled / pooled.norm())
    return torch.stack(class_embeddings).float()
