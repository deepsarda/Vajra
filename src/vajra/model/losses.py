"""Training helpers: focal loss for rare events, and switching sensors off at random."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017). Lightning is rare, so easy "no lightning" pixels
    are down-weighted and the model spends its effort on the storms."""
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def drop_modalities(
    available: torch.Tensor, p: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Randomly switch off whole sensor groups during training, keeping at least one on per sample.

    This teaches the model to keep forecasting when a radar or the satellite feed is down.
    """
    draws = torch.rand(available.shape, generator=generator, device=available.device)
    keep = (draws >= p).to(available.dtype)
    out = available * keep
    empty = torch.nonzero(out.sum(1) == 0).squeeze(1)
    if empty.numel():
        weights = available[empty].float() + 1e-6
        pick = torch.multinomial(weights, 1, generator=generator).squeeze(1)
        out[empty, pick] = 1
    return out
