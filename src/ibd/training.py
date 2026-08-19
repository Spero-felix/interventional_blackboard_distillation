"""Loss primitives for outcome, intervention, and Stage D alignment training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def response_token_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal next-token log probabilities and a response-only mask."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [B,T,V] and labels [B,T] must share B,T")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_scores = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_scores.masked_fill(~mask, 0.0), mask


def length_normalized_score(
    token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if token_log_probs.shape != response_mask.shape:
        raise ValueError("token log probabilities and mask must have equal shapes")
    lengths = response_mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("every sequence must contain at least one response token")
    return (token_log_probs * response_mask).sum(dim=-1) / lengths


def margin_alignment_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return torch.relu(margin - chosen_scores + rejected_scores).mean()


def stage_d_loss(
    *,
    chosen_sft: torch.Tensor,
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    replay_loss: torch.Tensor,
    margin: float,
    replay_weight: float,
) -> torch.Tensor:
    """Chosen SFT + length-normalized hinge + low-weight Stage B/C replay."""
    if replay_weight < 0:
        raise ValueError("replay_weight must be non-negative")
    return (
        chosen_sft
        + margin_alignment_loss(chosen_scores, rejected_scores, margin)
        + replay_weight * replay_loss
    )

