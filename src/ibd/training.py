"""Loss primitives for Stage A/B/C training."""

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
    selected_logits = shifted_logits.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_scores = selected_logits - torch.logsumexp(shifted_logits, dim=-1)
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


def contrastive_alignment_loss(
    slots: torch.Tensor,
    anchor_bank: torch.Tensor,
    positive_rows: torch.Tensor,
    *,
    temperature: float,
    cosine_weight: float,
) -> torch.Tensor:
    """Full-bank cosine classification plus direct positive cosine distance."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if cosine_weight < 0:
        raise ValueError("cosine_weight must be non-negative")
    if slots.ndim != 2 or anchor_bank.ndim != 2 or slots.shape[1] != anchor_bank.shape[1]:
        raise ValueError("slots and anchor bank must be compatible rank-two tensors")
    if positive_rows.shape != (slots.shape[0],):
        raise ValueError("positive_rows must provide one row per slot")
    bank = F.normalize(anchor_bank.detach().float(), dim=-1)
    normalized_slots = F.normalize(slots.float(), dim=-1)
    rows = positive_rows.to(device=slots.device, dtype=torch.long)
    logits = normalized_slots @ bank.to(slots.device).T / temperature
    classification = F.cross_entropy(logits, rows)
    positives = bank.to(slots.device)[rows]
    cosine_distance = (1.0 - (normalized_slots * positives).sum(dim=-1)).mean()
    return classification + cosine_weight * cosine_distance


def stage_b_loss(
    *,
    response_loss: torch.Tensor,
    state_slots: torch.Tensor,
    plan_slots: torch.Tensor,
    state_bank: torch.Tensor,
    plan_bank: torch.Tensor,
    positive_rows: torch.Tensor,
    temperature: float,
    cosine_weight: float,
    state_weight: float,
    plan_weight: float,
) -> torch.Tensor:
    if state_weight < 0 or plan_weight < 0:
        raise ValueError("slot alignment weights must be non-negative")
    state_alignment = contrastive_alignment_loss(
        state_slots,
        state_bank,
        positive_rows,
        temperature=temperature,
        cosine_weight=cosine_weight,
    )
    plan_alignment = contrastive_alignment_loss(
        plan_slots,
        plan_bank,
        positive_rows,
        temperature=temperature,
        cosine_weight=cosine_weight,
    )
    return response_loss + state_weight * state_alignment + plan_weight * plan_alignment


def symmetric_margin_loss(
    *,
    original_clamp_original_score: torch.Tensor,
    original_clamp_counterfactual_score: torch.Tensor,
    counterfactual_clamp_original_score: torch.Tensor,
    counterfactual_clamp_counterfactual_score: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    original_direction = torch.relu(
        margin
        - (
            original_clamp_original_score
            - original_clamp_counterfactual_score
        )
    )
    counterfactual_direction = torch.relu(
        margin
        - (
            counterfactual_clamp_counterfactual_score
            - counterfactual_clamp_original_score
        )
    )
    return original_direction.mean() + counterfactual_direction.mean()
