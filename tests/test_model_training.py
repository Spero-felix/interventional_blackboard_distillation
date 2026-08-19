from types import SimpleNamespace

import pytest
import torch
from torch import nn


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=7, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, **kwargs):
        return SimpleNamespace(logits=self.head(inputs_embeds))


def test_response_log_probs_mask_prompt_and_keep_only_response_tokens():
    from ibd.training import response_token_log_probs

    logits = torch.zeros(1, 5, 3)
    labels = torch.tensor([[-100, -100, 0, 1, 2]])

    token_log_probs, mask = response_token_log_probs(logits, labels)

    assert mask.tolist() == [[False, True, True, True]]
    assert token_log_probs[mask].tolist() == pytest.approx([-1.0986123] * 3)


def test_length_normalized_score_does_not_reward_shorter_sequence():
    from ibd.training import length_normalized_score

    scores = torch.tensor([[-0.2, -0.2, 0.0], [-0.2, -0.2, -0.2]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    assert length_normalized_score(scores, mask).tolist() == pytest.approx([-0.2, -0.2])


def test_margin_loss_is_zero_when_chosen_exceeds_required_gap():
    from ibd.training import margin_alignment_loss

    chosen = torch.tensor([0.9, 0.6])
    rejected = torch.tensor([0.2, 0.3])

    assert margin_alignment_loss(chosen, rejected, margin=0.25).item() == 0.0


def test_stage_d_combines_chosen_sft_margin_and_low_weight_replay():
    from ibd.training import stage_d_loss

    result = stage_d_loss(
        chosen_sft=torch.tensor(1.0),
        chosen_scores=torch.tensor([0.2]),
        rejected_scores=torch.tensor([0.1]),
        replay_loss=torch.tensor(2.0),
        margin=0.5,
        replay_weight=0.1,
    )

    assert result.item() == pytest.approx(1.6)


def test_adapter_has_exactly_state_and_plan_latent_slots_and_supports_clamp():
    from ibd.model import LatentBlackboardCausalLM

    adapter = LatentBlackboardCausalLM(TinyCausalLM(), hidden_size=4)
    with torch.no_grad():
        adapter.latent_slots["STATE"].fill_(2.0)
        adapter.latent_slots["PLAN"].fill_(3.0)
    embeddings = torch.zeros(1, 4, 4)
    positions = {"STATE": torch.tensor([1]), "PLAN": torch.tensor([2])}

    injected = adapter.inject_slots(
        embeddings,
        positions,
        clamps={"STATE": torch.full((1, 4), 5.0)},
    )

    assert set(adapter.latent_slots) == {"STATE", "PLAN"}
    assert len(adapter.latent_slots) == 2
    assert injected[0, 1].tolist() == [5.0] * 4
    assert injected[0, 2].tolist() == [3.0] * 4
    with pytest.raises(ValueError, match="STATE or PLAN"):
        adapter.inject_slots(embeddings, positions, clamps={"CRITIC": torch.ones(1, 4)})

