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


def test_response_log_probs_do_not_save_a_second_full_vocab_tensor():
    from ibd.training import response_token_log_probs

    logits = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, 1, -100, 3], [-100, 2, 0, 1]])
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.untyped_storage().data_ptr()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        scores, _ = response_token_log_probs(logits, labels)
        scores.sum().backward()

    full_vocab_storages = {
        storage for shape, storage in saved if shape == (2, 3, 5)
    }
    assert full_vocab_storages == {logits.untyped_storage().data_ptr()}


def test_response_log_probs_match_log_softmax_values_and_gradients():
    from ibd.training import response_token_log_probs

    labels = torch.tensor([[-100, 1, -100, 3], [-100, 2, 0, 1]])
    values = torch.tensor(
        [
            [
                [0.1, -0.2, 0.3, 0.0],
                [0.7, 0.1, -0.4, 0.2],
                [0.0, 0.5, 0.2, -0.1],
                [0.4, 0.3, -0.2, 0.6],
            ],
            [
                [-0.3, 0.2, 0.4, 0.1],
                [0.1, -0.5, 0.6, 0.2],
                [0.8, 0.0, -0.1, 0.3],
                [-0.2, 0.4, 0.5, 0.0],
            ],
        ],
        dtype=torch.float64,
    )
    actual_logits = values.detach().clone().requires_grad_()
    expected_logits = values.detach().clone().requires_grad_()

    actual_scores, actual_mask = response_token_log_probs(actual_logits, labels)
    shifted_labels = labels[:, 1:]
    expected_mask = shifted_labels.ne(-100)
    expected_scores = torch.log_softmax(expected_logits[:, :-1, :], dim=-1).gather(
        -1, shifted_labels.masked_fill(~expected_mask, 0).unsqueeze(-1)
    ).squeeze(-1).masked_fill(~expected_mask, 0.0)

    assert torch.equal(actual_mask, expected_mask)
    assert torch.allclose(actual_scores, expected_scores)

    actual_scores.sum().backward()
    expected_scores.sum().backward()
    assert torch.allclose(actual_logits.grad, expected_logits.grad)


def test_length_normalized_score_does_not_reward_shorter_sequence():
    from ibd.training import length_normalized_score

    scores = torch.tensor([[-0.2, -0.2, 0.0], [-0.2, -0.2, -0.2]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    assert length_normalized_score(scores, mask).tolist() == pytest.approx([-0.2, -0.2])


def test_stage_c_natural_sft_always_uses_original_response_identity():
    from ibd.trainer import _stage_c_natural_sft_scores

    selected = _stage_c_natural_sft_scores(
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.7, 0.8]),
    )

    assert torch.equal(selected, torch.tensor([0.2, 0.3]))


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


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_qwen_slot_wrapper_is_normal_forward_equivalent_and_captures_slots():
    from ibd.model import QwenSlotCausalLM

    torch.manual_seed(7)
    base = _tiny_qwen()
    input_ids = torch.tensor([[4, 5, 30, 31, 6, 7]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = base(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = QwenSlotCausalLM(
        base,
        state_token_id=30,
        plan_token_id=31,
        slot_layer=1,
    )
    with torch.no_grad():
        actual = wrapper(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.equal(actual.logits, expected)
    assert actual.slots.STATE.shape == (1, 16)
    assert actual.slots.PLAN.shape == (1, 16)


def test_qwen_slot_wrapper_clamps_state_without_directly_overwriting_plan():
    from ibd.model import QwenSlotCausalLM

    torch.manual_seed(11)
    wrapper = QwenSlotCausalLM(
        _tiny_qwen(),
        state_token_id=30,
        plan_token_id=31,
        slot_layer=1,
    )
    input_ids = torch.tensor([[4, 30, 31, 6]])
    clamp = torch.full((1, 16), 0.25)

    normal = wrapper(input_ids=input_ids)
    intervened = wrapper(input_ids=input_ids, clamps={"STATE": clamp})

    assert torch.equal(intervened.slots.STATE, clamp)
    assert torch.equal(intervened.slots.PLAN, normal.slots.PLAN)
    assert not torch.equal(intervened.logits, normal.logits)


def test_qwen_slot_wrapper_supports_cached_generation_with_prefill_clamp():
    from ibd.model import QwenSlotCausalLM

    torch.manual_seed(13)
    wrapper = QwenSlotCausalLM(
        _tiny_qwen(),
        state_token_id=30,
        plan_token_id=31,
        slot_layer=1,
    )
    input_ids = torch.tensor([[4, 30, 31]])

    generated = wrapper.generate(
        input_ids=input_ids,
        clamps={"PLAN": torch.zeros(1, 16)},
        max_new_tokens=2,
        do_sample=False,
    )

    assert generated.shape == (1, 5)
    assert torch.equal(generated[:, :3], input_ids)


def test_non_reentrant_checkpointing_preserves_slot_alignment_gradients():
    from ibd.model import QwenSlotCausalLM

    torch.manual_seed(17)
    base = _tiny_qwen()
    base.train()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    wrapper = QwenSlotCausalLM(
        base,
        state_token_id=30,
        plan_token_id=31,
        slot_layer=1,
    )
    output = wrapper(input_ids=torch.tensor([[4, 30, 31, 6]]), use_cache=False)

    assert output.slots.STATE.requires_grad
    output.slots.STATE.square().mean().backward()
    assert base.model.embed_tokens.weight.grad is not None


def test_qwen_slot_wrapper_requires_exactly_one_of_each_structure_token():
    from ibd.model import QwenSlotCausalLM

    wrapper = QwenSlotCausalLM(
        _tiny_qwen(),
        state_token_id=30,
        plan_token_id=31,
        slot_layer=1,
    )

    with pytest.raises(ValueError, match="exactly one STATE token"):
        wrapper(input_ids=torch.tensor([[4, 31, 6]]))
    with pytest.raises(ValueError, match="exactly one PLAN token"):
        wrapper(input_ids=torch.tensor([[4, 30, 31, 31, 6]]))


def test_contrastive_alignment_uses_correct_anchor_rows_and_freezes_bank():
    from ibd.training import contrastive_alignment_loss

    slots = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positive_rows = torch.tensor([0, 1])

    correct = contrastive_alignment_loss(
        slots,
        bank,
        positive_rows,
        temperature=0.1,
        cosine_weight=0.5,
    )
    swapped = contrastive_alignment_loss(
        slots,
        bank,
        torch.tensor([1, 0]),
        temperature=0.1,
        cosine_weight=0.5,
    )
    correct.backward()

    assert correct.item() < swapped.item()
    assert slots.grad is not None
    assert bank.grad is None


def test_stage_b_combines_response_and_both_slot_alignment_losses():
    from ibd.training import slot_alignment_loss, stage_b_loss

    state = torch.tensor([[1.0, 0.0]], requires_grad=True)
    plan = torch.tensor([[0.0, 1.0]], requires_grad=True)
    kwargs = {
        "state_slots": state,
        "plan_slots": plan,
        "state_bank": torch.eye(2),
        "plan_bank": torch.flip(torch.eye(2), dims=[0]),
        "positive_rows": torch.tensor([0]),
        "temperature": 0.2,
        "cosine_weight": 1.0,
        "state_weight": 0.5,
        "plan_weight": 0.25,
    }
    alignment = slot_alignment_loss(**kwargs)
    result = stage_b_loss(
        response_loss=torch.tensor(1.0),
        **kwargs,
    )

    expected_alignment = torch.nn.functional.cross_entropy(
        torch.tensor([[5.0, 0.0]]), torch.tensor([0])
    )
    assert result.item() == pytest.approx(
        (1.0 + 0.5 * expected_alignment + 0.25 * expected_alignment).item()
    )
    torch.testing.assert_close(result, torch.tensor(1.0) + alignment)
    alignment.backward()
    assert state.grad is not None
    assert plan.grad is not None
