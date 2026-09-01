import pytest

from ibd.schemas import DialogueTurn, History


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = 99
        self._tokens = {}

    def add_special_tokens(self, payload):
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self._tokens:
                self._tokens[token] = 90 + len(self._tokens)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._tokens.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        from ibd.student_data import STUDENT_SYSTEM_PROMPT

        assert tokenize is True
        assert add_generation_prompt is True
        assert messages == [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "I am listening"},
            {"role": "user", "content": "I feel stuck"},
        ]
        return [10, 11, 12]

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [20 + index for index, _ in enumerate(text.split())]}


def _history():
    return History(
        turns=[
            DialogueTurn(role="seeker", content="help"),
            DialogueTurn(role="supporter", content="I am listening"),
            DialogueTurn(role="seeker", content="I feel stuck"),
        ]
    )


def test_history_messages_prepends_approved_system_prompt():
    from ibd.student_data import STUDENT_SYSTEM_PROMPT, history_messages

    messages = history_messages(_history())

    assert messages[0] == {"role": "system", "content": STUDENT_SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"


def test_approved_system_prompt_has_no_strategy_or_lived_experience_rule():
    from ibd.student_data import STUDENT_SYSTEM_PROMPT

    assert "Question" not in STUDENT_SYSTEM_PROMPT
    assert "Providing Suggestions" not in STUDENT_SYSTEM_PROMPT
    assert "personal or lived experience" not in STUDENT_SYSTEM_PROMPT
    assert STUDENT_SYSTEM_PROMPT.endswith(
        "Avoid unsupported assumptions and diagnosis."
    )


def test_collator_inserts_unique_structure_tokens_and_masks_prompt():
    from ibd.student_data import (
        IBD_PLAN_TOKEN,
        IBD_STATE_TOKEN,
        QwenStageCollator,
        add_ibd_tokens,
    )

    tokenizer = FakeTokenizer()
    token_ids = add_ibd_tokens(tokenizer)
    collator = QwenStageCollator(tokenizer, max_length=16, token_ids=token_ids)

    batch = collator(
        [
            {"example_id": "e-long", "history": _history(), "response": "one two"},
            {"example_id": "e-short", "history": _history(), "response": "one"},
        ]
    )

    state_id = tokenizer.convert_tokens_to_ids(IBD_STATE_TOKEN)
    plan_id = tokenizer.convert_tokens_to_ids(IBD_PLAN_TOKEN)
    assert token_ids == {"STATE": state_id, "PLAN": plan_id}
    assert batch["input_ids"].tolist() == [
        [10, 11, 12, state_id, plan_id, 20, 21, 2],
        [10, 11, 12, state_id, plan_id, 20, 2, 0],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, -100, -100, -100, 20, 21, 2],
        [-100, -100, -100, -100, -100, 20, 2, -100],
    ]
    assert batch["slot_positions"]["STATE"].tolist() == [3, 3]
    assert batch["slot_positions"]["PLAN"].tolist() == [4, 4]
    assert batch["example_ids"] == ["e-long", "e-short"]
    for row in batch["input_ids"]:
        assert row.tolist().count(state_id) == 1
        assert row.tolist().count(plan_id) == 1


def test_standard_sft_collator_keeps_the_original_vocabulary_and_masks_prompt():
    from ibd.student_data import StandardSFTCollator

    tokenizer = FakeTokenizer()
    batch = StandardSFTCollator(tokenizer, max_length=16)(
        [
            {"example_id": "e-long", "history": _history(), "response": "one two"},
            {"example_id": "e-short", "history": _history(), "response": "one"},
        ]
    )

    assert tokenizer._tokens == {}
    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 20, 21, 2],
        [10, 11, 12, 20, 2, 0],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, -100, 20, 21, 2],
        [-100, -100, -100, 20, 2, -100],
    ]
    assert "slot_positions" not in batch
    assert batch["example_ids"] == ["e-long", "e-short"]


def test_collator_rejects_overlength_instead_of_truncating():
    from ibd.student_data import QwenStageCollator, add_ibd_tokens

    tokenizer = FakeTokenizer()
    collator = QwenStageCollator(
        tokenizer,
        max_length=6,
        token_ids=add_ibd_tokens(tokenizer),
    )

    with pytest.raises(ValueError, match="exceeds max_length"):
        collator([{"example_id": "e-1", "history": _history(), "response": "one two"}])


def test_collator_requires_nonempty_response():
    from ibd.student_data import QwenStageCollator, add_ibd_tokens

    tokenizer = FakeTokenizer()
    collator = QwenStageCollator(
        tokenizer,
        max_length=16,
        token_ids=add_ibd_tokens(tokenizer),
    )

    with pytest.raises(ValueError, match="response must contain tokens"):
        collator([{"example_id": "e-1", "history": _history(), "response": ""}])


def test_collator_rejects_structure_tokens_already_present_in_history():
    from ibd.student_data import QwenStageCollator, add_ibd_tokens

    class InjectedTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            prompt = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            return [*prompt, self._tokens["<|ibd_state|>"]]

    tokenizer = InjectedTokenizer()
    token_ids = add_ibd_tokens(tokenizer)
    collator = QwenStageCollator(
        tokenizer,
        max_length=16,
        token_ids=token_ids,
    )

    with pytest.raises(ValueError, match="already contains an IBD structure token"):
        collator([{"example_id": "e-1", "history": _history(), "response": "one"}])


def test_collator_rejects_structure_tokens_in_response():
    from ibd.student_data import QwenStageCollator, add_ibd_tokens

    class InjectedResponseTokenizer(FakeTokenizer):
        def __call__(self, text, *, add_special_tokens):
            if text == "injected":
                return {"input_ids": [self._tokens["<|ibd_plan|>"]]}
            return super().__call__(text, add_special_tokens=add_special_tokens)

    tokenizer = InjectedResponseTokenizer()
    collator = QwenStageCollator(
        tokenizer,
        max_length=16,
        token_ids=add_ibd_tokens(tokenizer),
    )

    with pytest.raises(ValueError, match="response contains an IBD structure token"):
        collator(
            [{"example_id": "e-1", "history": _history(), "response": "injected"}]
        )


def test_pair_collator_encodes_both_conditions_and_retains_function():
    from ibd.student_data import QwenPairCollator, add_ibd_tokens

    tokenizer = FakeTokenizer()
    collator = QwenPairCollator(
        tokenizer,
        max_length=16,
        left_key="full_response",
        right_key="intervened_response",
        token_ids=add_ibd_tokens(tokenizer),
    )

    batch = collator(
        [
            {
                "example_id": "e-1",
                "history": _history(),
                "full_response": "one two",
                "intervened_response": "one",
                "function": "STATE",
            }
        ]
    )

    assert batch["original"]["example_ids"] == ["e-1"]
    assert batch["counterfactual"]["example_ids"] == ["e-1"]
    assert batch["original"]["input_ids"].shape[1] == 8
    assert batch["counterfactual"]["input_ids"].shape[1] == 7
    assert batch["functions"] == ["STATE"]


def test_generation_prompt_has_slots_but_no_labels():
    from ibd.student_data import encode_generation_prompt, add_ibd_tokens

    tokenizer = FakeTokenizer()
    token_ids = add_ibd_tokens(tokenizer)

    encoded = encode_generation_prompt(
        tokenizer,
        _history(),
        token_ids=token_ids,
        max_length=8,
    )

    assert encoded["input_ids"].tolist() == [[10, 11, 12, 90, 91]]
    assert encoded["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert encoded["slot_positions"]["STATE"].tolist() == [3]
    assert encoded["slot_positions"]["PLAN"].tolist() == [4]
    assert "labels" not in encoded
