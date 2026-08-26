"""Qwen chat-template encoding and response-only stage collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .schemas import History


IBD_STATE_TOKEN = "<|ibd_state|>"
IBD_PLAN_TOKEN = "<|ibd_plan|>"


def add_ibd_tokens(tokenizer: Any) -> dict[str, int]:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [IBD_STATE_TOKEN, IBD_PLAN_TOKEN]}
    )
    state_id = int(tokenizer.convert_tokens_to_ids(IBD_STATE_TOKEN))
    plan_id = int(tokenizer.convert_tokens_to_ids(IBD_PLAN_TOKEN))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if state_id == plan_id or state_id == unk_id or plan_id == unk_id:
        raise ValueError("tokenizer failed to add distinct IBD STATE and PLAN tokens")
    return {"STATE": state_id, "PLAN": plan_id}


def history_messages(history: History | Mapping[str, Any]) -> list[dict[str, str]]:
    parsed = history if isinstance(history, History) else History.model_validate(history)
    role_map = {"seeker": "user", "supporter": "assistant"}
    return [
        {"role": role_map[turn.role], "content": turn.content}
        for turn in parsed.turns
    ]


def encode_student_example(
    tokenizer: Any,
    example: Mapping[str, Any],
    *,
    max_length: int,
    token_ids: Mapping[str, int],
    response_key: str = "response",
) -> dict[str, Any]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    messages = history_messages(example["history"])
    prompt_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if set(prompt_ids) & {int(token_ids["STATE"]), int(token_ids["PLAN"])}:
        raise ValueError("history already contains an IBD structure token")
    state_position = len(prompt_ids)
    prompt_ids.extend([int(token_ids["STATE"]), int(token_ids["PLAN"])])
    encoded_response = tokenizer(
        str(example[response_key]),
        add_special_tokens=False,
    )
    response_ids = list(encoded_response["input_ids"])
    if not response_ids:
        raise ValueError("response must contain tokens")
    if set(response_ids) & {int(token_ids["STATE"]), int(token_ids["PLAN"])}:
        raise ValueError("response contains an IBD structure token")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and response_ids[-1] != eos_token_id:
        response_ids.append(int(eos_token_id))

    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"example {example.get('example_id', '<unknown>')} exceeds max_length "
            f"({len(input_ids)} > {max_length})"
        )
    return {
        "example_id": str(example["example_id"]),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
        "slot_positions": {"STATE": state_position, "PLAN": state_position + 1},
    }


def encode_generation_prompt(
    tokenizer: Any,
    history: History | Mapping[str, Any],
    *,
    token_ids: Mapping[str, int],
    max_length: int,
) -> dict[str, Any]:
    prompt_ids = list(
        tokenizer.apply_chat_template(
            history_messages(history),
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if set(prompt_ids) & {int(token_ids["STATE"]), int(token_ids["PLAN"])}:
        raise ValueError("history already contains an IBD structure token")
    state_position = len(prompt_ids)
    prompt_ids.extend([int(token_ids["STATE"]), int(token_ids["PLAN"])])
    if len(prompt_ids) > max_length:
        raise ValueError(
            f"generation prompt exceeds max_length ({len(prompt_ids)} > {max_length})"
        )
    return {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
        "slot_positions": {
            "STATE": torch.tensor([state_position], dtype=torch.long),
            "PLAN": torch.tensor([state_position + 1], dtype=torch.long),
        },
    }


def encode_base_generation_prompt(
    tokenizer: Any,
    history: History | Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    """Encode the original Qwen chat prompt without Student-only IBD tokens."""

    prompt_ids = list(
        tokenizer.apply_chat_template(
            history_messages(history),
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if len(prompt_ids) > max_length:
        raise ValueError(
            f"generation prompt exceeds max_length ({len(prompt_ids)} > {max_length})"
        )
    return {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
    }


class QwenStageCollator:
    """Right-pad one-response stage rows while retaining slot coordinates."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        token_ids: Mapping[str, int] | None = None,
        response_key: str = "response",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_ids = dict(token_ids or add_ibd_tokens(tokenizer))
        self.response_key = response_key

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        encoded = [
            encode_student_example(
                self.tokenizer,
                example,
                max_length=self.max_length,
                token_ids=self.token_ids,
                response_key=self.response_key,
            )
            for example in examples
        ]
        target_length = max(len(item["input_ids"]) for item in encoded)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (target_length - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(item["input_ids"], int(pad_id)) for item in encoded],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(item["attention_mask"], 0) for item in encoded],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(item["labels"], -100) for item in encoded],
                dtype=torch.long,
            ),
            "slot_positions": {
                name: torch.tensor(
                    [item["slot_positions"][name] for item in encoded],
                    dtype=torch.long,
                )
                for name in ("STATE", "PLAN")
            },
            "example_ids": [item["example_id"] for item in encoded],
        }


class QwenPairCollator:
    """Encode original/counterfactual responses without quality labels."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        left_key: str,
        right_key: str,
        token_ids: Mapping[str, int] | None = None,
    ):
        resolved_ids = dict(token_ids or add_ibd_tokens(tokenizer))
        self.left = QwenStageCollator(
            tokenizer,
            max_length=max_length,
            token_ids=resolved_ids,
            response_key=left_key,
        )
        self.right = QwenStageCollator(
            tokenizer,
            max_length=max_length,
            token_ids=resolved_ids,
            response_key=right_key,
        )

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "original": self.left(examples),
            "counterfactual": self.right(examples),
            "example_ids": [str(example["example_id"]) for example in examples],
        }
        if any("function" in example for example in examples):
            if not all("function" in example for example in examples):
                raise ValueError("pair rows must either all provide function or all omit it")
            result["functions"] = [str(example["function"]) for example in examples]
        return result
