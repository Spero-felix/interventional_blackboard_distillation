"""Conversation-safe SocialSim preparation for the Qwen training pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from .progress import track
from .schemas import DialogueTurn, History, StrictModel


SocialSimSplit = Literal["train", "dev", "diagnostic_holdout"]
_SPLIT_ORDER: tuple[SocialSimSplit, ...] = (
    "train",
    "dev",
    "diagnostic_holdout",
)


class SocialSimExample(StrictModel):
    """One final seeker context from one source conversation."""

    example_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    split: SocialSimSplit
    history: History


class PreparedSocialSim(StrictModel):
    protocol_version: str = "socialsim-qwen-conversation-v1"
    seed: int
    splits: dict[SocialSimSplit, list[SocialSimExample]]
    manifest: dict[str, Any]

    @model_validator(mode="after")
    def validate_conversation_splits(self) -> "PreparedSocialSim":
        if set(self.splits) != set(_SPLIT_ORDER):
            raise ValueError("SocialSim output must contain all three splits")
        seen: set[str] = set()
        for split, examples in self.splits.items():
            for example in examples:
                if example.split != split:
                    raise ValueError("SocialSim example split does not match its container")
                if example.conversation_id in seen:
                    raise ValueError("conversation IDs must not cross splits")
                seen.add(example.conversation_id)
        return self


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _records_by_id(records: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        conversation_id = _text(record.get("ID"))
        if not conversation_id or conversation_id in by_id:
            raise ValueError(f"SocialSim {kind} IDs must be non-empty and unique")
        by_id[conversation_id] = record
    return by_id


def _parse_dialogue(raw: Mapping[str, Any]) -> History:
    conversation_id = _text(raw.get("ID"))
    raw_turns = raw.get("Dialogue")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"Dialogue {conversation_id!r} has no turns")

    turns: list[DialogueTurn] = []
    for index, item in enumerate(raw_turns):
        if not isinstance(item, Mapping):
            raise ValueError(f"Dialogue {conversation_id!r} turn {index + 1} is not an object")
        source_turn = item.get("Turn")
        if source_turn != index + 1:
            raise ValueError(f"Dialogue {conversation_id!r} has non-contiguous Turn values")
        seeker = _text(item.get("Seeker"))
        supporter = _text(item.get("Supporter"))
        if bool(seeker) == bool(supporter):
            raise ValueError(
                f"Dialogue {conversation_id!r} turn {source_turn} must have exactly one role"
            )
        role: Literal["seeker", "supporter"] = "seeker" if seeker else "supporter"
        expected = "seeker" if len(turns) % 2 == 0 else "supporter"
        if role != expected:
            if index == len(raw_turns) - 1 and role == "supporter":
                break
            raise ValueError(f"Dialogue {conversation_id!r} does not alternate from seeker")
        turns.append(DialogueTurn(role=role, content=seeker or supporter))

    last_observed_supporter = max(
        (index for index, turn in enumerate(turns) if turn.role == "supporter"),
        default=-1,
    )
    if last_observed_supporter < 1:
        raise ValueError(f"Dialogue {conversation_id!r} has no eligible seeker/supporter pair")
    history_turns = turns[:last_observed_supporter]
    if not history_turns or history_turns[-1].role != "seeker":
        raise ValueError(f"Dialogue {conversation_id!r} has no final seeker context")
    return History(turns=history_turns)


def _validate_split_sizes(
    split_sizes: Mapping[str, int],
    *,
    limit: int,
) -> dict[SocialSimSplit, int]:
    if set(split_sizes) != set(_SPLIT_ORDER):
        raise ValueError("split_sizes must contain train, dev, and diagnostic_holdout")
    normalized = {name: int(split_sizes[name]) for name in _SPLIT_ORDER}
    if any(size < 0 for size in normalized.values()):
        raise ValueError("split sizes must be non-negative")
    if sum(normalized.values()) != limit:
        raise ValueError("split sizes must sum to limit")
    return normalized


def prepare_socialsim_examples(
    conversations: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    limit: int = 128,
    split_sizes: Mapping[str, int] | None = None,
) -> PreparedSocialSim:
    sizes = _validate_split_sizes(
        split_sizes
        or {"train": 96, "dev": 16, "diagnostic_holdout": 16},
        limit=limit,
    )
    conversations_by_id = _records_by_id(conversations, kind="conversation")
    profiles_by_id = _records_by_id(profiles, kind="profile")
    if set(conversations_by_id) != set(profiles_by_id):
        raise ValueError("SocialSim profile IDs do not match conversation IDs")

    eligible: list[tuple[str, History]] = []
    excluded: dict[str, str] = {}
    conversations = track(
        conversations_by_id.items(),
        desc="prepare SocialSim",
        total=len(conversations_by_id),
        unit="conversation",
    )
    for conversation_id, raw in conversations:
        try:
            eligible.append((conversation_id, _parse_dialogue(raw)))
        except ValueError as error:
            excluded[conversation_id] = str(error)

    eligible.sort(key=lambda item: item[0])
    random.Random(seed).shuffle(eligible)
    if len(eligible) < limit:
        raise ValueError(f"only {len(eligible)} eligible SocialSim conversations for limit {limit}")
    selected = eligible[:limit]

    splits: dict[SocialSimSplit, list[SocialSimExample]] = {
        name: [] for name in _SPLIT_ORDER
    }
    offset = 0
    split_ids: dict[SocialSimSplit, list[str]] = {name: [] for name in _SPLIT_ORDER}
    for split in _SPLIT_ORDER:
        for conversation_id, history in selected[offset : offset + sizes[split]]:
            split_ids[split].append(conversation_id)
            splits[split].append(
                SocialSimExample(
                    example_id=f"ssconv-{conversation_id}",
                    conversation_id=conversation_id,
                    split=split,
                    history=history,
                )
            )
        offset += sizes[split]

    return PreparedSocialSim(
        seed=seed,
        splits=splits,
        manifest={
            "protocol_version": "socialsim-qwen-conversation-v1",
            "seed": seed,
            "limit": limit,
            "counts": sizes,
            "split_ids": split_ids,
            "eligible_count": len(eligible),
            "excluded": dict(sorted(excluded.items())),
            "profile_usage": "ID integrity only; profile content is never exported",
        },
    )


def _load_json_objects(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SocialSim source file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
        raise ValueError(f"Expected a JSON list of objects in {path}")
    return payload


def load_socialsim_files(
    dialogue_path: str | Path,
    profile_path: str | Path,
    **prepare_kwargs: Any,
) -> PreparedSocialSim:
    dialogue_source = Path(dialogue_path).resolve()
    profile_source = Path(profile_path).resolve()
    prepared = prepare_socialsim_examples(
        _load_json_objects(dialogue_source),
        _load_json_objects(profile_source),
        **prepare_kwargs,
    )
    prepared.manifest.update(
        {
            "dialogue_path": str(dialogue_source),
            "profile_path": str(profile_source),
        }
    )
    return prepared
