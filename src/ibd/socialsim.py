"""Conversation-safe SocialSim preparation for the Qwen training pipeline."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from .progress import track
from .schemas import DialogueTurn, History, StrictModel


SocialSimSplit = Literal["train", "dev", "diagnostic_holdout"]
DialoguePhase = Literal["early", "middle", "late"]
SocialSimProtocol = Literal[
    "socialsim-qwen-conversation-v1",
    "socialsim-qwen-conversation-v2",
]
_SPLIT_ORDER: tuple[SocialSimSplit, ...] = (
    "train",
    "dev",
    "diagnostic_holdout",
)
_PHASE_ORDER: tuple[DialoguePhase, ...] = ("early", "middle", "late")


class SocialSimExample(StrictModel):
    """One phase-selected seeker context from one source conversation."""

    example_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    split: SocialSimSplit
    history: History
    conversation_phase: DialoguePhase | None = None
    target_turn: int | None = Field(default=None, gt=0)
    target_rank: int | None = Field(default=None, gt=0)
    eligible_target_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_phase_metadata(self) -> "SocialSimExample":
        metadata = (
            self.conversation_phase,
            self.target_turn,
            self.target_rank,
            self.eligible_target_count,
        )
        if any(value is not None for value in metadata) and not all(
            value is not None for value in metadata
        ):
            raise ValueError("SocialSim phase metadata must be all present or all absent")
        if self.target_rank is not None and self.eligible_target_count is not None:
            if self.target_rank > self.eligible_target_count:
                raise ValueError("SocialSim target_rank exceeds eligible_target_count")
            if self.conversation_phase != _phase_for_rank(
                self.target_rank, self.eligible_target_count
            ):
                raise ValueError("SocialSim target rank does not match conversation phase")
        return self


class PreparedSocialSim(StrictModel):
    protocol_version: SocialSimProtocol = "socialsim-qwen-conversation-v2"
    seed: int
    splits: dict[SocialSimSplit, list[SocialSimExample]]
    manifest: dict[str, Any]

    @model_validator(mode="after")
    def validate_conversation_splits(self) -> "PreparedSocialSim":
        if set(self.splits) != set(_SPLIT_ORDER):
            raise ValueError("SocialSim output must contain all three splits")
        seen: set[str] = set()
        phase_counts: Counter[str] = Counter()
        split_phase_counts: dict[SocialSimSplit, Counter[str]] = {
            split: Counter() for split in _SPLIT_ORDER
        }
        for split, examples in self.splits.items():
            for example in examples:
                if example.split != split:
                    raise ValueError("SocialSim example split does not match its container")
                if example.conversation_id in seen:
                    raise ValueError("conversation IDs must not cross splits")
                seen.add(example.conversation_id)
                if self.protocol_version == "socialsim-qwen-conversation-v2":
                    if example.conversation_phase is None:
                        raise ValueError("SocialSim v2 examples require phase metadata")
                    phase_counts[example.conversation_phase] += 1
                    split_phase_counts[split][example.conversation_phase] += 1
        if self.protocol_version == "socialsim-qwen-conversation-v2":
            actual = {phase: phase_counts[phase] for phase in _PHASE_ORDER}
            declared = self.manifest.get("phase_counts")
            if declared != actual:
                raise ValueError("SocialSim manifest phase_counts do not match examples")
            actual_by_split = {
                split: {
                    phase: split_phase_counts[split][phase]
                    for phase in _PHASE_ORDER
                }
                for split in _SPLIT_ORDER
            }
            if self.manifest.get("split_phase_counts") != actual_by_split:
                raise ValueError(
                    "SocialSim manifest split_phase_counts do not match examples"
                )
        return self


@dataclass(frozen=True)
class _EligibleConversation:
    conversation_id: str
    turns: tuple[DialogueTurn, ...]
    target_indices: tuple[int, ...]


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


def _parse_dialogue(
    raw: Mapping[str, Any],
) -> tuple[list[DialogueTurn], tuple[int, ...]]:
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

    target_indices = tuple(
        index for index, turn in enumerate(turns) if turn.role == "supporter"
    )
    if len(target_indices) < 3:
        raise ValueError(
            f"Dialogue {conversation_id!r} has fewer than three response targets"
        )
    return turns, target_indices


def _phase_for_rank(target_rank: int, target_count: int) -> DialoguePhase:
    if target_count < 3 or not 1 <= target_rank <= target_count:
        raise ValueError("target rank requires at least three targets and must be in range")
    phase_index = min(2, ((target_rank - 1) * 3) // target_count)
    return _PHASE_ORDER[phase_index]


def _validate_phase_fractions(
    early_fraction: float,
    late_fraction: float,
) -> dict[DialoguePhase, float]:
    values = (early_fraction, late_fraction)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("phase fractions must be finite and non-negative")
    middle_fraction = 1.0 - early_fraction - late_fraction
    if not math.isfinite(middle_fraction) or middle_fraction <= 0.0:
        raise ValueError("phase fractions must leave a positive middle fraction")
    return {
        "early": early_fraction,
        "middle": middle_fraction,
        "late": late_fraction,
    }


def _phase_counts(
    size: int,
    fractions: Mapping[DialoguePhase, float],
) -> dict[DialoguePhase, int]:
    exact = {phase: size * fractions[phase] for phase in _PHASE_ORDER}
    counts = {phase: math.floor(exact[phase]) for phase in _PHASE_ORDER}
    remaining = size - sum(counts.values())
    phase_priority = {phase: index for index, phase in enumerate(_PHASE_ORDER)}
    by_remainder = sorted(
        _PHASE_ORDER,
        key=lambda phase: (
            -(exact[phase] - counts[phase]),
            phase_priority[phase],
        ),
    )
    for phase in by_remainder[:remaining]:
        counts[phase] += 1
    return counts


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
    early_fraction: float = 0.1,
    late_fraction: float = 0.1,
) -> PreparedSocialSim:
    sizes = _validate_split_sizes(
        split_sizes
        or {"train": 96, "dev": 16, "diagnostic_holdout": 16},
        limit=limit,
    )
    phase_fractions = _validate_phase_fractions(early_fraction, late_fraction)
    conversations_by_id = _records_by_id(conversations, kind="conversation")
    profiles_by_id = _records_by_id(profiles, kind="profile")
    if set(conversations_by_id) != set(profiles_by_id):
        raise ValueError("SocialSim profile IDs do not match conversation IDs")

    eligible: list[_EligibleConversation] = []
    excluded: dict[str, str] = {}
    conversations = track(
        conversations_by_id.items(),
        desc="prepare SocialSim",
        total=len(conversations_by_id),
        unit="conversation",
    )
    for conversation_id, raw in conversations:
        try:
            turns, target_indices = _parse_dialogue(raw)
            eligible.append(
                _EligibleConversation(
                    conversation_id=conversation_id,
                    turns=tuple(turns),
                    target_indices=target_indices,
                )
            )
        except ValueError as error:
            excluded[conversation_id] = str(error)

    eligible.sort(key=lambda item: item.conversation_id)
    random.Random(seed).shuffle(eligible)
    if len(eligible) < limit:
        raise ValueError(f"only {len(eligible)} eligible SocialSim conversations for limit {limit}")
    selected = eligible[:limit]

    splits: dict[SocialSimSplit, list[SocialSimExample]] = {
        name: [] for name in _SPLIT_ORDER
    }
    offset = 0
    split_ids: dict[SocialSimSplit, list[str]] = {name: [] for name in _SPLIT_ORDER}
    split_phase_counts: dict[SocialSimSplit, dict[DialoguePhase, int]] = {}
    phase_counts: Counter[str] = Counter()
    for split_index, split in enumerate(_SPLIT_ORDER):
        selected_for_split = selected[offset : offset + sizes[split]]
        counts = _phase_counts(sizes[split], phase_fractions)
        split_phase_counts[split] = counts
        phase_counts.update(counts)
        assigned_phases = [
            phase
            for phase in _PHASE_ORDER
            for _ in range(counts[phase])
        ]
        split_rng = random.Random(seed + (split_index + 1) * 1_000_003)
        split_rng.shuffle(assigned_phases)
        for conversation, phase in zip(selected_for_split, assigned_phases, strict=True):
            target_ranks = [
                rank
                for rank in range(1, len(conversation.target_indices) + 1)
                if _phase_for_rank(rank, len(conversation.target_indices)) == phase
            ]
            target_rank = split_rng.choice(target_ranks)
            target_index = conversation.target_indices[target_rank - 1]
            history = History(turns=list(conversation.turns[:target_index]))
            conversation_id = conversation.conversation_id
            split_ids[split].append(conversation_id)
            splits[split].append(
                SocialSimExample(
                    example_id=f"ssconv-{conversation_id}",
                    conversation_id=conversation_id,
                    split=split,
                    history=history,
                    conversation_phase=phase,
                    target_turn=target_index + 1,
                    target_rank=target_rank,
                    eligible_target_count=len(conversation.target_indices),
                )
            )
        offset += sizes[split]

    return PreparedSocialSim(
        protocol_version="socialsim-qwen-conversation-v2",
        seed=seed,
        splits=splits,
        manifest={
            "protocol_version": "socialsim-qwen-conversation-v2",
            "seed": seed,
            "limit": limit,
            "counts": sizes,
            "split_ids": split_ids,
            "phase_fractions": phase_fractions,
            "phase_counts": {
                phase: phase_counts[phase] for phase in _PHASE_ORDER
            },
            "split_phase_counts": split_phase_counts,
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
