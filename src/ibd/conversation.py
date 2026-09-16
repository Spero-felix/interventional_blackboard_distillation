"""Orchestration for complete profile-driven seeker-supporter conversations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .conversation_schemas import (
    ConversationCheckpoint,
    ConversationGenerationConfig,
    ConversationRoundAudit,
    ConversationTermination,
    ConversationTrace,
    ConversationTurn,
)
from .dialogue_manager import DialogueManager
from .schemas import DialogueTurn, History, TeacherTrace, UserContext
from .seeding import SeedDeriver
from .seeker import SeekerSimulator, ValidatedProfile
from .teacher import TeacherRunner, TeacherSession

ConversationStage = Literal[
    "seeker_simulator",
    "teacher_observe",
    "dialogue_manager",
    "teacher_respond",
]


class ConversationStageError(RuntimeError):
    def __init__(
        self,
        stage: ConversationStage,
        completed_rounds: int,
    ) -> None:
        super().__init__(
            f"conversation generation failed during {stage} "
            f"after {completed_rounds} complete rounds"
        )
        self.stage = stage
        self.completed_rounds = completed_rounds


def _public_history(turns: list[ConversationTurn]) -> History:
    return History(
        turns=[DialogueTurn(role=turn.role, content=turn.content) for turn in turns]
    )


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def flatten_teacher_traces(trace: ConversationTrace) -> list[TeacherTrace]:
    return [round_audit.teacher_trace for round_audit in trace.rounds]


class ConversationGenerator:
    def __init__(
        self,
        *,
        seeker: SeekerSimulator,
        teacher_runner: TeacherRunner,
        dialogue_manager: DialogueManager,
        generation_config: ConversationGenerationConfig,
        protocol_version: str,
    ) -> None:
        self.seeker = seeker
        self.teacher_runner = teacher_runner
        self.dialogue_manager = dialogue_manager
        self.generation_config = generation_config
        self.protocol_version = protocol_version

    def _terminal_trace(
        self,
        *,
        conversation_id: str,
        profile_id: str,
        base_seed: int,
        turns: list[ConversationTurn],
        rounds: list[ConversationRoundAudit],
        seeker_call_records,
    ) -> ConversationTrace | None:
        if not rounds:
            return None
        final_mode = rounds[-1].dialogue_decision.decision.mode
        if final_mode == "closing":
            status = "completed"
            reason = "dialogue_manager_closing"
        elif len(rounds) >= self.generation_config.hard_max_rounds:
            status = "truncated"
            reason = "hard_max_without_closing"
        else:
            return None
        return ConversationTrace(
            protocol_version=self.protocol_version,
            conversation_id=conversation_id,
            profile_id=profile_id,
            seed=base_seed,
            split=self.generation_config.split,
            status=status,
            turns=turns,
            rounds=rounds,
            termination=ConversationTermination(
                reason=reason,
                round_count=len(rounds),
                final_mode=final_mode,
            ),
            seeker_call_records=seeker_call_records,
        )

    def generate(
        self,
        profile: ValidatedProfile,
        *,
        base_seed: int,
        checkpoint: ConversationCheckpoint | None = None,
        on_round_completed: Callable[[ConversationCheckpoint], None] | None = None,
    ) -> ConversationTrace:
        profile_id = profile.profile_id
        if _contains_control_characters(profile_id):
            raise ValueError("profile ID must not contain control characters")
        conversation_id = f"{profile_id}-seed-{base_seed}"

        if checkpoint is None:
            turns: list[ConversationTurn] = []
            rounds: list[ConversationRoundAudit] = []
            seeker_call_records = []
            previous_mode = "opening"
            next_round_index = 1
            session = TeacherSession(self.teacher_runner, context=UserContext.empty())
        else:
            if (
                checkpoint.protocol_version != self.protocol_version
                or checkpoint.conversation_id != conversation_id
                or checkpoint.profile_id != profile_id
                or checkpoint.seed != base_seed
                or checkpoint.generation_config != self.generation_config
            ):
                raise ValueError("checkpoint identity or generation configuration mismatch")
            turns = list(checkpoint.turns)
            rounds = list(checkpoint.rounds)
            seeker_call_records = list(checkpoint.seeker_call_records)
            previous_mode = checkpoint.previous_mode
            next_round_index = checkpoint.next_round_index
            session = TeacherSession(self.teacher_runner, context=checkpoint.last_context)

        terminal = self._terminal_trace(
            conversation_id=conversation_id,
            profile_id=profile_id,
            base_seed=base_seed,
            turns=turns,
            rounds=rounds,
            seeker_call_records=seeker_call_records,
        )
        if terminal is not None:
            return terminal

        round_index = next_round_index
        while True:
            seeds = SeedDeriver(
                self.protocol_version,
                conversation_id,
                base_seed,
                round_index,
            )
            try:
                seeker_generation = self.seeker.generate(
                    profile,
                    turns,
                    previous_mode,
                    round_index=round_index,
                    seed=seeds.for_call("seeker_simulator"),
                    conversation_id=conversation_id,
                )
            except Exception as exc:
                raise ConversationStageError(
                    "seeker_simulator", len(rounds)
                ) from exc

            seeker_turn = ConversationTurn(
                turn_index=len(turns) + 1,
                round_index=round_index,
                role="seeker",
                content=seeker_generation.utterance,
            )
            turns.append(seeker_turn)
            history = _public_history(turns)
            example_id = f"{conversation_id}:round:{round_index}"

            try:
                observation = session.observe(
                    example_id,
                    history,
                    seed_deriver=seeds,
                )
            except Exception as exc:
                raise ConversationStageError("teacher_observe", len(rounds)) from exc

            recent_strategies = [
                item.teacher_trace.final_selection.selected_strategy
                for item in rounds[-3:]
            ]
            try:
                decision_record = self.dialogue_manager.decide(
                    observation,
                    previous_mode=previous_mode,
                    recent_selected_strategies=recent_strategies,
                    round_index=round_index,
                    generation_config=self.generation_config,
                    seed=seeds.for_call("dialogue_manager"),
                    conversation_id=conversation_id,
                )
            except Exception as exc:
                raise ConversationStageError("dialogue_manager", len(rounds)) from exc

            try:
                teacher_trace = session.respond(
                    observation,
                    split=self.generation_config.split,
                    dialogue_mode=decision_record.decision.mode,
                    seed_deriver=seeds,
                )
            except Exception as exc:
                raise ConversationStageError("teacher_respond", len(rounds)) from exc

            supporter_turn = ConversationTurn(
                turn_index=len(turns) + 1,
                round_index=round_index,
                role="supporter",
                content=teacher_trace.final_response,
            )
            turns.append(supporter_turn)
            seeker_call_records.append(seeker_generation.call_record)
            rounds.append(
                ConversationRoundAudit(
                    round_index=round_index,
                    seeker_turn_index=seeker_turn.turn_index,
                    supporter_turn_index=supporter_turn.turn_index,
                    dialogue_decision=decision_record,
                    teacher_trace=teacher_trace,
                )
            )

            checkpoint_record = ConversationCheckpoint(
                protocol_version=self.protocol_version,
                conversation_id=conversation_id,
                profile_id=profile_id,
                seed=base_seed,
                generation_config=self.generation_config,
                turns=turns,
                rounds=rounds,
                last_context=session.context,
                previous_mode=decision_record.decision.mode,
                next_round_index=round_index + 1,
                seeker_call_records=seeker_call_records,
            )
            if on_round_completed is not None:
                on_round_completed(checkpoint_record)

            terminal = self._terminal_trace(
                conversation_id=conversation_id,
                profile_id=profile_id,
                base_seed=base_seed,
                turns=turns,
                rounds=rounds,
                seeker_call_records=seeker_call_records,
            )
            if terminal is not None:
                return terminal

            previous_mode = decision_record.decision.mode
            round_index += 1
