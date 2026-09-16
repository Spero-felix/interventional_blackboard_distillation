"""LLM-based broad dialogue-mode selection over public conversation state."""

from __future__ import annotations

from collections.abc import Sequence

from .backend import LLMBackend, StructuredCallError, StructuredCaller
from .config import AppConfig
from .conversation_schemas import (
    ConversationGenerationConfig,
    DialogueDecisionRecord,
    DialogueManagementDecision,
    DialogueMode,
)
from .prompting import build_messages
from .schemas import StrategyName, TurnObservation


class DialogueManager:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.caller = StructuredCaller(backend, config)

    def decide(
        self,
        observation: TurnObservation,
        *,
        previous_mode: DialogueMode,
        recent_selected_strategies: Sequence[StrategyName],
        round_index: int,
        generation_config: ConversationGenerationConfig,
        seed: int,
        conversation_id: str,
    ) -> DialogueDecisionRecord:
        context = {
            "user_context": observation.context_after,
            "state_analysis": observation.state_analysis,
            "previous_mode": previous_mode,
            "recent_selected_strategies": list(recent_selected_strategies),
            "round_index": round_index,
            "min_rounds": generation_config.min_rounds,
            "soft_max_rounds": generation_config.soft_max_rounds,
            "hard_max_rounds": generation_config.hard_max_rounds,
        }
        try:
            decision, call_records = self.caller.call(
                "dialogue_manager",
                build_messages(
                    "dialogue_manager",
                    observation.history,
                    DialogueManagementDecision,
                    context=context,
                ),
                DialogueManagementDecision,
                seed=seed,
                example_id=conversation_id,
                cache_variant=f"round-{round_index}",
            )
        except StructuredCallError as exc:
            decision = DialogueManagementDecision(
                mode=previous_mode,
                transition_reason=(
                    "fallback_to_previous_mode_after_schema_failure: "
                    f"{exc}"
                ),
            )
            call_records = list(exc.records)
        return DialogueDecisionRecord(
            round_index=round_index,
            mode_before=previous_mode,
            decision=decision,
            call_records=call_records,
        )
