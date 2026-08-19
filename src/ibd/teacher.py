"""Four-expert Teacher controller with deterministic call accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from .backend import LLMBackend, StructuredCaller
from .config import AppConfig
from .prompting import build_messages
from .schemas import (
    CallRecord,
    Candidate,
    CritiqueReport,
    ExpertName,
    ExpertOutput,
    History,
    QualityGate,
    StateBlackboard,
    StrictModel,
    SupportPlan,
    TeacherTrace,
)

NORMAL_ROLES = (
    "emotion_expert",
    "need_expert",
    "relationship_expert",
    "intent_expert",
    "state_integrator",
    "planner",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "emotion_critic",
    "effectiveness_critic",
    "safety_critic",
    "final_integrator",
    "quality_gate",
)


class FinalAnswer(StrictModel):
    response: str = Field(min_length=1)


@dataclass(frozen=True)
class DownstreamResult:
    response: str
    plan: SupportPlan
    candidates: list[Candidate]
    critiques: list[CritiqueReport]
    quality_gate: QualityGate
    call_records: list[CallRecord]


class TeacherRunner:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.config = config
        self.caller = StructuredCaller(backend, config)

    def _call(
        self,
        role: str,
        history: History,
        response_model: type[StrictModel],
        records: list[CallRecord],
        *,
        context: dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        parsed, new_records = self.caller.call(
            role,
            build_messages(role, history, response_model, context=context),
            response_model,
            seed=seed,
        )
        records.extend(new_records)
        return parsed

    def run(
        self,
        example_id: str,
        history: History,
        *,
        split: Literal["train", "dev", "test"] = "train",
    ) -> TeacherTrace:
        records: list[CallRecord] = []
        expert_outputs: dict[ExpertName, ExpertOutput] = {}
        for expert in ("emotion", "need", "relationship", "intent"):
            output = self._call(f"{expert}_expert", history, ExpertOutput, records)
            expert_outputs[expert] = output

        state = self._call(
            "state_integrator",
            history,
            StateBlackboard,
            records,
            context={"expert_outputs": expert_outputs},
        )
        downstream = self._run_from_state(history, state, records)
        final_response = downstream.response
        repaired_response: str | None = None
        gate = downstream.quality_gate
        if not gate.accepted:
            repaired = self._call(
                "repair",
                history,
                FinalAnswer,
                records,
                context={"response": final_response, "repair_instruction": gate.repair_instruction},
            )
            repaired_response = repaired.response
            final_response = repaired.response
            gate = self._call(
                "quality_gate_recheck",
                history,
                QualityGate,
                records,
                context={"response": final_response},
            )

        return TeacherTrace(
            example_id=example_id,
            split=split,
            history=history,
            expert_outputs=expert_outputs,
            state=state,
            plan=downstream.plan,
            candidates=downstream.candidates,
            critiques=downstream.critiques,
            final_response=final_response,
            quality_gate=gate,
            call_records=records,
            repaired_response=repaired_response,
        )

    def _run_from_state(
        self,
        history: History,
        state: StateBlackboard,
        records: list[CallRecord],
    ) -> DownstreamResult:
        plan = self._call("planner", history, SupportPlan, records, context={"state": state})
        return self._run_from_plan(history, state, plan, records)

    def _run_from_plan(
        self,
        history: History,
        state: StateBlackboard,
        plan: SupportPlan,
        records: list[CallRecord],
    ) -> DownstreamResult:
        candidates: list[Candidate] = []
        for index, seed in enumerate(self.config.candidate_seeds, start=1):
            candidate_id = str(index)
            candidate = self._call(
                f"candidate_{index}",
                history,
                Candidate,
                records,
                context={
                    "state": state,
                    "plan": plan,
                    "candidate_id": candidate_id,
                    "seed": seed,
                },
                seed=seed,
            )
            if candidate.candidate_id != candidate_id or candidate.seed != seed:
                raise ValueError(f"candidate metadata mismatch for candidate_{index}")
            candidates.append(candidate)
        critiques = [
            self._call(
                role,
                history,
                CritiqueReport,
                records,
                context={"candidates": candidates},
            )
            for role in ("emotion_critic", "effectiveness_critic", "safety_critic")
        ]
        final = self._call(
            "final_integrator",
            history,
            FinalAnswer,
            records,
            context={"state": state, "plan": plan, "candidates": candidates, "critiques": critiques},
        )
        gate = self._call(
            "quality_gate",
            history,
            QualityGate,
            records,
            context={"response": final.response},
        )
        return DownstreamResult(final.response, plan, candidates, critiques, gate, records)

    def rerun_downstream(
        self,
        history: History,
        state: StateBlackboard,
        plan: SupportPlan | None = None,
    ) -> DownstreamResult:
        """Rerun only nodes downstream of a clamped STATE or PLAN."""
        records: list[CallRecord] = []
        if plan is None:
            return self._run_from_state(history, state, records)
        return self._run_from_plan(history, state, plan, records)
