"""Minimal STATE/PLAN interventions and critique-grounded Stage D pairs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import Field

from .backend import LLMBackend, StructuredCaller
from .config import AppConfig
from .prompting import build_messages
from .schemas import (
    Candidate,
    InterventionRecord,
    MarginPair,
    Mutation,
    NonSafetyDimension,
    StateBlackboard,
    StrictModel,
    SupportPlan,
    TeacherTrace,
)
from .teacher import TeacherRunner

_NON_SAFETY_DIMENSIONS = {
    "emotion",
    "need",
    "relationship",
    "intent",
    "specificity",
    "timing",
    "effectiveness",
    "autonomy",
    "factuality",
    "non_template",
}


class PairVerdict(StrictModel):
    preferred: Literal["A", "B", "tie"]
    defect_dimension: NonSafetyDimension
    evidence: str = Field(min_length=1)


def mutate_state(state: StateBlackboard) -> tuple[StateBlackboard, Mutation]:
    """Downgrade one supported need to uncertainty without inventing facts."""
    if not state.needs:
        raise ValueError("STATE needs must contain at least one field")
    field = "primary" if "primary" in state.needs else sorted(state.needs)[0]
    before = state.needs[field]
    updated = state.model_copy(deep=True)
    updated.needs[field] = "uncertain"
    return updated, Mutation(
        function="STATE",
        operation="downgrade",
        field=f"needs.{field}",
        before=before,
        after="uncertain",
        changed_field_count=1,
    )


def mutate_plan(plan: SupportPlan) -> tuple[SupportPlan, Mutation]:
    """Swap the first two acts while changing only the response_acts field."""
    if len(plan.response_acts) < 2:
        raise ValueError("PLAN needs at least two response acts for a reorder intervention")
    before = list(plan.response_acts)
    updated = plan.model_copy(deep=True)
    updated.response_acts[0], updated.response_acts[1] = (
        updated.response_acts[1],
        updated.response_acts[0],
    )
    return updated, Mutation(
        function="PLAN",
        operation="reorder",
        field="response_acts",
        before=before,
        after=list(updated.response_acts),
        changed_field_count=1,
    )


class InterventionBuilder:
    """Build a retained trajectory after an external swapped-order effect check."""

    def __init__(
        self,
        runner: TeacherRunner,
        verify_effect: Callable[[str, str, str], bool],
    ):
        self.runner = runner
        self.verify_effect = verify_effect

    def build(
        self,
        trace: TeacherTrace,
        function: Literal["STATE", "PLAN"],
    ) -> InterventionRecord | None:
        if function == "STATE":
            state, mutation = mutate_state(trace.state)
            downstream = self.runner.rerun_downstream(trace.history, state)
            dimension: NonSafetyDimension = "specificity"
        else:
            plan, mutation = mutate_plan(trace.plan)
            downstream = self.runner.rerun_downstream(trace.history, trace.state, plan)
            dimension = "timing"
        verified = self.verify_effect(trace.final_response, downstream.response, dimension)
        if not (verified and downstream.quality_gate.safety_pass):
            return None
        return InterventionRecord(
            example_id=trace.example_id,
            function=function,
            mutation=mutation,
            full_response=trace.final_response,
            ablated_response=downstream.response,
            target_dimension=dimension,
            localized_degradation=True,
            order_swap_verified=True,
        )


class MarginPairBuilder:
    """Choose a localized non-safety negative and verify preference in A/B and B/A order."""

    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.caller = StructuredCaller(backend, config)

    @staticmethod
    def _candidate_by_id(trace: TeacherTrace, candidate_id: str) -> Candidate | None:
        return next((item for item in trace.candidates if item.candidate_id == candidate_id), None)

    def _select_negative(self, trace: TeacherTrace):
        unsafe_ids: set[str] = set()
        scored: list[tuple[int, str, str, str]] = []
        for report in trace.critiques:
            for candidate_id, issues in report.candidate_issues.items():
                if report.critic == "safety" and issues:
                    unsafe_ids.add(candidate_id)
                    continue
                if report.critic == "safety":
                    continue
                for issue in issues:
                    if issue.dimension in _NON_SAFETY_DIMENSIONS:
                        scored.append(
                            (issue.severity, candidate_id, issue.dimension, issue.evidence)
                        )
        for _, candidate_id, dimension, evidence in sorted(scored, reverse=True):
            candidate = self._candidate_by_id(trace, candidate_id)
            if candidate is not None and candidate_id not in unsafe_ids:
                return candidate, dimension, evidence
        return None

    def _verify_order_swap(
        self,
        trace: TeacherTrace,
        rejected: Candidate,
        dimension: str,
    ) -> tuple[bool, PairVerdict, PairVerdict]:
        first, _ = self.caller.call(
            "pair_verifier",
            build_messages(
                "pair_verifier",
                trace.history,
                PairVerdict,
                context={"A": trace.final_response, "B": rejected.response, "dimension": dimension},
            ),
            PairVerdict,
        )
        second, _ = self.caller.call(
            "pair_verifier",
            build_messages(
                "pair_verifier",
                trace.history,
                PairVerdict,
                context={"A": rejected.response, "B": trace.final_response, "dimension": dimension},
            ),
            PairVerdict,
        )
        valid = (
            first.preferred == "A"
            and second.preferred == "B"
            and first.defect_dimension == second.defect_dimension == dimension
        )
        return valid, first, second

    def build(self, trace: TeacherTrace) -> MarginPair | None:
        selected = self._select_negative(trace)
        if selected is None or not trace.quality_gate.safety_pass:
            return None
        rejected, dimension, evidence = selected
        verified, first, _ = self._verify_order_swap(trace, rejected, dimension)
        if not verified:
            return None
        return MarginPair(
            example_id=trace.example_id,
            prompt=trace.history.as_prompt(),
            chosen=trace.final_response,
            rejected_candidate_id=rejected.candidate_id,
            rejected=rejected.response,
            defect_dimension=first.defect_dimension,
            defect_evidence=evidence,
            order_swap_verified=True,
            safety_filter_passed=True,
        )
