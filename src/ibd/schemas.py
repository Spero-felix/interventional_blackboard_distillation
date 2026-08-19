"""Typed, auditable contracts for Teacher traces and Student datasets."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExpertName = Literal["emotion", "need", "relationship", "intent"]
FunctionName = Literal["STATE", "PLAN"]
NonSafetyDimension = Literal[
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
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DialogueTurn(StrictModel):
    role: Literal["seeker", "supporter"]
    content: str = Field(min_length=1)


class History(StrictModel):
    turns: list[DialogueTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def require_seeker_last(self) -> "History":
        if self.turns[-1].role != "seeker":
            raise ValueError("history must end with a seeker turn")
        return self

    def as_prompt(self) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self.turns)


_EXPERT_FIELDS: dict[str, frozenset[str]] = {
    "emotion": frozenset({"emotion", "intensity", "trajectory", "coping_signal"}),
    "need": frozenset({"need", "priority", "readiness", "constraint"}),
    "relationship": frozenset(
        {"relationship_type", "relationship_pattern", "power_dynamic", "boundary_signal"}
    ),
    "intent": frozenset({"intent", "explicit_ask", "implicit_goal", "decision_stage"}),
}


class ExpertOutput(StrictModel):
    expert: ExpertName
    fields: dict[str, Any]
    evidence: list[str]
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def keep_domain_isolated(self) -> "ExpertOutput":
        invalid = set(self.fields) - _EXPERT_FIELDS[self.expert]
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"fields {names} are not allowed for {self.expert} expert")
        return self


class StateBlackboard(StrictModel):
    emotions: dict[str, Any]
    needs: dict[str, Any]
    relationship: dict[str, Any]
    intent: dict[str, Any]
    readiness: str
    uncertainties: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SupportPlan(StrictModel):
    support_goals: list[str] = Field(min_length=1)
    response_acts: list[str] = Field(min_length=1)
    avoid: list[str] = Field(default_factory=list)
    rationale: str = ""


class Candidate(StrictModel):
    candidate_id: str
    response: str = Field(min_length=1)
    seed: int


class CritiqueIssue(StrictModel):
    dimension: str
    evidence: str
    severity: int = Field(ge=1, le=5)
    suggested_revision: str = ""


class CritiqueReport(StrictModel):
    critic: Literal["emotion", "effectiveness", "safety"]
    candidate_issues: dict[str, list[CritiqueIssue]]
    summary: str = ""


class QualityGate(StrictModel):
    accepted: bool
    safety_pass: bool
    defects: list[str] = Field(default_factory=list)
    repair_instruction: str = ""


class CallRecord(StrictModel):
    role: str
    attempt: int = Field(ge=1)
    request_hash: str
    raw_text: str
    parsed: dict[str, Any]
    schema_retry: bool = False


class TeacherTrace(StrictModel):
    example_id: str
    split: Literal["train", "validation", "test"] = "train"
    history: History
    expert_outputs: dict[ExpertName, ExpertOutput]
    state: StateBlackboard
    plan: SupportPlan
    candidates: list[Candidate]
    critiques: list[CritiqueReport]
    final_response: str
    quality_gate: QualityGate
    call_records: list[CallRecord]
    repaired_response: str | None = None


class Mutation(StrictModel):
    function: FunctionName
    operation: Literal["downgrade", "reorder"]
    field: str
    before: Any
    after: Any
    changed_field_count: int = 1

    @model_validator(mode="after")
    def exactly_one_field(self) -> "Mutation":
        if self.changed_field_count != 1:
            raise ValueError("a minimal intervention must change exactly one field")
        return self


class InterventionRecord(StrictModel):
    example_id: str
    function: FunctionName
    mutation: Mutation
    full_response: str
    ablated_response: str
    target_dimension: NonSafetyDimension
    localized_degradation: bool
    order_swap_verified: bool

    @model_validator(mode="after")
    def function_matches_mutation(self) -> "InterventionRecord":
        if self.function != self.mutation.function:
            raise ValueError("intervention and mutation functions must match")
        return self


class MarginPair(StrictModel):
    example_id: str
    prompt: str
    chosen: str
    rejected_candidate_id: str
    rejected: str
    defect_dimension: NonSafetyDimension
    defect_evidence: str
    order_swap_verified: bool
    safety_filter_passed: bool

    @model_validator(mode="after")
    def require_safe_pair(self) -> "MarginPair":
        if not self.safety_filter_passed:
            raise ValueError("margin pairs must use a safety-filter-passing rejected candidate")
        if not self.order_swap_verified:
            raise ValueError("margin pairs must pass order-swap verification")
        return self

