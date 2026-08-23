import pytest
from pydantic import ValidationError

from ibd.schemas import (
    CritiqueIssue,
    CritiqueReport,
    DialogueTurn,
    ExpertOutput,
    History,
    InterventionRecord,
    MarginPair,
    Mutation,
    StateCounterfactual,
    StrategyUse,
)


def test_critique_report_allows_at_most_one_issue_per_candidate():
    issue = CritiqueIssue(
        dimension="timing",
        evidence="The suggestion arrives too early.",
        severity=2,
        suggested_revision="Ask permission first.",
    )

    with pytest.raises(ValidationError, match="at most 1 item"):
        CritiqueReport(
            critic="effectiveness",
            candidate_issues={"1": [issue, issue], "2": [], "3": []},
        )


def test_critique_report_schema_exposes_single_issue_limit():
    schema = CritiqueReport.model_json_schema()

    issue_list = schema["properties"]["candidate_issues"]["additionalProperties"]
    assert issue_list["maxItems"] == 1


def test_history_must_end_with_seeker_turn():
    with pytest.raises(ValidationError, match="end with a seeker"):
        History(
            turns=[
                DialogueTurn(role="seeker", content="我有点难过"),
                DialogueTurn(role="supporter", content="发生了什么？"),
            ]
        )


def test_expert_rejects_fields_from_another_domain():
    with pytest.raises(ValidationError, match="not allowed for emotion"):
        ExpertOutput(
            expert="emotion",
            fields={"emotion": "失落", "relationship_pattern": "疏远"},
            evidence=["我有点难过"],
            uncertainties=[],
        )


def test_intervention_function_is_only_state_or_plan():
    with pytest.raises(ValidationError):
        InterventionRecord(
            example_id="e-1",
            function="CRITIC",
            mutation=Mutation(
                operation="mask_state_field",
                field="primary_need",
                before="被理解",
                after="<MASKED>",
            ),
            full_response="完整回复",
            counterfactual_response="退化回复",
            target_dimension="specificity",
            localized_degradation=True,
            bidirectional_verified=True,
        )


def test_state_counterfactual_has_replacement_only_contract():
    value = StateCounterfactual(replacement="ready to take immediate action")

    assert value.model_dump() == {"replacement": "ready to take immediate action"}
    with pytest.raises(ValidationError):
        StateCounterfactual(replacement="different value", rationale="not allowed")


def test_margin_pair_rejects_safety_as_training_dimension():
    with pytest.raises(ValidationError):
        MarginPair(
            example_id="e-1",
            prompt="用户历史",
            chosen="更好的回复",
            rejected_candidate_id="c2",
            rejected="较差回复",
            defect_dimension="safety",
            defect_evidence="存在安全问题",
            order_swap_verified=True,
            safety_filter_passed=True,
        )


def test_margin_pair_requires_teacher_safety_filter_to_pass():
    with pytest.raises(ValidationError, match="safety-filter-passing"):
        MarginPair(
            example_id="e-1",
            prompt="用户历史",
            chosen="更好的回复",
            chosen_strategy_uses=[
                StrategyUse(
                    strategy_id="S1",
                    strategy="Question",
                    contribution="Invites reflection.",
                )
            ],
            rejected_candidate_id="c2",
            rejected_strategy_id="S2",
            rejected_strategy="Information",
            rejected="较差回复",
            defect_dimension="timing",
            defect_evidence="建议出现过早",
            order_swap_verified=True,
            safety_filter_passed=False,
        )
