import pytest

from conftest import ScriptedBackend

from ibd.schemas import InterventionRecord, MarginPair, StrategyUse
from ibd.teacher import TeacherRunner


def test_margin_export_uses_an_exact_four_field_allowlist():
    from ibd.export import margin_row

    pair = MarginPair(
        example_id="e-1",
        prompt="seeker: 我不知道怎么开口",
        chosen="我们可以先想一句温和的开场。",
        chosen_strategy_uses=[
            StrategyUse(
                strategy_id="S1",
                strategy="Providing Suggestions",
                contribution="Offers an optional opening.",
            )
        ],
        rejected_candidate_id="2",
        rejected_strategy_id="S2",
        rejected_strategy="Question",
        rejected="你就直接说。",
        defect_dimension="timing",
        defect_evidence="TEACHER_ONLY_SAFETY_AUDIT_TEXT",
        order_swap_verified=True,
        safety_filter_passed=True,
    )

    row = margin_row(pair)

    assert row == {
        "example_id": "e-1",
        "prompt": "seeker: 我不知道怎么开口",
        "chosen": "我们可以先想一句温和的开场。",
        "rejected": "你就直接说。",
        "chosen_strategy_uses": [
            {
                "strategy_id": "S1",
                "strategy": "Providing Suggestions",
                "contribution": "Offers an optional opening.",
            }
        ],
        "rejected_strategy_id": "S2",
        "rejected_strategy": "Question",
    }
    assert "TEACHER_ONLY" not in str(row)


def test_slot_export_contains_state_and_plan_but_no_critic_or_gate(history, app_config):
    from ibd.export import slot_row

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-slot", history)
    row = slot_row(trace)

    assert set(row) == {"example_id", "prompt", "state", "plan"}
    assert "critiques" not in row
    assert "quality_gate" not in row
    assert "safety" not in str(row).lower()
    assert row["plan"]["strategy_uses"] == trace.final_answer.model_dump(mode="json")[
        "strategy_uses"
    ]


def test_student_export_rejects_test_split_trace(history, app_config):
    from ibd.export import sft_row

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-test", history, split="test")

    with pytest.raises(ValueError, match="test split"):
        sft_row(trace)


def test_student_export_rejects_diagnostic_holdout_trace(history, app_config):
    from ibd.export import sft_row

    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-holdout", history, split="diagnostic_holdout"
    )

    with pytest.raises(ValueError, match="diagnostic holdout"):
        sft_row(trace)


def test_sft_export_records_strategy_uses_outside_the_visible_prompt(history, app_config):
    from ibd.export import sft_row

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-strategy", history)
    row = sft_row(trace)

    assert row["strategy_uses"] == trace.final_answer.model_dump(mode="json")[
        "strategy_uses"
    ]
    assert "strategy_uses" not in row["prompt"]


def test_intervention_export_contains_only_student_visible_effect_fields(history, app_config):
    from ibd.export import intervention_row
    from ibd.interventions import mask_state_field

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-int", history)
    mutated_state, mutation = mask_state_field(trace.state, "emotion")
    record = InterventionRecord(
        example_id="e-int",
        function="STATE",
        mutation=mutation,
        mutated_state=mutated_state,
        full_response="具体承接",
        counterfactual_response="泛化承接",
        target_dimension="specificity",
        localized_degradation=True,
        bidirectional_verified=True,
    )

    row = intervention_row(record, "seeker: 我觉得被忽略")

    assert set(row) == {
        "example_id",
        "prompt",
        "function",
        "clamp",
        "full_response",
        "counterfactual_response",
        "target_dimension",
    }
    assert row["clamp"] == mutated_state.model_dump(mode="json")
    assert row["counterfactual_response"] == "泛化承接"


def test_intervention_reader_rejects_legacy_ablated_response(history, app_config):
    from pydantic import ValidationError

    from ibd.interventions import mask_state_field

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-legacy", history)
    mutated_state, mutation = mask_state_field(trace.state, "emotion")
    stale = {
        "example_id": "e-legacy",
        "function": "STATE",
        "mutation": mutation.model_dump(mode="json"),
        "mutated_state": mutated_state.model_dump(mode="json"),
        "mutated_plan": None,
        "full_response": "full",
        "ablated_response": "legacy",
        "target_dimension": "emotion",
        "localized_degradation": True,
        "bidirectional_verified": True,
    }

    with pytest.raises(ValidationError):
        InterventionRecord.model_validate(stale)
