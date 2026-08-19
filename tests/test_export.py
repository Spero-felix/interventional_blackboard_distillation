import pytest

from conftest import ScriptedBackend

from ibd.schemas import InterventionRecord, MarginPair, Mutation
from ibd.teacher import TeacherRunner


def test_margin_export_uses_an_exact_four_field_allowlist():
    from ibd.export import margin_row

    pair = MarginPair(
        example_id="e-1",
        prompt="seeker: 我不知道怎么开口",
        chosen="我们可以先想一句温和的开场。",
        rejected_candidate_id="2",
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


def test_student_export_rejects_test_split_trace(history, app_config):
    from ibd.export import sft_row

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-test", history, split="test")

    with pytest.raises(ValueError, match="test split"):
        sft_row(trace)


def test_intervention_export_contains_only_student_visible_effect_fields():
    from ibd.export import intervention_row

    record = InterventionRecord(
        example_id="e-int",
        function="STATE",
        mutation=Mutation(
            function="STATE",
            operation="downgrade",
            field="needs.primary",
            before="被理解",
            after="uncertain",
        ),
        full_response="具体承接",
        ablated_response="泛化承接",
        target_dimension="specificity",
        localized_degradation=True,
        order_swap_verified=True,
    )

    row = intervention_row(record, "seeker: 我觉得被忽略")

    assert set(row) == {
        "example_id",
        "prompt",
        "function",
        "clamp",
        "full_response",
        "intervened_response",
        "target_dimension",
    }
