import pytest

from conftest import ScriptedBackend
from ibd.export import sft_row, slot_row, visible_sft_row
from ibd.teacher import TeacherRunner


def test_sft_export_contains_natural_response_and_one_strategy(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-sft", history)
    assert sft_row(trace) == {
        "example_id": "e-sft",
        "prompt": trace.history.as_prompt(),
        "response": trace.final_response,
        "selected_strategy": trace.final_selection.selected_strategy,
    }


def test_slot_export_contains_only_state_and_final_single_plan(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-slot", history)
    row = slot_row(trace)
    assert set(row) == {"example_id", "prompt", "state", "plan"}
    assert row["plan"]["strategies"] == [trace.final_selection.selected_strategy]
    assert "views" not in row


def test_visible_sft_target_preserves_state_strategy_and_response(history, app_config):
    from ibd.visible_sft import parse_visible_sft_response, serialize_visible_sft

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-visible", history)

    assert serialize_visible_sft(trace) == (
        f"[dominant_emotion]{trace.state.dominant_emotion}"
        f"[distress_level]{trace.state.distress_level}"
        f"[primary_support_need]{trace.state.primary_support_need}"
        f"[advice_receptivity]{trace.state.advice_receptivity}"
        f"[action_intent]{trace.state.action_intent}"
        f"[action_capacity]{trace.state.action_capacity}"
        f"[continuation_intent]{trace.state.continuation_intent}"
        f"[selected_strategy]{trace.final_selection.selected_strategy}"
        f"[response]{trace.final_response}"
    )
    assert parse_visible_sft_response(serialize_visible_sft(trace)) == trace.final_response


def test_visible_sft_response_parser_rejects_missing_or_empty_marker():
    from ibd.visible_sft import parse_visible_sft_response

    with pytest.raises(ValueError, match=r"\[response\]"):
        parse_visible_sft_response("[dominant_emotion]sadness_loss")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_visible_sft_response("[response]")


def test_visible_sft_export_keeps_native_history_and_split(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-visible-row", history, split="dev"
    )

    row = visible_sft_row(trace)

    assert set(row) == {"example_id", "split", "history", "response"}
    assert row["example_id"] == "e-visible-row"
    assert row["split"] == "dev"
    assert row["history"] == history.model_dump(mode="json")
    assert row["response"].endswith("[response]" + trace.final_response)


def test_student_exports_do_not_leak_context_audit_fields(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-no-context", history)
    rows = (sft_row(trace), slot_row(trace), visible_sft_row(trace))

    assert set(rows[0]) == {
        "example_id",
        "prompt",
        "response",
        "selected_strategy",
    }
    assert set(rows[1]) == {"example_id", "prompt", "state", "plan"}
    assert set(rows[2]) == {"example_id", "split", "history", "response"}
    for row in rows:
        assert "context_before" not in row
        assert "context_patch" not in row
        assert "context_after" not in row
        assert "context_merge_errors" not in row


@pytest.mark.parametrize("split", ["test", "diagnostic_holdout"])
def test_student_export_rejects_non_trainable_splits(history, app_config, split):
    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        f"e-{split}", history, split=split
    )
    with pytest.raises(ValueError):
        sft_row(trace)
