import json

import pytest

from conftest import ScriptedBackend
from ibd.backend import LLMResult
from ibd.prompting import PROMPT_ROLES, build_messages
from ibd.schemas import Candidate, FinalSelectionDecision
from ibd.teacher import NORMAL_ROLES, TeacherRunner


def test_normal_teacher_path_is_one_analyzer_three_candidates_one_selector(
    history, app_config
):
    backend = ScriptedBackend()
    trace = TeacherRunner(backend, app_config).run("e-1", history)

    roles = [call["role"] for call in backend.calls]
    assert roles == list(NORMAL_ROLES)
    assert trace.state == trace.state_analysis.state
    assert len(trace.plan.strategies) == 3
    assert len(trace.candidates) == 3
    assert trace.final_selection.selected_candidate_id == "2"
    assert trace.final_response == trace.candidates[1].response
    assert not any(role.endswith("_critic") for role in roles)


def test_teacher_trace_does_not_persist_non_cache_hashes(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-no-hashes", history)

    payload = trace.model_dump(mode="json")
    assert "teacher_protocol_hash" not in payload
    assert all("request_hash" not in record for record in payload["call_records"])


def test_candidate_prompt_contains_only_its_assigned_strategy(history):
    prompt = build_messages(
        "candidate_1",
        history,
        Candidate,
        context={
            "state": {
                "emotion": "失落",
                "intensity": "中等",
                "primary_need": "被理解",
                "support_goal": "准备沟通",
                "readiness": "愿意探索",
                "main_constraint": "担心回避",
                "relationship_context": "亲密关系",
            },
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Question",
            "seed": 11,
        },
    )[0]["content"]
    assert "Question:" in prompt
    assert "Providing Suggestions:" not in prompt
    assert '"const": "Question"' in prompt


def test_final_selector_schema_cannot_contain_rewritten_response(history):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": {}, "candidates": []},
    )[0]["content"]
    schema = json.loads(prompt.split("JSON Schema:\n", 1)[1])
    assert set(schema["properties"]) == {
        "selected_candidate_id",
        "response_goal",
        "response_act",
    }


def test_final_selector_prompt_uses_ordered_response_quality_policy(history):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": {}, "candidates": []},
    )[0]["content"]

    assert "actual candidate response" in prompt
    assert "Direct fit to the seeker's latest turn" in prompt
    assert "Emotional attunement and respect for the seeker's autonomy" in prompt
    assert "concrete, autonomy-preserving response" in prompt
    assert "Do not favor a candidate because of its ID, order, or strategy name." in prompt
    assert "must accurately describe the selected response" in prompt


def test_selector_response_is_taken_verbatim_from_candidate(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-verbatim", history)
    selected = next(
        item
        for item in trace.candidates
        if item.candidate_id == trace.final_selection.selected_candidate_id
    )
    assert trace.final_response == selected.response


def test_selector_rejects_unknown_candidate_id(history, app_config):
    class UnknownIdBackend(ScriptedBackend):
        def _payload(self, role, seed):
            if role == "final_selector":
                return {
                    "selected_candidate_id": "99",
                    "response_goal": "目标",
                    "response_act": "动作",
                }
            return super()._payload(role, seed)

    with pytest.raises(ValueError, match="exactly one supplied candidate"):
        TeacherRunner(UnknownIdBackend(), app_config).run("e-bad", history)


def test_state_counterfactual_generator_is_a_single_local_call(history, app_config):
    backend = ScriptedBackend()
    runner = TeacherRunner(backend, app_config)
    trace = runner.run("e-base", history)
    backend.calls.clear()
    replacement = runner.generate_state_counterfactual(
        history,
        trace.state,
        "readiness",
        "timing",
        example_id="e-base:cf",
    )
    assert replacement == "准备立即采取具体行动"
    assert [call["role"] for call in backend.calls] == [
        "state_counterfactual_generator"
    ]


def test_prompt_registry_contains_only_current_roles():
    assert set(PROMPT_ROLES) == {
        "multi_view_state_analyzer",
        "planner",
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "final_selector",
        "state_counterfactual_generator",
        "condition_effect_verifier",
        "state_effect_verifier",
        "safety_verifier",
    }


def test_plain_text_candidate_provider_is_wrapped(history):
    from ibd.config import AppConfig, ModelConfig

    class PlainBackend(ScriptedBackend):
        def complete(self, *, role, messages, model_config, json_mode=True, seed=None):
            if role.startswith("candidate_"):
                self.calls.append({"role": role, "messages": messages})
                return LLMResult(text="一句自然回复", usage={})
            return super().complete(
                role=role,
                messages=messages,
                model_config=model_config,
                json_mode=json_mode,
                seed=seed,
            )

    config = AppConfig(
        default_model=ModelConfig(model="fake"),
        roles={
            f"candidate_{index}": ModelConfig(
                model="fake", provider_json_mode=False
            )
            for index in range(1, 4)
        },
    )
    trace = TeacherRunner(PlainBackend(), config).run("e-plain", history)
    assert [item.response for item in trace.candidates] == ["一句自然回复"] * 3
