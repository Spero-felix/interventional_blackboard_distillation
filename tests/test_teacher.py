import json

import pytest

from conftest import ScriptedBackend
from ibd.backend import LLMResult
from ibd.config import ModelConfig
from ibd.prompting import PROMPT_ROLES, build_messages
from ibd.schemas import Candidate, FinalSelectionDecision
from ibd.teacher import NORMAL_ROLES, TeacherRunner, _PlainTextCandidateBackend


class FixedTextBackend:
    def __init__(self, text):
        self.text = text

    def complete(self, **kwargs):
        return LLMResult(text=self.text)


def candidate_adapter(text):
    return _PlainTextCandidateBackend(
        FixedTextBackend(text),
        {
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Question",
            "seed": 11,
        },
    )


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


def test_normal_teacher_uses_one_candidate_role_without_provider_seeds(
    history, app_config
):
    backend = ScriptedBackend()
    trace = TeacherRunner(backend, app_config).run("e-no-position-binding", history)

    candidate_calls = [call for call in backend.calls if call["role"] == "candidate"]
    assert len(candidate_calls) == 3
    assert [call["seed"] for call in candidate_calls] == [None, None, None]
    assert [item.seed for item in trace.candidates] == [None, None, None]
    assert [item.candidate_id for item in trace.candidates] == ["1", "2", "3"]
    assert [item.response for item in trace.candidates] == [
        "候选回复-1",
        "候选回复-2",
        "候选回复-3",
    ]


def test_teacher_trace_does_not_persist_non_cache_hashes(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-no-hashes", history)

    payload = trace.model_dump(mode="json")
    assert "teacher_protocol_hash" not in payload
    assert all("request_hash" not in record for record in payload["call_records"])


def test_candidate_prompt_contains_only_its_assigned_strategy(history):
    prompt = build_messages(
        "candidate",
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
        },
    )[0]["content"]
    assert "Question:" in prompt
    assert "Providing Suggestions:" not in prompt
    assert '"const": "Question"' in prompt
    assert '"seed"' not in prompt


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


def test_fixed_plan_response_uses_the_plan_as_an_authoritative_candidate_condition(
    history, app_config
):
    backend = ScriptedBackend()
    runner = TeacherRunner(backend, app_config)
    trace = runner.run("e-base", history)
    backend.calls.clear()

    response = runner.generate_response_under_fixed_plan(
        history,
        trace.state,
        trace.final_selection.to_plan_selection(),
        clamped_state_field="readiness",
        example_id="e-base:fixed-plan",
    )

    assert response == "候选回复-1"
    assert [call["role"] for call in backend.calls] == ["candidate"]
    system_prompt = backend.calls[0]["messages"][0]["content"]
    assert "Experimental PLAN Clamp" in system_prompt
    assert "must not replan" in system_prompt


def test_prompt_registry_contains_only_current_roles():
    assert set(PROMPT_ROLES) == {
        "multi_view_state_analyzer",
        "planner",
        "candidate",
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
            if role == "candidate":
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
            "candidate": ModelConfig(model="fake", provider_json_mode=False)
        },
    )
    trace = TeacherRunner(PlainBackend(), config).run("e-plain", history)
    assert [item.response for item in trace.candidates] == ["一句自然回复"] * 3


def test_plain_text_candidate_adapter_unwraps_one_valid_json_fence():
    payload = {
        "candidate_id": "1",
        "strategy_id": "S1",
        "strategy": "Question",
        "response": "inner response",
        "seed": 11,
        "response_goal": "clarify the next step",
        "response_act": "ask one focused question",
    }
    text = f"provider preamble\n```json\n{json.dumps(payload)}\n```"

    result = candidate_adapter(text).complete(
        role="candidate",
        messages=[],
        model_config=ModelConfig(model="fake", provider_json_mode=False),
    )

    assert Candidate.model_validate_json(result.text).response == "inner response"


def test_plain_text_candidate_adapter_does_not_wrap_malformed_fence():
    raw = "```json\n{broken\n```"

    result = candidate_adapter(raw).complete(
        role="candidate",
        messages=[],
        model_config=ModelConfig(model="fake", provider_json_mode=False),
    )

    assert result.text == raw


def test_malformed_candidate_fence_triggers_schema_retry(history):
    from ibd.config import AppConfig

    class MalformedOnceBackend(ScriptedBackend):
        malformed_sent = False

        def complete(self, *, role, messages, model_config, json_mode=True, seed=None):
            if role == "candidate" and not self.malformed_sent:
                self.malformed_sent = True
                return LLMResult(text="```json\n{broken\n```")
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
            "candidate": ModelConfig(model="fake", provider_json_mode=False),
        },
    )

    trace = TeacherRunner(MalformedOnceBackend(), config).run("e-retry-fence", history)
    candidate_records = [
        record for record in trace.call_records if record.role == "candidate"
    ]

    assert len(candidate_records) == 4
    assert candidate_records[0].parsed == {}
    assert candidate_records[1].schema_retry is True
    assert all("```" not in candidate.response for candidate in trace.candidates)


def test_selector_presentation_is_deterministic_and_not_position_fixed(
    history, app_config
):
    def selector_order_for(example_id):
        backend = ScriptedBackend()
        TeacherRunner(backend, app_config).run(example_id, history)
        selector_call = next(
            call for call in backend.calls if call["role"] == "final_selector"
        )
        payload = json.loads(selector_call["messages"][1]["content"])
        return [
            candidate["candidate_id"]
            for candidate in payload["context"]["candidates"]
        ]

    first = selector_order_for("stable-example")
    assert selector_order_for("stable-example") == first
    observed = {position: set() for position in range(3)}
    for index in range(120):
        for position, candidate_id in enumerate(selector_order_for(f"e-{index}")):
            observed[position].add(candidate_id)

    assert all(ids == {"1", "2", "3"} for ids in observed.values())

    runner = TeacherRunner(ScriptedBackend(), app_config)
    trace = runner.run("distribution-template", history)
    counts = [dict.fromkeys(("1", "2", "3"), 0) for _ in range(3)]
    for index in range(2000):
        ordered = runner._selector_candidates(trace.candidates, f"sample-{index}")
        for position, candidate in enumerate(ordered):
            counts[position][candidate.candidate_id] += 1

    assert all(
        0.30 <= count / 2000 <= 0.37
        for position_counts in counts
        for count in position_counts.values()
    )
