import json
from pathlib import Path

import pytest

from conftest import ScriptedBackend, VALID_STATE
from ibd.backend import LLMResult
from ibd.config import ModelConfig
from ibd.prompting import ESCONV_STRATEGIES, PROMPT_ROLES, build_messages
from ibd.seeding import SeedDeriver
from ibd.schemas import (
    Candidate,
    ContextPatch,
    FinalSelectionDecision,
    History,
    MultiViewStateAnalysis,
    StrategyPlanSet,
    TeacherTrace,
    UserContext,
)
from ibd.teacher import (
    NORMAL_ROLES,
    TeacherRunner,
    TeacherSession,
    _PlainTextCandidateBackend,
)


UNKNOWN_STATE_POLICY = (
    "unknown means no usable evidence for that field. It must not be "
    "interpreted as any direction, preference, boundary, or default support "
    "action, and it must not count for or against a strategy or candidate."
)


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


class ContextAwareScriptedBackend(ScriptedBackend):
    def __init__(self, updater_patch: ContextPatch) -> None:
        super().__init__()
        self.updater_patch = updater_patch

    def _payload(self, role: str, seed: int | None):
        if role == "context_updater":
            return self.updater_patch.model_dump(mode="json")
        return super()._payload(role, seed)

    def user_payloads(self, role: str) -> list[dict[str, object]]:
        return [
            json.loads(call["messages"][1]["content"])
            for call in self.calls
            if call["role"] == role
        ]


class InvalidUpdaterThenValidTeacherBackend(ScriptedBackend):
    def complete(
        self,
        *,
        role,
        messages,
        model_config,
        json_mode=True,
        seed=None,
    ):
        if role != "context_updater":
            return super().complete(
                role=role,
                messages=messages,
                model_config=model_config,
                json_mode=json_mode,
                seed=seed,
            )
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "model": model_config.model,
                "json_mode": json_mode,
                "seed": seed,
            }
        )
        return LLMResult(text="{", usage={"total_tokens": 1})


class SequentialContextBackend(ContextAwareScriptedBackend):
    def __init__(self) -> None:
        super().__init__(ContextPatch.empty())
        self.update_index = 0

    def _payload(self, role: str, seed: int | None):
        if role != "context_updater":
            return ScriptedBackend._payload(self, role, seed)

        self.update_index += 1
        payload = ContextPatch.empty().model_dump(mode="json")
        if self.update_index == 1:
            payload["add"]["active_concerns"] = ["担心答辩"]
        else:
            payload["replace"]["active_concerns"] = [
                {"old": "担心答辩", "new": "担心明天答辩卡住"}
            ]
        return payload


def test_normal_teacher_path_updates_context_then_runs_response_pipeline(
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


def test_observe_and_respond_rebuild_the_legacy_run_trace(history, app_config):
    split_backend = ScriptedBackend()
    split_runner = TeacherRunner(split_backend, app_config)

    observation = split_runner.observe("e-split", history)
    response = split_runner.respond(observation)
    rebuilt = TeacherTrace.from_parts(observation, response, split="train")
    legacy = TeacherRunner(ScriptedBackend(), app_config).run("e-split", history)

    assert rebuilt == legacy
    assert [call["role"] for call in split_backend.calls] == list(NORMAL_ROLES)


def test_observe_and_respond_own_disjoint_teacher_stages(history, app_config):
    backend = ScriptedBackend()
    runner = TeacherRunner(backend, app_config)

    observation = runner.observe("e-stages", history)
    assert [call["role"] for call in backend.calls] == [
        "context_updater",
        "multi_view_state_analyzer",
    ]

    runner.respond(observation)
    assert [call["role"] for call in backend.calls[2:]] == [
        "planner",
        "candidate",
        "candidate",
        "candidate",
        "final_selector",
    ]


def test_teacher_session_commits_context_only_after_response_succeeds(
    history,
    app_config,
):
    patch_payload = ContextPatch.empty().model_dump(mode="json")
    patch_payload["add"]["active_concerns"] = ["担心答辩"]

    class TogglePlannerBackend(ContextAwareScriptedBackend):
        fail_planner = True

        def _payload(self, role, seed):
            if role == "planner" and self.fail_planner:
                return {"strategies": []}
            return super()._payload(role, seed)

    backend = TogglePlannerBackend(ContextPatch.model_validate(patch_payload))
    session = TeacherSession(TeacherRunner(backend, app_config))
    observation = session.observe("e-transaction", history)

    with pytest.raises(ValueError, match="planner failed schema validation"):
        session.respond(observation)
    assert session.context == UserContext.empty()

    backend.fail_planner = False
    trace = session.respond(observation)
    assert session.context == observation.context_after
    assert trace.context_after.active_concerns == ["担心答辩"]


def test_split_teacher_uses_stable_role_and_candidate_seeds(history, app_config):
    backend = ScriptedBackend()
    runner = TeacherRunner(backend, app_config)
    seeds = SeedDeriver("protocol-v1", "profile-1-seed-42", 42, 1)

    observation = runner.observe("profile-1-seed-42:round:1", history, seed_deriver=seeds)
    runner.respond(observation, seed_deriver=seeds)

    calls_by_role = {}
    for call in backend.calls:
        calls_by_role.setdefault(call["role"], []).append(call["seed"])
    assert calls_by_role["context_updater"] == [seeds.for_call("context_updater")]
    assert calls_by_role["multi_view_state_analyzer"] == [
        seeds.for_call("multi_view_state_analyzer")
    ]
    assert calls_by_role["planner"] == [seeds.for_call("planner")]
    assert calls_by_role["candidate"] == [
        seeds.for_call("candidate", "S1"),
        seeds.for_call("candidate", "S2"),
        seeds.for_call("candidate", "S3"),
    ]
    assert calls_by_role["final_selector"] == [seeds.for_call("final_selector")]


def test_teacher_updates_context_before_state_and_passes_it_downstream(
    history,
    app_config,
):
    patch_payload = ContextPatch.empty().model_dump()
    patch_payload["add"]["support_preferences_and_boundaries"] = [
        "当前不想听长期建议"
    ]
    backend = ContextAwareScriptedBackend(
        ContextPatch.model_validate(patch_payload)
    )
    trace = TeacherRunner(backend, app_config).run(
        "e-context", history, split="train"
    )

    assert [call["role"] for call in backend.calls] == list(NORMAL_ROLES)
    assert trace.context_before == UserContext.empty()
    assert trace.context_after.support_preferences_and_boundaries == [
        "当前不想听长期建议"
    ]
    downstream_payloads = [
        json.loads(call["messages"][1]["content"])
        for call in backend.calls
        if call["role"]
        in {
            "multi_view_state_analyzer",
            "planner",
            "candidate",
            "final_selector",
        }
    ]
    assert len(downstream_payloads) == 6
    assert all(
        payload["context"]["user_context"] == trace.context_after.model_dump()
        for payload in downstream_payloads
    )


def test_context_updater_sees_only_last_supporter_and_latest_seeker_turn(
    history,
    app_config,
):
    backend = ContextAwareScriptedBackend(ContextPatch.empty())

    TeacherRunner(backend, app_config).run("e-window", history)

    updater_history = backend.user_payloads("context_updater")[0]["history"]["turns"]
    assert updater_history == history.model_dump()["turns"][-2:]
    assert updater_history[-1]["role"] == "seeker"


def test_context_updater_omits_an_unpaired_previous_seeker(app_config):
    history = History.model_validate(
        {
            "turns": [
                {"role": "seeker", "content": "昨天我也很担心。"},
                {"role": "seeker", "content": "现在我只想说今天的事。"},
            ]
        }
    )
    backend = ContextAwareScriptedBackend(ContextPatch.empty())

    TeacherRunner(backend, app_config).run("e-unpaired", history)

    assert backend.user_payloads("context_updater")[0]["history"]["turns"] == [
        history.model_dump()["turns"][-1]
    ]


def test_first_seeker_turn_is_a_valid_context_update_window(app_config):
    history = History.model_validate(
        {"turns": [{"role": "seeker", "content": "我今天完全睡不着。"}]}
    )
    backend = ContextAwareScriptedBackend(ContextPatch.empty())

    TeacherRunner(backend, app_config).run("e-first", history)

    assert backend.user_payloads("context_updater")[0]["history"] == history.model_dump()


def test_context_updater_failure_preserves_previous_context_and_continues(
    history,
    app_config,
):
    previous_payload = UserContext.empty().model_dump()
    previous_payload["active_concerns"] = ["担心答辩"]
    previous = UserContext.model_validate(previous_payload)
    backend = InvalidUpdaterThenValidTeacherBackend()

    trace = TeacherRunner(backend, app_config).run(
        "e-invalid-updater",
        history,
        context_before=previous,
    )

    assert trace.context_patch == ContextPatch.empty()
    assert trace.context_after == previous
    assert trace.context_merge_errors == [
        "context_updater_failed: context_updater failed schema validation"
    ]
    assert trace.final_response


def test_teacher_session_carries_context_between_seeker_turns(app_config):
    backend = SequentialContextBackend()
    session = TeacherSession(TeacherRunner(backend, app_config))
    first_history = History.model_validate(
        {"turns": [{"role": "seeker", "content": "我有点担心答辩。"}]}
    )
    second_history = History.model_validate(
        {
            "turns": [
                {"role": "seeker", "content": "我有点担心答辩。"},
                {"role": "supporter", "content": "最担心哪个部分？"},
                {
                    "role": "seeker",
                    "content": "更准确地说，我担心明天答辩时卡住。",
                },
            ]
        }
    )

    first = session.run("turn-1", first_history)
    second = session.run("turn-2", second_history)

    assert first.context_before == UserContext.empty()
    assert second.context_before == first.context_after
    assert second.context_after.active_concerns == ["担心明天答辩卡住"]
    assert session.context == second.context_after
    assert backend.user_payloads("context_updater")[1]["context"][
        "previous_context"
    ] == first.context_after.model_dump()


def test_legacy_teacher_trace_defaults_to_empty_context_audit(
    history,
    app_config,
):
    payload = TeacherRunner(ScriptedBackend(), app_config).run(
        "legacy", history
    ).model_dump(mode="json")
    for field in (
        "context_before",
        "context_patch",
        "context_after",
        "context_merge_errors",
    ):
        payload.pop(field)

    trace = TeacherTrace.model_validate(payload)

    assert trace.context_before == UserContext.empty()
    assert trace.context_patch == ContextPatch.empty()
    assert trace.context_after == UserContext.empty()
    assert trace.context_merge_errors == []


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
            "state": VALID_STATE,
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Question",
        },
    )[0]["content"]
    assert "Question:" in prompt
    assert "Providing Suggestions:" not in prompt
    assert '"const": "Question"' in prompt
    assert '"seed"' not in prompt


def test_analyzer_prompt_uses_seven_state_guide(history):
    prompt = build_messages(
        "multi_view_state_analyzer", history, MultiViewStateAnalysis
    )[0]["content"]

    assert "seven-dimensional user-state analyzer" in prompt
    assert "advice_receptivity" in prompt
    assert "unknown as a middle or low value" in prompt
    assert "at most 20 words" not in prompt


def test_context_updater_prompt_receives_previous_context_and_recent_turns():
    recent = History.model_validate(
        {
            "turns": [
                {"role": "supporter", "content": "你愿意说说最担心什么吗？"},
                {
                    "role": "seeker",
                    "content": "我只想先把今晚熬过去，不想听长期建议。",
                },
            ]
        }
    )
    messages = build_messages(
        "context_updater",
        recent,
        ContextPatch,
        context={"previous_context": UserContext.empty()},
    )
    payload = json.loads(messages[1]["content"])

    assert payload["history"] == recent.model_dump()
    assert payload["context"]["previous_context"] == UserContext.empty().model_dump()
    assert "Only the latest seeker turn may change User Context" in messages[0]["content"]
    assert (
        "Do not infer willingness, readiness, diagnosis, or personality"
        in messages[0]["content"]
    )


def test_state_and_response_roles_receive_same_context_policy(history):
    user_context = UserContext.empty()
    role_inputs = (
        (
            "multi_view_state_analyzer",
            MultiViewStateAnalysis,
            {"user_context": user_context},
        ),
        (
            "planner",
            StrategyPlanSet,
            {"user_context": user_context, "state": VALID_STATE},
        ),
        (
            "candidate",
            Candidate,
            {
                "user_context": user_context,
                "state": VALID_STATE,
                "candidate_id": "1",
                "strategy_id": "S1",
                "strategy": "Question",
            },
        ),
        (
            "final_selector",
            FinalSelectionDecision,
            {
                "user_context": user_context,
                "state": VALID_STATE,
                "candidates": [],
            },
        ),
    )

    for role, response_model, context in role_inputs:
        system = build_messages(
            role,
            history,
            response_model,
            context=context,
        )[0]["content"]
        assert "User Context is factual background" in system
        assert "the latest explicit seeker message takes priority" in system


@pytest.mark.parametrize(
    ("role", "response_model", "context"),
    [
        ("multi_view_state_analyzer", MultiViewStateAnalysis, {}),
        ("planner", StrategyPlanSet, {"state": VALID_STATE}),
        (
            "candidate",
            Candidate,
            {
                "state": VALID_STATE,
                "candidate_id": "1",
                "strategy_id": "S1",
                "strategy": "Question",
            },
        ),
        (
            "final_selector",
            FinalSelectionDecision,
            {"state": VALID_STATE, "candidates": []},
        ),
    ],
)
def test_state_consuming_prompts_include_unknown_state_policy(
    history, role, response_model, context
):
    prompt = build_messages(role, history, response_model, context=context)[0]["content"]

    assert UNKNOWN_STATE_POLICY in prompt


def test_planner_prompt_allows_only_meaningful_one_to_three_options(history):
    prompt = build_messages(
        "planner",
        history,
        StrategyPlanSet,
        context={"state": VALID_STATE},
    )[0]["content"]

    assert "between one and three" in prompt
    assert "return only Others" in prompt
    assert "merely to increase the number" in prompt
    for strategy in ESCONV_STRATEGIES:
        assert f"{strategy}:" in prompt


def test_others_candidate_prompt_uses_only_the_positive_assigned_definition(history):
    prompt = build_messages(
        "candidate",
        history,
        Candidate,
        context={
            "state": {},
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Others",
        },
    )[0]["content"]

    assert "# Assigned Strategy" in prompt
    assert "Others:" in prompt
    assert "interaction-management function is primary" in prompt
    assert "more specific strategy instead" in prompt
    assert '"const": "Others"' in prompt
    assert "Providing Suggestions:" not in prompt


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


def test_self_disclosure_allows_only_low_risk_synthetic_experience(history):
    prompt = build_messages(
        "candidate",
        history,
        Candidate,
        context={
            "state": VALID_STATE,
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Self-disclosure",
        },
    )[0]["content"]

    assert "natural, contextually appropriate" in prompt
    assert "differs functionally from other candidates" not in prompt
    assert "brief, generic, low-risk synthetic supporter experience" in prompt
    assert "I went through something similar" in prompt
    assert "professional qualifications" in prompt
    assert "self-harm, suicide, abuse, crime, or severe trauma" in prompt
    assert "Do not invent facts about the seeker" in prompt
    assert "Do not claim personal history" not in prompt


def test_selector_does_not_treat_compliant_synthetic_disclosure_as_unsupported(
    history,
):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": VALID_STATE, "candidates": []},
    )[0]["content"]

    assert "permitted generic synthetic supporter experience" in prompt
    assert "solely because it is not factually verifiable" in prompt
    assert "does not receive preference" in prompt


def test_final_selector_uses_conditional_fit_then_quality(history):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": VALID_STATE, "candidates": []},
    )[0]["content"]

    assert "Stage 1 — Conditional fit" in prompt
    assert "Stage 2 — Response quality" in prompt
    assert "STATE.primary_need" not in prompt
    assert "concrete, autonomy-preserving" not in prompt


def test_final_selector_prompt_gates_others_by_primary_function(history):
    prompt = build_messages(
        "final_selector",
        history,
        FinalSelectionDecision,
        context={"state": {}, "candidates": []},
    )[0]["content"]

    assert "actual response" in prompt
    assert "strategy name as selection evidence" in prompt


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


def test_prompt_registry_contains_only_current_roles():
    assert set(PROMPT_ROLES) == {
        "context_updater",
        "multi_view_state_analyzer",
        "planner",
        "candidate",
        "final_selector",
    }


def test_teacher_config_uses_seeker_context_protocol():
    from ibd.config import AppConfig

    repository = Path(__file__).parents[1]
    for relative_path in ("configs/deepseek_teacher.yaml",):
        config = AppConfig.from_yaml(repository / relative_path)
        assert (
            config.protocol_version
            == "qwen25-socialsim-seven-state-context-v1"
        )


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
    candidate_records = [
        record for record in trace.call_records if record.role == "candidate"
    ]
    assert len(candidate_records) == 6
    assert [record.metadata_fallback for record in candidate_records] == [
        False,
        True,
    ] * 3


@pytest.mark.parametrize("count", [1, 2, 3])
def test_teacher_generates_one_candidate_per_planned_strategy(
    history, app_config, count
):
    class VariablePlanBackend(ScriptedBackend):
        def _payload(self, role, seed):
            if role == "planner":
                return {
                    "strategies": [
                        "Question",
                        "Reflection of feelings",
                        "Information",
                    ][:count]
                }
            if role == "final_selector":
                return {
                    "selected_candidate_id": "1",
                    "response_goal": "support the immediate goal",
                    "response_act": "apply the selected strategy",
                }
            return super()._payload(role, seed)

    trace = TeacherRunner(VariablePlanBackend(), app_config).run(
        f"e-cardinality-{count}", history
    )

    assert len(trace.plan.strategies) == count
    assert len(trace.candidates) == count
    assert [candidate.candidate_id for candidate in trace.candidates] == [
        str(index) for index in range(1, count + 1)
    ]


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
