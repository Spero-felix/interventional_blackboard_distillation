import json

import pytest

from conftest import ScriptedBackend

from ibd.teacher import NORMAL_ROLES, TeacherRunner


class SelectedStrategyBackend(ScriptedBackend):
    def __init__(self, *, reverse_final_strategies=False):
        super().__init__()
        self.reverse_final_strategies = reverse_final_strategies

    @staticmethod
    def _context(messages):
        return json.loads(messages[1]["content"])["context"]

    def complete(self, *, role, messages, model_config, json_mode=True, seed=None):
        from ibd.backend import LLMResult

        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "model": model_config.model,
                "json_mode": json_mode,
                "seed": seed,
            }
        )
        context = self._context(messages)
        if role.startswith("candidate_"):
            payload = {
                "candidate_id": context["candidate_id"],
                "strategy_id": context["strategy_id"],
                "strategy": context["strategy"],
                "response": f"generated {context['strategy']}",
                "seed": seed,
            }
        elif role.endswith("_critic"):
            payload = {
                "critic": role.removesuffix("_critic"),
                "candidate_issues": {
                    candidate["candidate_id"]: []
                    for candidate in context["candidates"]
                },
                "summary": "checked selected candidates",
            }
        elif role == "final_integrator":
            strategies = list(context["required_strategies"])
            if self.reverse_final_strategies:
                strategies.reverse()
            payload = {
                "response": "integrated selected candidates",
                "strategy_uses": [
                    {
                        "strategy_id": f"S{index}",
                        "strategy": strategy,
                        "contribution": "selected contribution",
                    }
                    for index, strategy in enumerate(strategies, start=1)
                ],
            }
        else:
            raise AssertionError(f"unexpected selected-strategy role: {role}")
        return LLMResult(text=json.dumps(payload), usage={"total_tokens": 10})


def _state():
    from ibd.schemas import StateBlackboard

    return StateBlackboard(
        emotion="sad",
        intensity="medium",
        primary_need="understanding",
        support_goal="prepare a conversation",
        readiness="willing to reflect",
        main_constraint="fear of avoidance",
        relationship_context="close relationship conflict",
    )


def _original_candidates():
    from ibd.schemas import Candidate

    return [
        Candidate(
            candidate_id=str(index),
            strategy_id=f"S{index}",
            strategy=strategy,
            response=f"original {strategy}",
            seed=seed,
        )
        for index, (strategy, seed) in enumerate(
            [
                ("Reflection of feelings", 11),
                ("Question", 29),
                ("Providing Suggestions", 47),
            ],
            start=1,
        )
    ]


def test_scripted_backend_emits_compact_state_payload(app_config):
    from ibd.schemas import STATE_ANCHOR_FIELDS, StateBlackboard

    result = ScriptedBackend().complete(
        role="state_integrator",
        messages=[],
        model_config=app_config.default_model,
    )
    state = StateBlackboard.model_validate_json(result.text)

    assert tuple(state.model_dump()) == STATE_ANCHOR_FIELDS


def test_scripted_backend_emits_compact_planner_payload(app_config):
    from ibd.schemas import StrategyPlanSet

    result = ScriptedBackend().complete(
        role="planner",
        messages=[],
        model_config=app_config.default_model,
    )
    plan = StrategyPlanSet.model_validate_json(result.text)

    assert plan.strategies == [
        "Reflection of feelings",
        "Question",
        "Providing Suggestions",
    ]


def test_strategy_plan_compatibility_export_keeps_legacy_audit_shape_separate():
    from ibd import StrategyPlan
    from ibd.schemas import StrategyPlanSet

    legacy = StrategyPlan(
        strategy_id="S1",
        strategy="Question",
        therapeutic_goal="Clarify the seeker's immediate need.",
        rationale="A focused question invites the seeker to choose the next step.",
        response_acts=["Ask which kind of support would help most."],
        tone="warm and curious",
        avoid=["pressure"],
    )
    compact = StrategyPlanSet(
        strategies=["Question", "Reflection of feelings", "Providing Suggestions"]
    )

    assert legacy.model_dump() == {
        "strategy_id": "S1",
        "strategy": "Question",
        "therapeutic_goal": "Clarify the seeker's immediate need.",
        "rationale": "A focused question invites the seeker to choose the next step.",
        "response_acts": ["Ask which kind of support would help most."],
        "tone": "warm and curious",
        "avoid": ["pressure"],
    }
    assert compact.model_dump() == {
        "strategies": [
            "Question",
            "Reflection of feelings",
            "Providing Suggestions",
        ]
    }


def test_normal_teacher_trace_has_exactly_thirteen_roles(history, app_config):
    backend = ScriptedBackend()
    trace = TeacherRunner(backend, app_config).run("e-1", history)

    assert tuple(call["role"] for call in backend.calls) == NORMAL_ROLES
    assert len(trace.call_records) == 13
    assert set(trace.expert_outputs) == {"emotion", "need", "relationship", "intent"}
    assert not any("risk" in call.role.lower() for call in trace.call_records)


def test_teacher_records_three_plans_and_final_strategy_provenance(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-2", history)

    assert trace.plan.strategies == [
        "Reflection of feelings",
        "Question",
        "Providing Suggestions",
    ]
    assert [item.strategy_id for item in trace.final_answer.strategy_uses] == ["S1", "S2"]
    assert trace.final_response == trace.final_answer.response


def test_each_expert_prompt_sees_history_but_no_peer_output(history, app_config):
    backend = ScriptedBackend()
    TeacherRunner(backend, app_config).run("e-3", history)

    expert_calls = backend.calls[:4]
    prompts = {call["role"]: str(call["messages"]) for call in expert_calls}
    assert all("最近总觉得被忽略" in prompt for prompt in prompts.values())
    assert "relationship_pattern" not in prompts["emotion_expert"]
    assert "decision_stage" not in prompts["need_expert"]


def test_state_counterfactual_generator_sends_complete_localized_prompt_contract(
    history, app_config
):
    from ibd.schemas import STATE_ANCHOR_FIELDS

    class CounterfactualBackend(ScriptedBackend):
        def _payload(self, role, seed):
            if role == "state_counterfactual_generator":
                return {"replacement": "ready to take immediate action"}
            return super()._payload(role, seed)

    backend = CounterfactualBackend()
    runner = TeacherRunner(backend, app_config)
    trace = runner.run("e-counterfactual-source", history)
    backend.calls.clear()

    replacement = runner.generate_state_counterfactual(
        history,
        trace.state,
        "readiness",
        "timing",
        example_id="e-counterfactual:intervention:STATE:readiness:counterfactual",
    )

    assert replacement == "ready to take immediate action"
    assert [call["role"] for call in backend.calls] == [
        "state_counterfactual_generator"
    ]
    messages = backend.calls[0]["messages"]
    system_prompt = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    context = payload["context"]

    assert payload["history"] == history.model_dump(mode="json")
    assert context == {
        "state": trace.state.model_dump(mode="json"),
        "target_field": "readiness",
        "target_dimension": "timing",
        "original_value": trace.state.readiness,
    }
    for field in STATE_ANCHOR_FIELDS:
        assert f"`{field}`" in system_prompt
    for required_concept in (
        "plausible",
        "contrastive",
        "less supported",
        "localized",
        "<MASKED>",
        "missingness",
        "simple negation",
        "strategy names",
        "final response wording",
        "diagnosis",
        "invented events",
        "explanation",
    ):
        assert required_concept in system_prompt
    schema = json.loads(system_prompt.split("\n\nJSON Schema:\n", 1)[1])
    assert set(schema["properties"]) == {"replacement"}
    assert set(schema["required"]) == {"replacement"}


def test_deepseek_teacher_configures_counterfactual_generator_deterministically():
    from ibd.config import AppConfig

    config = AppConfig.from_yaml("configs/deepseek_teacher.yaml")
    role = config.roles["state_counterfactual_generator"]

    assert role.model == "deepseek-v4-pro"
    assert role.temperature == 0.0
    assert role.max_tokens == 300
    assert role.provider_json_mode is True
    assert role.thinking_enabled is False


@pytest.mark.parametrize(
    ("role", "field_keys"),
    [
        ("emotion_expert", "emotion, intensity, trajectory, coping_signal"),
        ("need_expert", "need, priority, readiness, constraint"),
        (
            "relationship_expert",
            "relationship_type, relationship_pattern, power_dynamic, boundary_signal",
        ),
        ("intent_expert", "intent, explicit_ask, implicit_goal, decision_stage"),
    ],
)
def test_expert_prompt_names_exact_allowed_field_keys(history, role, field_keys):
    from ibd.prompting import build_messages
    from ibd.schemas import ExpertOutput

    messages = build_messages(role, history, ExpertOutput)

    assert f"fields keys: {field_keys}." in messages[0]["content"]


@pytest.mark.parametrize(
    "role",
    ["emotion_critic", "effectiveness_critic", "safety_critic"],
)
def test_critic_prompt_limits_each_candidate_to_one_concise_issue(history, role):
    from ibd.prompting import build_messages
    from ibd.schemas import CritiqueReport

    messages = build_messages(role, history, CritiqueReport)
    system_prompt = messages[0]["content"]

    assert "At most one highest-priority issue per candidate" in system_prompt
    assert "one concise sentence" in system_prompt


def test_candidate_prompt_freezes_candidate_id_and_seed(history, app_config):
    backend = ScriptedBackend()
    TeacherRunner(backend, app_config).run("e-seed", history)

    candidate_calls = [call for call in backend.calls if call["role"].startswith("candidate_")]
    contexts = [json.loads(call["messages"][1]["content"])["context"] for call in candidate_calls]

    assert [(item["candidate_id"], item["seed"]) for item in contexts] == [
        ("1", 11),
        ("2", 29),
        ("3", 47),
    ]
    assert [set(item) for item in contexts] == [
        {
            "state",
            "candidate_id",
            "strategy_id",
            "strategy",
            "strategy_definition",
            "seed",
        },
        {
            "state",
            "candidate_id",
            "strategy_id",
            "strategy",
            "strategy_definition",
            "seed",
        },
        {
            "state",
            "candidate_id",
            "strategy_id",
            "strategy",
            "strategy_definition",
            "seed",
        },
    ]
    assert [item["strategy"] for item in contexts] == [
        "Reflection of feelings",
        "Question",
        "Providing Suggestions",
    ]


def test_selected_strategies_reuses_candidates_without_planner_or_candidate_calls(
    history, app_config
):
    from ibd.schemas import PlanSelection

    backend = SelectedStrategyBackend()
    selection = PlanSelection(strategies=["Question", "Reflection of feelings"])

    result = TeacherRunner(backend, app_config).run_selected_strategies(
        example_id="selected-reuse",
        history=history,
        state=_state(),
        selection=selection,
        reusable_candidates=_original_candidates(),
    )

    assert [candidate.model_dump() for candidate in result.candidates] == [
        {
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Question",
            "response": "original Question",
            "seed": 11,
        },
        {
            "candidate_id": "2",
            "strategy_id": "S2",
            "strategy": "Reflection of feelings",
            "response": "original Reflection of feelings",
            "seed": 29,
        },
    ]
    assert [call["role"] for call in backend.calls] == [
        "emotion_critic",
        "effectiveness_critic",
        "safety_critic",
        "final_integrator",
    ]
    critic_contexts = [
        json.loads(call["messages"][1]["content"])["context"]
        for call in backend.calls[:3]
    ]
    assert all(context["state"] == _state().model_dump() for context in critic_contexts)
    assert [item.strategy for item in result.final_answer.strategy_uses] == selection.strategies
    assert len(result.call_records) == 4


def test_selected_strategies_generates_only_missing_category(history, app_config):
    from ibd.schemas import PlanSelection

    backend = SelectedStrategyBackend()
    selection = PlanSelection(strategies=["Reflection of feelings", "Information"])

    result = TeacherRunner(backend, app_config).run_selected_strategies(
        example_id="selected-missing",
        history=history,
        state=_state(),
        selection=selection,
        reusable_candidates=_original_candidates(),
    )

    assert [candidate.strategy for candidate in result.candidates] == selection.strategies
    assert [candidate.candidate_id for candidate in result.candidates] == ["1", "2"]
    assert [candidate.strategy_id for candidate in result.candidates] == ["S1", "S2"]
    assert [call["role"] for call in backend.calls] == [
        "candidate_2",
        "emotion_critic",
        "effectiveness_critic",
        "safety_critic",
        "final_integrator",
    ]
    assert len(result.call_records) == 5


def test_selected_strategies_rejects_final_strategy_order_mismatch(history, app_config):
    from ibd.schemas import PlanSelection

    backend = SelectedStrategyBackend(reverse_final_strategies=True)
    selection = PlanSelection(strategies=["Reflection of feelings", "Question"])

    with pytest.raises(ValueError, match="selected strategies"):
        TeacherRunner(backend, app_config).run_selected_strategies(
            example_id="selected-wrong-order",
            history=history,
            state=_state(),
            selection=selection,
            reusable_candidates=_original_candidates(),
        )


def test_teacher_wraps_plain_text_candidate_when_json_mode_is_disabled(history):
    from ibd.backend import LLMResult
    from ibd.config import AppConfig, ModelConfig
    from ibd.schemas import Candidate

    class PlainTextBackend:
        def complete(self, **kwargs):
            return LLMResult(text="A grounded supportive response.")

    config = AppConfig(
        default_model=ModelConfig(model="fake-model"),
        roles={
            "candidate_1": ModelConfig(
                model="fake-model",
                provider_json_mode=False,
            )
        },
        schema_retries=0,
    )
    runner = TeacherRunner(PlainTextBackend(), config)
    records = []

    candidate = runner._call(
        "candidate_1",
        history,
        Candidate,
        records,
        context={
            "candidate_id": "1",
            "strategy_id": "S1",
            "strategy": "Question",
            "seed": 11,
        },
        seed=11,
        example_id="plain-candidate",
    )

    assert candidate == Candidate(
        candidate_id="1",
        strategy_id="S1",
        strategy="Question",
        response="A grounded supportive response.",
        seed=11,
    )


def test_teacher_parses_structured_final_when_provider_json_mode_is_disabled(history):
    from ibd.backend import LLMResult
    from ibd.config import AppConfig, ModelConfig
    from ibd.teacher import FinalAnswer

    class PlainTextBackend:
        def complete(self, **kwargs):
            return LLMResult(
                text='{"response":"A concise integrated response.","strategy_uses":[{"strategy_id":"S1","strategy":"Question","contribution":"Invites reflection."}]}'
            )

    config = AppConfig(
        default_model=ModelConfig(model="fake-model"),
        roles={
            "final_integrator": ModelConfig(
                model="fake-model",
                provider_json_mode=False,
            )
        },
        schema_retries=0,
    )
    runner = TeacherRunner(PlainTextBackend(), config)

    final = runner._call(
        "final_integrator",
        history,
        FinalAnswer,
        [],
        example_id="prompt-structured-final",
    )

    assert final.response == "A concise integrated response."


def test_teacher_rejects_candidate_with_mismatched_frozen_metadata(history, app_config):
    class WrongSeedBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "candidate_1":
                payload["seed"] = 999
            return payload

    with pytest.raises(ValueError, match="candidate metadata"):
        TeacherRunner(WrongSeedBackend(), app_config).run("e-wrong-seed", history)


def test_teacher_preserves_expert_evidence_without_string_matching(history, app_config):
    class SummarizedEvidenceBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "emotion_expert":
                payload["evidence"] = ["The seeker summarizes feeling overlooked."]
            return payload

    trace = TeacherRunner(SummarizedEvidenceBackend(), app_config).run(
        "e-summarized-evidence",
        history,
    )

    assert trace.expert_outputs["emotion"].evidence == [
        "The seeker summarizes feeling overlooked."
    ]


def test_teacher_rejects_role_identity_mismatch(history, app_config):
    class WrongCriticBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "safety_critic":
                payload["critic"] = "emotion"
            return payload

    with pytest.raises(ValueError, match="safety_critic failed schema validation"):
        TeacherRunner(WrongCriticBackend(), app_config).run("e-wrong-critic", history)


def test_teacher_retries_critic_identity_mismatch_before_caching(history, app_config):
    class WrongOnceCriticBackend(ScriptedBackend):
        def __init__(self):
            super().__init__()
            self.effectiveness_attempts = 0

        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "effectiveness_critic":
                self.effectiveness_attempts += 1
                if self.effectiveness_attempts == 1:
                    payload["critic"] = "emotion"
            return payload

    backend = WrongOnceCriticBackend()
    trace = TeacherRunner(backend, app_config).run("e-retry-critic-identity", history)

    assert backend.effectiveness_attempts == 2
    assert [critique.critic for critique in trace.critiques] == [
        "emotion",
        "effectiveness",
        "safety",
    ]


def test_teacher_requires_every_critic_to_cover_all_candidates(history, app_config):
    class MissingSafetyCandidateBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "safety_critic":
                payload["candidate_issues"].pop("2")
            return payload

    with pytest.raises(ValueError, match="candidate coverage mismatch"):
        TeacherRunner(MissingSafetyCandidateBackend(), app_config).run(
            "e-missing-safety", history
        )


def test_teacher_expands_empty_all_safe_report_to_every_candidate(history, app_config):
    class AllSafeBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "safety_critic":
                payload["candidate_issues"] = {}
                payload["summary"] = "No safety issues found in any candidate."
            return payload

    trace = TeacherRunner(AllSafeBackend(), app_config).run("e-all-safe", history)
    safety = next(report for report in trace.critiques if report.critic == "safety")

    assert safety.candidate_issues == {"1": [], "2": [], "3": []}


def test_teacher_trace_records_resolved_teacher_protocol_hash(history, app_config):
    runner = TeacherRunner(ScriptedBackend(), app_config)

    trace = runner.run("e-protocol", history)

    assert trace.teacher_protocol_hash == runner.caller.protocol_hash
    assert len(trace.teacher_protocol_hash) == 64


def test_teacher_cache_is_isolated_by_example_role_and_protocol(tmp_path, history):
    from ibd.config import AppConfig, BackendConfig, ModelConfig

    config = AppConfig(
        protocol_version="cache-v1",
        backend=BackendConfig(cache_dir=tmp_path / "teacher-cache"),
        default_model=ModelConfig(model="fake-model"),
    )
    backend = ScriptedBackend()
    runner = TeacherRunner(backend, config)

    first = runner.run("e-cache", history)
    second = runner.run("e-cache", history)
    runner.run("e-other", history)

    assert len(first.call_records) == len(second.call_records) == 13
    assert not any(record.cached for record in first.call_records)
    assert all(record.cached for record in second.call_records)
    assert len(backend.calls) == 26

    changed_protocol = config.model_copy(update={"protocol_version": "cache-v2"})
    changed_backend = ScriptedBackend()
    TeacherRunner(changed_backend, changed_protocol).run("e-cache", history)
    assert len(changed_backend.calls) == 13


def test_backend_base_url_can_reuse_supervisor_environment(monkeypatch):
    from ibd.config import BackendConfig

    monkeypatch.setenv("SUPERVISOR_TEST_BASE_URL", "https://router.example/v1")
    inherited = BackendConfig(base_url_env="SUPERVISOR_TEST_BASE_URL")
    explicit = BackendConfig(
        base_url="https://explicit.example/v1",
        base_url_env="SUPERVISOR_TEST_BASE_URL",
    )

    assert inherited.resolve_base_url() == "https://router.example/v1"
    assert explicit.resolve_base_url() == "https://explicit.example/v1"


def test_backend_base_url_strips_environment_whitespace(monkeypatch):
    from ibd.config import BackendConfig

    monkeypatch.setenv(
        "SUPERVISOR_TEST_BASE_URL",
        "\thttps://router.example/v1\n",
    )

    assert (
        BackendConfig(base_url_env="SUPERVISOR_TEST_BASE_URL").resolve_base_url()
        == "https://router.example/v1"
    )


def test_openai_backend_can_disable_thinking_for_structured_calls(monkeypatch):
    from types import SimpleNamespace

    import openai

    from ibd.backend import OpenAIBackend
    from ibd.config import AppConfig, BackendConfig, ModelConfig

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("SUPERVISOR_TEST_API_KEY", "test-key")
    config = AppConfig(
        backend=BackendConfig(
            base_url="https://router.example/v1",
            api_key_env="SUPERVISOR_TEST_API_KEY",
        ),
        default_model=ModelConfig(
            model="deepseek-v4-pro",
            thinking_enabled=False,
        ),
    )

    OpenAIBackend(config).complete(
        role="emotion_expert",
        messages=[{"role": "user", "content": "Return JSON"}],
        model_config=config.default_model,
    )

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_teacher_disables_thinking_for_every_role():
    from ibd.config import AppConfig

    config = AppConfig.from_yaml("configs/deepseek_teacher.yaml")
    model_configs = [config.default_model, *config.roles.values()]

    assert all(model.thinking_enabled is False for model in model_configs)


def test_openai_backend_empty_content_reports_safe_diagnostics(monkeypatch):
    from types import SimpleNamespace

    import openai

    from ibd.backend import OpenAIBackend
    from ibd.config import AppConfig, BackendConfig, ModelConfig

    usage = SimpleNamespace(
        completion_tokens=1200,
        model_dump=lambda: {"completion_tokens": 1200},
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    reasoning_content="private reasoning",
                ),
                finish_reason="length",
            )
        ],
        usage=usage,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("SUPERVISOR_TEST_API_KEY", "test-key")
    config = AppConfig(
        backend=BackendConfig(
            base_url="https://router.example/v1",
            api_key_env="SUPERVISOR_TEST_API_KEY",
        ),
        default_model=ModelConfig(model="deepseek-v4-pro", thinking_enabled=False),
    )

    with pytest.raises(RuntimeError) as captured:
        OpenAIBackend(config).complete(
            role="emotion_expert",
            messages=[{"role": "user", "content": "Return JSON"}],
            model_config=config.default_model,
        )

    message = str(captured.value)
    assert "finish_reason='length'" in message
    assert "reasoning_content_present=True" in message
    assert "reasoning_chars=17" in message
    assert "completion_tokens=1200" in message
    assert "private reasoning" not in message


def test_openai_backend_whitespace_content_reports_safe_diagnostics(monkeypatch):
    from types import SimpleNamespace

    import openai

    from ibd.backend import OpenAIBackend
    from ibd.config import AppConfig, BackendConfig, ModelConfig

    usage = SimpleNamespace(
        completion_tokens=800,
        model_dump=lambda: {"completion_tokens": 800},
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="   ", reasoning_content=None),
                finish_reason="length",
            )
        ],
        usage=usage,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("SUPERVISOR_TEST_API_KEY", "test-key")
    config = AppConfig(
        backend=BackendConfig(
            base_url="https://router.example/v1",
            api_key_env="SUPERVISOR_TEST_API_KEY",
        ),
        default_model=ModelConfig(model="deepseek-v4-pro", thinking_enabled=False),
    )

    with pytest.raises(RuntimeError, match="completion_tokens=800"):
        OpenAIBackend(config).complete(
            role="candidate_1",
            messages=[{"role": "user", "content": "Return JSON"}],
            model_config=config.default_model,
        )


def test_openai_backend_omits_unsupported_seed(monkeypatch):
    from types import SimpleNamespace

    import openai

    from ibd.backend import OpenAIBackend
    from ibd.config import AppConfig, BackendConfig, ModelConfig

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("SUPERVISOR_TEST_API_KEY", "test-key")
    config = AppConfig(
        backend=BackendConfig(
            base_url="https://router.example/v1",
            api_key_env="SUPERVISOR_TEST_API_KEY",
        ),
        default_model=ModelConfig(model="deepseek-v4-pro", supports_seed=False),
    )

    OpenAIBackend(config).complete(
        role="candidate_1",
        messages=[{"role": "user", "content": "Return JSON"}],
        model_config=config.default_model,
        seed=11,
    )

    assert "seed" not in captured


def test_openai_backend_can_disable_provider_json_mode(monkeypatch):
    from types import SimpleNamespace

    import openai

    from ibd.backend import OpenAIBackend
    from ibd.config import AppConfig, BackendConfig, ModelConfig

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("SUPERVISOR_TEST_API_KEY", "test-key")
    config = AppConfig(
        backend=BackendConfig(
            base_url="https://router.example/v1",
            api_key_env="SUPERVISOR_TEST_API_KEY",
        ),
        default_model=ModelConfig(
            model="deepseek-v4-pro",
            provider_json_mode=False,
        ),
    )

    OpenAIBackend(config).complete(
        role="candidate_1",
        messages=[{"role": "user", "content": "Return JSON"}],
        model_config=config.default_model,
        json_mode=True,
    )

    assert "response_format" not in captured


def test_deepseek_teacher_disables_provider_json_mode_for_response_roles():
    from ibd.config import AppConfig

    config = AppConfig.from_yaml("configs/deepseek_teacher.yaml")
    prompt_only_roles = {
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "final_integrator",
    }

    assert config.default_model.provider_json_mode is True
    assert all(config.roles[role].provider_json_mode is False for role in prompt_only_roles)
    assert config.roles["final_integrator"].provider_json_mode is False
    assert all(
        model.provider_json_mode is True
        for role, model in config.roles.items()
        if role not in prompt_only_roles
    )


def test_schema_retry_doubles_token_budget_after_invalid_json():
    from ibd.backend import LLMResult, StructuredCaller
    from ibd.config import AppConfig, ModelConfig
    from ibd.teacher import FinalAnswer

    class InvalidThenValidBackend:
        def __init__(self):
            self.max_tokens = []

        def complete(self, *, model_config, **kwargs):
            self.max_tokens.append(model_config.max_tokens)
            if len(self.max_tokens) == 1:
                return LLMResult(text='{"response":')
            return LLMResult(text='{"response":"recovered","strategy_uses":[{"strategy_id":"S1","strategy":"Question","contribution":"Invites reflection."}]}')

    backend = InvalidThenValidBackend()
    config = AppConfig(
        default_model=ModelConfig(model="fake-model", max_tokens=1000),
        schema_retries=1,
    )

    parsed, records = StructuredCaller(backend, config).call(
        "final_integrator",
        [{"role": "user", "content": "Return JSON"}],
        FinalAnswer,
    )

    assert parsed.response == "recovered"
    assert backend.max_tokens == [1000, 2000]
    assert [record.schema_retry for record in records] == [False, True]


def test_structured_caller_disables_provider_json_mode_for_final_integrator_after_invalid_json():
    from ibd.backend import LLMResult, StructuredCaller
    from ibd.config import AppConfig, ModelConfig
    from ibd.teacher import FinalAnswer

    class InvalidThenValidBackend:
        def __init__(self):
            self.provider_json_mode = []

        def complete(self, *, model_config, **kwargs):
            self.provider_json_mode.append(model_config.provider_json_mode)
            if len(self.provider_json_mode) == 1:
                return LLMResult(text="not json at all")
            return LLMResult(
                text='{"response":"recovered","strategy_uses":[{"strategy_id":"S1","strategy":"Question","contribution":"Invites reflection."}]}'
            )

    backend = InvalidThenValidBackend()
    config = AppConfig(
        default_model=ModelConfig(model="fake-model", max_tokens=1000),
        schema_retries=1,
    )

    parsed, records = StructuredCaller(backend, config).call(
        "final_integrator",
        [{"role": "user", "content": "Return JSON"}],
        FinalAnswer,
    )

    assert parsed.response == "recovered"
    assert backend.provider_json_mode == [True, False]
    assert [record.schema_retry for record in records] == [False, True]


def test_teacher_wraps_plain_text_final_response_with_strategy_metadata(history, app_config):
    class PlainTextBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "final_integrator":
                return "A concise natural-language supporter response."
            return payload

    trace = TeacherRunner(PlainTextBackend(), app_config).run(
        "e-plain-final-response", history
    )

    assert trace.final_answer.response == "A concise natural-language supporter response."
    assert len(trace.final_answer.strategy_uses) == 1
    assert trace.final_answer.strategy_uses[0].strategy_id == "S1"
    assert trace.final_answer.strategy_uses[0].strategy == trace.plan.strategies[0]


def test_structured_caller_retries_empty_provider_content(monkeypatch):
    from types import SimpleNamespace

    import openai

    from ibd.backend import OpenAIBackend, StructuredCaller
    from ibd.config import AppConfig, BackendConfig, ModelConfig
    from ibd.teacher import FinalAnswer

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(completion_tokens=18),
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"response":"recovered","strategy_uses":[{"strategy_id":"S1","strategy":"Question","contribution":"Invites reflection."}]}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: responses.pop(0))
        )
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("SUPERVISOR_TEST_API_KEY", "test-key")
    config = AppConfig(
        backend=BackendConfig(
            base_url="https://router.example/v1",
            api_key_env="SUPERVISOR_TEST_API_KEY",
        ),
        default_model=ModelConfig(model="deepseek-v4-pro"),
        schema_retries=1,
    )

    parsed, records = StructuredCaller(OpenAIBackend(config), config).call(
        "final_integrator",
        [{"role": "user", "content": "Return JSON"}],
        FinalAnswer,
    )

    assert parsed.response == "recovered"
    assert len(records) == 1
    assert responses == []


def test_schema_retry_keeps_token_budget_after_schema_validation_error():
    from ibd.backend import LLMResult, StructuredCaller
    from ibd.config import AppConfig, ModelConfig
    from ibd.teacher import FinalAnswer

    class InvalidThenValidBackend:
        def __init__(self):
            self.max_tokens = []

        def complete(self, *, model_config, **kwargs):
            self.max_tokens.append(model_config.max_tokens)
            if len(self.max_tokens) == 1:
                return LLMResult(text='{"unexpected": true}')
            return LLMResult(text='{"response":"recovered","strategy_uses":[{"strategy_id":"S1","strategy":"Question","contribution":"Invites reflection."}]}')

    backend = InvalidThenValidBackend()
    config = AppConfig(
        default_model=ModelConfig(model="fake-model", max_tokens=1000),
        schema_retries=1,
    )

    StructuredCaller(backend, config).call(
        "final_integrator",
        [{"role": "user", "content": "Return JSON"}],
        FinalAnswer,
    )

    assert backend.max_tokens == [1000, 1000]

def test_teacher_canonicalizes_final_strategy_name_from_strategy_id(
    history, app_config
):
    class WrongFinalStrategyNameBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "final_integrator":
                # S1 actually corresponds to "Reflection of feelings",
                # but simulate the LLM returning a wrong redundant label.
                payload["strategy_uses"][0]["strategy"] = "Information"
            return payload

    trace = TeacherRunner(
        WrongFinalStrategyNameBackend(),
        app_config,
    ).run("e-wrong-final-strategy-name", history)

    assert trace.final_answer.strategy_uses[0].strategy_id == "S1"
    assert (
        trace.final_answer.strategy_uses[0].strategy
        == "Reflection of feelings"
    )
