import json
from typing import Any

import pytest


class ScriptedBackend:
    """Network-free backend that emits complete production-shaped JSON."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

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
        payload = self._payload(role, seed)
        return LLMResult(text=json.dumps(payload, ensure_ascii=False), usage={"total_tokens": 10})

    def _payload(self, role: str, seed: int | None):
        expert_payloads = {
            "emotion_expert": {
                "expert": "emotion",
                "fields": {"emotion": "失落", "intensity": "medium"},
                "evidence": ["最近总觉得被忽略"],
                "uncertainties": [],
            },
            "need_expert": {
                "expert": "need",
                "fields": {"need": "被理解", "readiness": "愿意表达"},
                "evidence": ["我想知道怎么开口"],
                "uncertainties": [],
            },
            "relationship_expert": {
                "expert": "relationship",
                "fields": {"relationship_type": "亲密关系", "relationship_pattern": "回避沟通"},
                "evidence": ["我们一谈就躲开"],
                "uncertainties": [],
            },
            "intent_expert": {
                "expert": "intent",
                "fields": {"intent": "准备一次沟通", "decision_stage": "探索"},
                "evidence": ["我想知道怎么开口"],
                "uncertainties": [],
            },
        }
        if role in expert_payloads:
            return expert_payloads[role]
        if role == "state_integrator":
            return {
                "emotion": "失落",
                "intensity": "中等",
                "primary_need": "被理解",
                "support_goal": "准备一次坦诚沟通",
                "readiness": "愿意探索",
                "main_constraint": "担心对方继续回避",
                "relationship_context": "亲密关系中的沟通僵局",
            }
        if role == "planner":
            return {
                "strategies": [
                    "Reflection of feelings",
                    "Question",
                    "Providing Suggestions",
                ]
            }
        if role.startswith("candidate_"):
            index = int(role[-1])
            strategies = {
                1: ("S1", "Reflection of feelings"),
                2: ("S2", "Question"),
                3: ("S3", "Providing Suggestions"),
            }
            strategy_id, strategy = strategies[index]
            return {
                "candidate_id": role.removeprefix("candidate_"),
                "strategy_id": strategy_id,
                "strategy": strategy,
                "response": f"候选回复-{role[-1]}",
                "seed": seed,
            }
        if role.endswith("_critic"):
            critic = role.removesuffix("_critic")
            return {
                "critic": critic,
                "candidate_issues": {"1": [], "2": [], "3": []},
                "summary": "完成独立检查",
            }
        if role == "final_integrator":
            return {
                "response": "我能听出那种被忽略后的失落。你更希望先理清期待，还是一起准备一句开场？",
                "strategy_uses": [
                    {
                        "strategy_id": "S1",
                        "strategy": "Reflection of feelings",
                        "contribution": "承接用户的失落感",
                    },
                    {
                        "strategy_id": "S2",
                        "strategy": "Question",
                        "contribution": "邀请用户选择下一步",
                    },
                ],
            }
        raise AssertionError(f"unexpected role: {role}")


@pytest.fixture
def history():
    from ibd.schemas import DialogueTurn, History

    return History(
        turns=[
            DialogueTurn(role="seeker", content="最近总觉得被忽略。"),
            DialogueTurn(role="supporter", content="你愿意多说一点吗？"),
            DialogueTurn(role="seeker", content="我们一谈就躲开，我想知道怎么开口。"),
        ]
    )


@pytest.fixture
def app_config():
    from ibd.config import AppConfig, ModelConfig

    return AppConfig(default_model=ModelConfig(model="fake-model"))
