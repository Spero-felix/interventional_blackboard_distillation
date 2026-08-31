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
        if role == "candidate":
            candidate_context = json.loads(messages[1]["content"])["context"]
            candidate_id = candidate_context["candidate_id"]
            payload = {
                "candidate_id": candidate_id,
                "strategy_id": candidate_context["strategy_id"],
                "strategy": candidate_context["strategy"],
                "response": f"候选回复-{candidate_id}",
                "seed": seed,
                "response_goal": f"候选目标-{candidate_id}",
                "response_act": f"候选动作-{candidate_id}",
            }
        else:
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
        if role == "multi_view_state_analyzer":
            return {
                "views": {
                    "emotion": {
                        "summary": "失落",
                        "evidence": "最近总觉得被忽略",
                        "uncertainty": "",
                    },
                    "need": {
                        "summary": "希望被理解",
                        "evidence": "想知道怎么开口",
                        "uncertainty": "",
                    },
                    "relationship": {
                        "summary": "双方回避沟通",
                        "evidence": "一谈就躲开",
                        "uncertainty": "",
                    },
                    "intent": {
                        "summary": "准备一次沟通",
                        "evidence": "想知道怎么开口",
                        "uncertainty": "",
                    },
                },
                "state": {
                    "emotion": "失落",
                    "intensity": "中等",
                    "primary_need": "被理解",
                    "support_goal": "准备一次坦诚沟通",
                    "readiness": "愿意探索",
                    "main_constraint": "担心对方继续回避",
                    "relationship_context": "亲密关系中的沟通僵局",
                },
            }
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
        if role == "state_counterfactual_generator":
            return {"replacement": "准备立即采取具体行动"}
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
                "response_goal": f"候选目标-{role[-1]}",
                "response_act": f"候选动作-{role[-1]}",
            }
        if role == "final_selector":
            return {
                "selected_candidate_id": "2",
                "response_goal": "帮助用户明确下一步",
                "response_act": "提出一个开放式问题",
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
