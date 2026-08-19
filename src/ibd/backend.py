"""Model backend boundary and schema-validated call wrapper."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import AppConfig, ModelConfig
from .hashing import protocol_hash
from .schemas import CallRecord

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


class LLMBackend(Protocol):
    def complete(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        json_mode: bool = True,
        seed: int | None = None,
    ) -> LLMResult: ...


class OpenAIBackend:
    """OpenAI-compatible implementation, imported lazily for offline tests."""

    def __init__(self, config: AppConfig):
        from openai import OpenAI

        api_key = os.environ.get(config.backend.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {config.backend.api_key_env}")
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.backend.base_url,
            timeout=config.backend.timeout_seconds,
        )

    def complete(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        json_mode: bool = True,
        seed: int | None = None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model_config.model,
            "messages": messages,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if seed is not None:
            kwargs["seed"] = seed
        response = self._client.chat.completions.create(**kwargs)
        usage = response.usage.model_dump() if response.usage else {}
        return LLMResult(text=response.choices[0].message.content or "", usage=usage)


class StructuredCaller:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.backend = backend
        self.config = config

    def call(
        self,
        role: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        seed: int | None = None,
    ) -> tuple[T, list[CallRecord]]:
        records: list[CallRecord] = []
        current_messages = list(messages)
        last_error: Exception | None = None
        for index in range(self.config.schema_retries + 1):
            result = self.backend.complete(
                role=role,
                messages=current_messages,
                model_config=self.config.for_role(role),
                json_mode=True,
                seed=seed,
            )
            try:
                payload = json.loads(result.text)
                parsed = response_model.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                records.append(
                    CallRecord(
                        role=role,
                        attempt=index + 1,
                        request_hash=protocol_hash(current_messages),
                        raw_text=result.text,
                        parsed={},
                        schema_retry=index > 0,
                    )
                )
                if index >= self.config.schema_retries:
                    break
                current_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": f"Previous output violated the JSON schema: {exc}. Return corrected JSON only.",
                    },
                ]
                continue
            records.append(
                CallRecord(
                    role=role,
                    attempt=index + 1,
                    request_hash=protocol_hash(current_messages),
                    raw_text=result.text,
                    parsed=parsed.model_dump(mode="json"),
                    schema_retry=index > 0,
                )
            )
            return parsed, records
        raise ValueError(f"{role} failed schema validation") from last_error

