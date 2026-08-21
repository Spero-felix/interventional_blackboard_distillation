"""Model backend boundary and schema-validated call wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
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


class EmptyContentError(RuntimeError):
    """Raised when a provider completes successfully without response text."""


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
            base_url=config.backend.resolve_base_url(),
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
        if json_mode and model_config.provider_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if model_config.thinking_enabled is not None:
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled" if model_config.thinking_enabled else "disabled"
                }
            }
        if seed is not None and model_config.supports_seed:
            kwargs["seed"] = seed
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        content = (message.content or "").strip()
        if not content:
            reasoning_content = getattr(message, "reasoning_content", None)
            usage = getattr(response, "usage", None)
            raise EmptyContentError(
                f"{role} returned empty content "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r}, "
                f"reasoning_content_present={bool(reasoning_content)}, "
                f"reasoning_chars={len(reasoning_content or '')}, "
                f"completion_tokens={int(getattr(usage, 'completion_tokens', 0) or 0)})"
            )
        usage = response.usage.model_dump() if response.usage else {}
        return LLMResult(text=content, usage=usage)


class StructuredCaller:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.backend = backend
        self.config = config
        self._memory_cache: dict[str, dict[str, Any]] = {}
        protocol = config.model_dump(mode="json", exclude={"backend": {"cache_dir"}})
        protocol["backend"]["resolved_base_url"] = config.backend.resolve_base_url()
        self._protocol_hash = protocol_hash(protocol)

    @property
    def protocol_hash(self) -> str:
        return self._protocol_hash

    def _cache_key(self, example_id: str, role: str) -> str:
        return protocol_hash((example_id, role, self._protocol_hash))

    def _cache_path(self, key: str, *, failure: bool = False) -> Path | None:
        if self.config.backend.cache_dir is None:
            return None
        folder = "failures" if failure else "calls"
        return Path(self.config.backend.cache_dir) / folder / f"{key}.json"

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _read_cached(
        self,
        key: str,
        response_model: type[T],
    ) -> tuple[T, CallRecord] | None:
        payload = self._memory_cache.get(key)
        path = self._cache_path(key)
        if payload is None and path is not None and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        if payload is None:
            return None
        try:
            parsed = response_model.model_validate(payload["parsed"])
            record = CallRecord.model_validate(payload["record"]).model_copy(
                update={"cached": True}
            )
        except (KeyError, TypeError, ValidationError):
            return None
        self._memory_cache[key] = payload
        return parsed, record

    def call(
        self,
        role: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        seed: int | None = None,
        example_id: str | None = None,
    ) -> tuple[T, list[CallRecord]]:
        cache_key = self._cache_key(example_id, role) if example_id is not None else None
        if cache_key is not None:
            cached = self._read_cached(cache_key, response_model)
            if cached is not None:
                parsed, record = cached
                return parsed, [record]

        records: list[CallRecord] = []
        current_messages = list(messages)
        last_error: Exception | None = None
        for index in range(self.config.schema_retries + 1):
            model_config = self.config.for_role(role)
            if index > 0 and isinstance(last_error, json.JSONDecodeError):
                model_config = model_config.model_copy(
                    update={"max_tokens": model_config.max_tokens * 2}
                )
            try:
                result = self.backend.complete(
                    role=role,
                    messages=current_messages,
                    model_config=model_config,
                    json_mode=True,
                    seed=seed,
                )
            except EmptyContentError as exc:
                last_error = exc
                if index >= self.config.schema_retries:
                    raise
                current_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": "The previous response was empty. Return a non-empty JSON object matching the requested schema.",
                    },
                ]
                continue
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
            if cache_key is not None:
                record = records[-1]
                cache_payload = {
                    "example_id": example_id,
                    "role": role,
                    "protocol_hash": self._protocol_hash,
                    "parsed": parsed.model_dump(mode="json"),
                    "record": record.model_dump(mode="json"),
                }
                self._memory_cache[cache_key] = cache_payload
                path = self._cache_path(cache_key)
                if path is not None:
                    self._atomic_json(path, cache_payload)
            return parsed, records
        if cache_key is not None:
            path = self._cache_path(cache_key, failure=True)
            if path is not None:
                self._atomic_json(
                    path,
                    {
                        "example_id": example_id,
                        "role": role,
                        "protocol_hash": self._protocol_hash,
                        "records": [record.model_dump(mode="json") for record in records],
                        "error": str(last_error),
                    },
                )
        raise ValueError(f"{role} failed schema validation") from last_error
