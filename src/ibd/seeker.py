"""Private-profile seeker simulation with plain-text model output."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pydantic import ValidationError

from .backend import CallCache, EmptyContentError, LLMBackend
from .config import AppConfig
from .conversation_schemas import ConversationTurn, DialogueMode
from .schemas import CallRecord


class ValidatedProfile(Protocol):
    @property
    def profile_id(self) -> str: ...

    def render_for_seeker(self) -> str: ...


class SeekerProfileAdapter(Protocol):
    def validate(self, raw_profile: Mapping[str, Any]) -> ValidatedProfile: ...


@dataclass(frozen=True)
class _MappingValidatedProfile:
    profile_id: str
    _rendered_profile: str

    def render_for_seeker(self) -> str:
        return self._rendered_profile


class MappingProfileAdapter:
    """Minimal adapter that deliberately assigns no semantics beyond profile ID."""

    def validate(self, raw_profile: Mapping[str, Any]) -> ValidatedProfile:
        raw_id = raw_profile.get("ID")
        if raw_id is None or not str(raw_id).strip():
            raise ValueError("profile must contain a non-empty ID")
        copied = copy.deepcopy(dict(raw_profile))
        rendered = json.dumps(copied, ensure_ascii=False, sort_keys=True)
        return _MappingValidatedProfile(
            profile_id=str(raw_id).strip(),
            _rendered_profile=rendered,
        )


class TextCaller:
    """Retrying, cached caller for roles whose contract is plain text."""

    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.backend = backend
        self.config = config
        self.cache = CallCache(config)

    def _read_cached(self, key: str) -> tuple[str, CallRecord] | None:
        payload = self.cache.read(key)
        if payload is None:
            return None
        try:
            text = payload["text"]
            if not isinstance(text, str) or not text.strip():
                return None
            record = CallRecord.model_validate(payload["record"]).model_copy(
                update={"cached": True}
            )
        except (KeyError, TypeError, ValidationError):
            return None
        return text, record

    def call(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        seed: int,
        example_id: str,
        cache_variant: str | None = None,
    ) -> tuple[str, CallRecord]:
        key = self.cache.key(example_id, role, cache_variant)
        cached = self._read_cached(key)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(1, self.config.schema_retries + 2):
            try:
                result = self.backend.complete(
                    role=role,
                    messages=messages,
                    model_config=self.config.for_role(role),
                    json_mode=False,
                    seed=seed,
                )
                text = result.text.strip()
                if not text:
                    raise EmptyContentError(f"{role} returned empty text")
            except Exception as exc:
                last_error = exc
                if attempt > self.config.schema_retries:
                    raise
                continue

            record = CallRecord(
                role=role,
                attempt=attempt,
                raw_text=result.text,
                parsed={"text": text},
            )
            self.cache.write(
                key,
                {
                    "example_id": example_id,
                    "role": role,
                    "text": text,
                    "record": record.model_dump(mode="json"),
                    **(
                        {"cache_variant": cache_variant}
                        if cache_variant is not None
                        else {}
                    ),
                },
            )
            return text, record
        raise RuntimeError(f"{role} text call failed") from last_error


@dataclass(frozen=True)
class SeekerGeneration:
    utterance: str
    call_record: CallRecord


_SEEKER_SYSTEM_PROMPT = """You are simulating the seeker described by the private profile.

- Speak only as the seeker.
- Treat the profile as private background, not text to quote or summarize.
- Never mention that you were given a profile.
- Remain consistent with the profile throughout the conversation.
- Respond naturally to the latest supporter message.
- Reveal information gradually and only when conversationally relevant.
- Do not disclose every known fact in the opening turns.
- Do not invent events, relationships, goals, symptoms, or experiences that contradict or materially extend the profile.
- Preserve uncertainty present in the profile.
- You may disagree with, reject, question, or only partially accept support.
- Do not become satisfied merely because the supporter offered one suggestion.
- When the immediate conversational need has naturally been met, express closure through the utterance itself.
- Return only the next seeker utterance."""

_MODE_GUIDANCE: dict[DialogueMode, str] = {
    "opening": "Begin naturally and introduce the initial concern.",
    "exploration": "Respond to the current exchange and gradually add relevant background.",
    "comforting": "Naturally show whether you feel understood without forcing improvement.",
    "action": "Accept, hesitate about, reject, or clarify suggestions according to the profile and public dialogue.",
    "closing": "If the current need has naturally been met, express closure without introducing a major new concern.",
}


class SeekerSimulator:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.caller = TextCaller(backend, config)

    def generate(
        self,
        profile: ValidatedProfile,
        history: Sequence[ConversationTurn],
        mode: DialogueMode,
        *,
        round_index: int,
        seed: int,
        conversation_id: str,
    ) -> SeekerGeneration:
        if round_index < 1:
            raise ValueError("round_index must be at least one")
        payload = {
            "private_profile": profile.render_for_seeker(),
            "history": [turn.model_dump(mode="json") for turn in history],
            "mode": mode,
            "mode_guidance": _MODE_GUIDANCE[mode],
            "round_index": round_index,
        }
        messages = [
            {"role": "system", "content": _SEEKER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        utterance, call_record = self.caller.call(
            "seeker_simulator",
            messages,
            seed=seed,
            example_id=conversation_id,
            cache_variant=f"round-{round_index}",
        )
        return SeekerGeneration(utterance=utterance, call_record=call_record)
