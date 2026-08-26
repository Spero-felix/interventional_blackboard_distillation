"""YAML configuration for reproducible Teacher protocols."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


STATE_TOKEN_LIMIT = 128
STRATEGY_CATALOG = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    supports_seed: bool = False
    provider_json_mode: bool = True
    thinking_enabled: bool | None = None


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str | None = None
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = Field(default=60.0, gt=0)
    cache_dir: Path | None = None

    def resolve_base_url(self) -> str | None:
        value = self.base_url or os.environ.get(self.base_url_env)
        if value is None:
            return None
        return value.strip() or None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: str = "1"
    backend: BackendConfig = Field(default_factory=BackendConfig)
    default_model: ModelConfig
    roles: dict[str, ModelConfig] = Field(default_factory=dict)
    candidate_seeds: tuple[int, int, int] = (11, 29, 47)
    schema_retries: int = Field(default=1, ge=0, le=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with Path(path).open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)

    def for_role(self, role: str) -> ModelConfig:
        return self.roles.get(role, self.default_model)
