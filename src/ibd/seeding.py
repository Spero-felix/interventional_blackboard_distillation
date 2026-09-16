"""Stable seed derivation for profile-driven conversation model calls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedDeriver:
    protocol_version: str
    conversation_id: str
    base_seed: int
    round_index: int

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ValueError("round_index must be at least one")

    def for_call(self, role: str, cache_variant: str | None = None) -> int:
        normalized_role = role.strip()
        if not normalized_role:
            raise ValueError("role must be non-blank")
        components = (
            self.protocol_version,
            self.conversation_id,
            str(self.base_seed),
            str(self.round_index),
            normalized_role,
            cache_variant if cache_variant is not None else "<none>",
        )
        digest = hashlib.sha256("\0".join(components).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31)
