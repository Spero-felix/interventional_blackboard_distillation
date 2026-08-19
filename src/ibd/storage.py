"""Small durable JSON/JSONL helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


def _dumpable(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def append_jsonl(path: str | Path, record: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_dumpable(record), ensure_ascii=False, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_dumpable(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

