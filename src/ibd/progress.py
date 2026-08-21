"""Consistent terminal progress for user-facing iterative workflows."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm import tqdm


T = TypeVar("T")


def track(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
) -> Any:
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
        dynamic_ncols=True,
    )
