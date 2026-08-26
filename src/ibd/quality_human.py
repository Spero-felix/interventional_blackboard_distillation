"""Optional blinded pairwise human-review artifacts and aggregation."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from .quality import QualityResponse
from .storage import read_jsonl, write_jsonl


PUBLIC_FIELDS = (
    "pair_id",
    "split",
    "example_id",
    "history",
    "response_a",
    "response_b",
    "preference",
    "reviewer_id",
    "comment",
)


def _response_index(
    responses: Sequence[QualityResponse],
) -> tuple[dict[tuple[str, str, str], QualityResponse], list[str]]:
    indexed: dict[tuple[str, str, str], QualityResponse] = {}
    model_ids: set[str] = set()
    histories: dict[tuple[str, str], object] = {}
    for response in responses:
        key = (response.split, response.example_id, response.model_id)
        if key in indexed:
            raise ValueError(f"duplicate quality response: {key}")
        indexed[key] = response
        model_ids.add(response.model_id)
        history_key = (response.split, response.example_id)
        canonical = response.history.model_dump(mode="json")
        if history_key in histories and histories[history_key] != canonical:
            raise ValueError(f"conflicting histories for {history_key}")
        histories[history_key] = canonical
    ordered_models = sorted(model_ids)
    for split, example_id in histories:
        missing = [
            model_id
            for model_id in ordered_models
            if (split, example_id, model_id) not in indexed
        ]
        if missing:
            raise ValueError(
                f"incomplete response group for {(split, example_id)}: {missing}"
            )
    return indexed, ordered_models


def export_human_pairs(
    responses: Sequence[QualityResponse],
    public_csv: str | Path,
    mapping_jsonl: str | Path,
    *,
    seed: int,
) -> dict[str, int]:
    indexed, model_ids = _response_index(responses)
    public_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []
    pair_number = 1
    rng = random.Random(seed)
    splits = sorted({item.split for item in responses})
    for split in splits:
        for left, right in combinations(model_ids, 2):
            example_ids = sorted(
                example_id
                for row_split, example_id, model_id in indexed
                if row_split == split
                and model_id == left
                and (split, example_id, right) in indexed
            )
            rng.shuffle(example_ids)
            for position, example_id in enumerate(example_ids):
                first, second = (left, right) if position % 2 == 0 else (right, left)
                response_a = indexed[(split, example_id, first)]
                response_b = indexed[(split, example_id, second)]
                pair_id = f"pair-{pair_number:06d}"
                pair_number += 1
                public_rows.append(
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "example_id": example_id,
                        "history": json.dumps(
                            response_a.history.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "response_a": response_a.response,
                        "response_b": response_b.response,
                        "preference": "",
                        "reviewer_id": "",
                        "comment": "",
                    }
                )
                mapping_rows.append(
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "example_id": example_id,
                        "model_a": first,
                        "model_b": second,
                    }
                )

    target = Path(public_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PUBLIC_FIELDS))
        writer.writeheader()
        writer.writerows(public_rows)
    write_jsonl(mapping_jsonl, mapping_rows)
    return {"pairs": len(public_rows), "model_pairs": len(list(combinations(model_ids, 2)))}


def summarize_human_annotations(
    annotation_csv: str | Path,
    mapping_jsonl: str | Path,
) -> dict[str, Any]:
    mappings = {row["pair_id"]: row for row in read_jsonl(mapping_jsonl)}
    with Path(annotation_csv).open(encoding="utf-8", newline="") as handle:
        annotations = list(csv.DictReader(handle))

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "valid_votes": 0,
            "ties": 0,
            "missing": 0,
            "invalid": 0,
            "wins": defaultdict(int),
        }
    )
    seen_pair_ids: set[str] = set()
    for annotation in annotations:
        pair_id = str(annotation.get("pair_id", ""))
        mapping = mappings.get(pair_id)
        if mapping is None:
            raise ValueError(f"annotation has unknown pair_id: {pair_id}")
        models = sorted((str(mapping["model_a"]), str(mapping["model_b"])))
        group_key = f"{mapping['split']}:{models[0]}__vs__{models[1]}"
        group = groups[group_key]
        for model_id in models:
            group["wins"].setdefault(model_id, 0)
        seen_pair_ids.add(pair_id)
        preference = str(annotation.get("preference", "")).strip().casefold()
        if not preference:
            group["missing"] += 1
            continue
        if preference not in {"a", "b", "tie"}:
            group["invalid"] += 1
            continue
        group["valid_votes"] += 1
        if preference == "tie":
            group["ties"] += 1
        else:
            winner = str(mapping["model_a"] if preference == "a" else mapping["model_b"])
            group["wins"][winner] += 1

    for pair_id, mapping in mappings.items():
        if pair_id in seen_pair_ids:
            continue
        models = sorted((str(mapping["model_a"]), str(mapping["model_b"])))
        group_key = f"{mapping['split']}:{models[0]}__vs__{models[1]}"
        group = groups[group_key]
        for model_id in models:
            group["wins"].setdefault(model_id, 0)
        group["missing"] += 1

    result: dict[str, Any] = {}
    for key, raw in sorted(groups.items()):
        valid = int(raw["valid_votes"])
        wins = dict(sorted(raw["wins"].items()))
        result[key] = {
            "valid_votes": valid,
            "wins": wins,
            "win_rates": {
                model_id: round(count / valid, 4) if valid else None
                for model_id, count in wins.items()
            },
            "ties": int(raw["ties"]),
            "tie_rate": round(int(raw["ties"]) / valid, 4) if valid else None,
            "missing": int(raw["missing"]),
            "invalid": int(raw["invalid"]),
        }
    return {"protocol_version": "generation-quality-human-v1", "groups": result}
