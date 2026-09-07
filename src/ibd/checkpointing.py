"""Atomic single-process checkpoints with strict stage lineage."""

from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from pydantic import Field

from .schemas import StrictModel


StageName = Literal["A", "B", "SFT"]
_LEGAL_SOURCES: dict[StageName, set[StageName]] = {
    "A": {"A"},
    "B": {"A", "B"},
    "SFT": {"SFT"},
}


class CheckpointMetadata(StrictModel):
    checkpoint_version: str = "qwen-stage-checkpoint-v1"
    stage: StageName
    run_name: str = Field(min_length=1)
    seed: int
    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    slot_layer: int | None = Field(default=None, ge=0)
    special_token_ids: dict[Literal["STATE", "PLAN"], int] | None = None


def validate_stage_lineage(source_stage: StageName, target_stage: StageName) -> None:
    if source_stage not in _LEGAL_SOURCES[target_stage]:
        raise ValueError(f"cannot resume stage {target_stage} from stage {source_stage}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _model_state(model: torch.nn.Module) -> tuple[str, dict[str, torch.Tensor]]:
    if hasattr(model, "peft_config"):
        from peft import get_peft_model_state_dict

        return "peft", get_peft_model_state_dict(model)
    return "full", model.state_dict()


def _restore_model_state(
    model: torch.nn.Module,
    kind: str,
    state: dict[str, torch.Tensor],
) -> None:
    if kind == "peft":
        from peft import set_peft_model_state_dict

        result = set_peft_model_state_dict(model, state)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise ValueError(f"unexpected PEFT checkpoint keys: {unexpected}")
        return
    model.load_state_dict(state)


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_name: str,
        seed: int,
    ):
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.seed = seed
        self.run_dir.mkdir(parents=True, exist_ok=True)
        owner_path = self.run_dir / "run_owner.json"
        owner = {"run_name": run_name, "seed": seed}
        if owner_path.exists():
            existing = json.loads(owner_path.read_text(encoding="utf-8"))
            if existing.get("run_name") != run_name:
                raise ValueError("run directory belongs to a different run")
            if existing.get("seed") != seed:
                raise ValueError("run directory belongs to a different seed")
        else:
            _atomic_json(owner_path, owner)

    def save(
        self,
        metadata: CheckpointMetadata,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> Path:
        if metadata.run_name != self.run_name:
            raise ValueError("checkpoint metadata belongs to a different run")
        if metadata.seed != self.seed:
            raise ValueError("checkpoint metadata belongs to a different seed")
        name = f"stage-{metadata.stage}-step-{metadata.global_step}"
        target = self.run_dir / name
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        temporary = self.run_dir / f".{name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            model_kind, state = _model_state(model)
            model_path = temporary / "model_state.pt"
            torch.save({"kind": model_kind, "state": state}, model_path)
            training_path = temporary / "training_state.pt"
            torch.save(
                {
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "rng": _capture_rng(),
                },
                training_path,
            )
            checkpoint_path = temporary / "checkpoint.json"
            with checkpoint_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"metadata": metadata.model_dump(mode="json")},
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            for path in (model_path, training_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_directory(temporary)
            os.replace(temporary, target)
            _fsync_directory(self.run_dir)
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load(
        self,
        checkpoint_dir: str | Path,
        *,
        target_stage: StageName,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        expected: Mapping[str, Any] | None = None,
        restore_rng: bool = True,
    ) -> CheckpointMetadata:
        source = Path(checkpoint_dir)
        payload = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
        metadata = CheckpointMetadata.model_validate(payload["metadata"])
        validate_stage_lineage(metadata.stage, target_stage)
        if metadata.run_name != self.run_name:
            raise ValueError("checkpoint belongs to a different run")
        if metadata.seed != self.seed:
            raise ValueError("checkpoint belongs to a different seed")
        for key, value in (expected or {}).items():
            actual = getattr(metadata, key)
            if actual != value:
                raise ValueError(f"{key} mismatch: checkpoint={actual!r}, expected={value!r}")

        model_path = source / "model_state.pt"
        training_path = source / "training_state.pt"
        model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
        _restore_model_state(model, model_payload["kind"], model_payload["state"])
        training = torch.load(training_path, map_location="cpu", weights_only=False)
        if optimizer is not None and training["optimizer"] is not None:
            optimizer.load_state_dict(training["optimizer"])
        if scheduler is not None and training["scheduler"] is not None:
            scheduler.load_state_dict(training["scheduler"])
        if scaler is not None and training["scaler"] is not None:
            scaler.load_state_dict(training["scaler"])
        if restore_rng:
            _restore_rng(training["rng"])
        return metadata
