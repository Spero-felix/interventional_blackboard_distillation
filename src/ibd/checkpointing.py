"""Atomic single-process checkpoints with strict stage lineage and hashes."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from pydantic import Field

from .hashing import protocol_hash
from .schemas import StrictModel


StageName = Literal["A", "B", "C", "D"]
_LEGAL_SOURCES: dict[StageName, set[StageName]] = {
    "A": {"A"},
    "B": {"A", "B"},
    "C": {"B", "C"},
    "D": {"C", "D"},
}


class CheckpointMetadata(StrictModel):
    checkpoint_version: str = "qwen-stage-checkpoint-v1"
    stage: StageName
    run_name: str = Field(min_length=1)
    seed: int
    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    protocol_hash: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    model_hash: str = Field(min_length=1)
    tokenizer_hash: str = Field(min_length=1)
    data_manifest_hash: str = Field(min_length=1)
    split_hash: str = Field(min_length=1)
    anchor_hash: str = Field(min_length=1)
    slot_layer: int = Field(ge=0)
    special_token_ids: dict[Literal["STATE", "PLAN"], int]
    parent_checkpoint_hash: str | None = None


def validate_stage_lineage(source_stage: StageName, target_stage: StageName) -> None:
    if source_stage not in _LEGAL_SOURCES[target_stage]:
        raise ValueError(f"cannot resume stage {target_stage} from stage {source_stage}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_hash(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint artifact directory is empty: {path}")
    return protocol_hash(
        [
            {
                "path": str(item.relative_to(path)),
                "sha256": _file_sha256(item),
            }
            for item in files
        ]
    )


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
        config_hash: str,
    ):
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.seed = seed
        self.config_hash = config_hash
        self.run_dir.mkdir(parents=True, exist_ok=True)
        owner_path = self.run_dir / "run_owner.json"
        owner = {"run_name": run_name, "seed": seed, "config_hash": config_hash}
        if owner_path.exists():
            existing = json.loads(owner_path.read_text(encoding="utf-8"))
            if existing.get("run_name") != run_name:
                raise ValueError("run directory belongs to a different run")
            if existing.get("seed") != seed:
                raise ValueError("run directory belongs to a different seed")
            if existing.get("config_hash") != config_hash:
                raise ValueError("run directory belongs to a different config")
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
        tokenizer: Any = None,
    ) -> Path:
        if metadata.run_name != self.run_name:
            raise ValueError("checkpoint metadata belongs to a different run")
        if metadata.seed != self.seed:
            raise ValueError("checkpoint metadata belongs to a different seed")
        if metadata.config_hash != self.config_hash:
            raise ValueError("checkpoint metadata belongs to a different config")
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
            tokenizer_hash: str | None = None
            if tokenizer is not None:
                tokenizer_path = temporary / "tokenizer"
                tokenizer.save_pretrained(tokenizer_path)
                for artifact in tokenizer_path.rglob("*"):
                    if artifact.is_file():
                        with artifact.open("rb") as handle:
                            os.fsync(handle.fileno())
                _fsync_directory(tokenizer_path)
                tokenizer_hash = _directory_hash(tokenizer_path)
            checkpoint_hash = protocol_hash(
                {
                    "metadata": metadata.model_dump(mode="json"),
                    "model_state_sha256": _file_sha256(model_path),
                    "training_state_sha256": _file_sha256(training_path),
                    "tokenizer_hash": tokenizer_hash,
                }
            )
            checkpoint_path = temporary / "checkpoint.json"
            with checkpoint_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "metadata": metadata.model_dump(mode="json"),
                        "checkpoint_hash": checkpoint_hash,
                        "tokenizer_hash": tokenizer_hash,
                    },
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
        if metadata.config_hash != self.config_hash:
            raise ValueError("checkpoint belongs to a different config")
        for key, value in (expected or {}).items():
            actual = getattr(metadata, key)
            if actual != value:
                raise ValueError(f"{key} mismatch: checkpoint={actual!r}, expected={value!r}")

        model_path = source / "model_state.pt"
        training_path = source / "training_state.pt"
        computed_hash = protocol_hash(
            {
                "metadata": metadata.model_dump(mode="json"),
                "model_state_sha256": _file_sha256(model_path),
                "training_state_sha256": _file_sha256(training_path),
                "tokenizer_hash": (
                    _directory_hash(source / "tokenizer")
                    if payload.get("tokenizer_hash") is not None
                    else None
                ),
            }
        )
        if computed_hash != payload["checkpoint_hash"]:
            raise ValueError("checkpoint content hash mismatch")
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
