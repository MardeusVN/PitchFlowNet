"""Atomic training checkpoint save/resume."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

from .trainer import BanhmiTrainer


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), torch.cuda.current_device())


def _all_rank_rng_states() -> list[Dict[str, Any]]:
    local_state = _capture_rng_state()
    if not dist.is_available() or not dist.is_initialized():
        return [local_state]
    states: list[Dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def save_checkpoint(
    path: str | Path,
    trainer: BanhmiTrainer,
    global_step: int,
    extra: Dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    rng_states = _all_rank_rng_states()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "checkpoint_schema_version": 5,
        "global_step": int(global_step),
        "training_config": asdict(trainer.config),
        "model": trainer.model_module.state_dict(),
        "discriminator": trainer.discriminator_module.state_dict(),
        "duration_discriminator": (
            trainer.duration_discriminator_module.state_dict()
            if trainer.duration_discriminator_module is not None
            else None
        ),
        "generator_optimizer": trainer.generator_optimizer.state_dict(),
        "discriminator_optimizer": trainer.discriminator_optimizer.state_dict(),
        "generator_scheduler": trainer.generator_scheduler.state_dict(),
        "discriminator_scheduler": trainer.discriminator_scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "rng_states": rng_states,
        "extra": extra or {},
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path,
    trainer: BanhmiTrainer,
) -> Dict[str, Any]:
    state = torch.load(
        Path(path), map_location=trainer.device, weights_only=True
    )
    saved_config = state.get("training_config")
    if saved_config is not None and saved_config != asdict(trainer.config):
        raise ValueError("checkpoint training configuration does not match current config")
    trainer.model_module.load_state_dict(state["model"])
    trainer.discriminator_module.load_state_dict(state["discriminator"])
    duration_state = state.get("duration_discriminator")
    if trainer.duration_discriminator_module is not None and duration_state is not None:
        trainer.duration_discriminator_module.load_state_dict(duration_state)
    trainer.generator_optimizer.load_state_dict(state["generator_optimizer"])
    trainer.discriminator_optimizer.load_state_dict(
        state["discriminator_optimizer"]
    )
    if "generator_scheduler" in state:
        trainer.generator_scheduler.load_state_dict(state["generator_scheduler"])
    if "discriminator_scheduler" in state:
        trainer.discriminator_scheduler.load_state_dict(
            state["discriminator_scheduler"]
        )
    trainer.scaler.load_state_dict(state["scaler"])
    global_step = int(state["global_step"])
    if int(state.get("checkpoint_schema_version", 1)) == 1:
        global_step += 1
    rng_states = state.get("rng_states")
    if rng_states:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        _restore_rng_state(rng_states[min(rank, len(rng_states) - 1)])
    return {
        "global_step": global_step,
        "extra": state.get("extra", {}),
    }
