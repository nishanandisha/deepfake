"""Resumable checkpointing.

The training loops originally saved bare `model.state_dict()`, which is
enough to *evaluate* a trained model but not to *continue* training it --
optimizer moments, LR-schedule position, epoch counter and early-stopping
state were all lost. On a laptop GPU that matters: a 10-hour run has to be
splittable into short sessions, and an interrupted run shouldn't restart
from scratch.

Two artefacts are kept side by side:
  best.pt   weights only, for downstream stages (Stage 4/6/7 load this)
  last.pt   full training state, for resuming
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class TrainingState:
    epoch: int = 0
    best_auc: float = -1.0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    best_val_metrics: Dict[str, float] = field(default_factory=dict)


def save_resumable(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
) -> None:
    temp_path = Path(path).with_suffix(".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": asdict(state),
    }
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)  # atomic: never leave a half-written resume file
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def load_resumable(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Optional[TrainingState]:
    """Restores model/optimizer/scheduler in place and returns the training
    state, or None if there's no checkpoint to resume from."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    # Load to CPU, not straight to the GPU. `last.pt` holds the model AND
    # the optimizer's momentum buffers (roughly 3x the model size), so
    # map_location=device materialises all of it in VRAM *before*
    # load_state_dict copies it into the already-resident model -- briefly
    # doubling usage. On a 4GB card that spike alone caused an OOM on
    # resume, on a run that trained fine from scratch. Loading to host
    # memory first lets load_state_dict stream it across tensor by tensor.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])

    # optimizer.load_state_dict keeps state tensors on whatever device they
    # came from (CPU here), which would then mismatch the model's params.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])

    return TrainingState(**payload["state"])
