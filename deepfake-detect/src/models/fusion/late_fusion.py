"""Stage 4: trivial late-fusion baseline. Loads the frozen Stage 2/3
checkpoints (never modifies them) and averages each branch's standalone
probability. This is the comparison point Stage 5's cross-attention fusion
must beat.
"""

from typing import List

import torch
from torch.utils.data import DataLoader

from src.training.common import resolve_device


def predict_branch_probabilities(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
) -> List[float]:
    """Runs a frozen standalone-branch model over `dataset` in split order
    (no shuffling) and returns its sigmoid probability per sample."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = model.to(device)
    model.eval()

    probabilities = []
    with torch.no_grad():
        for inputs, padding_mask, _ in loader:
            logits = model(inputs.to(device), padding_mask=padding_mask.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return probabilities


def average_probabilities(
    visual_probs: List[float], acoustic_probs: List[float], visual_weight: float = 0.5
) -> List[float]:
    """Weighted average of two per-sample probability lists (same order,
    same underlying manifest rows). `visual_weight=0.5` is a plain average;
    override for a hand-tuned weighted average without retraining anything."""
    assert len(visual_probs) == len(acoustic_probs), "branch probability lists must be aligned"
    return [
        visual_weight * v + (1 - visual_weight) * a
        for v, a in zip(visual_probs, acoustic_probs)
    ]


def load_frozen_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


__all__ = [
    "predict_branch_probabilities",
    "average_probabilities",
    "load_frozen_checkpoint",
    "resolve_device",
]
