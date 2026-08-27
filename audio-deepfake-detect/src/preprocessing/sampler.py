"""Class-imbalance handling for the ~40:1 fake:real ratio in FakeAVCeleb.

Two independent, toggleable strategies -- pick one via config, not both:
- make_weighted_sampler: inverse-frequency WeightedRandomSampler for the
  training DataLoader.
- class_balanced_loss_weights: inverse-frequency per-class weights to pass
  into a weighted BCE/cross-entropy loss instead of resampling.
"""

from typing import Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def _class_counts(labels: Sequence[str]) -> dict:
    counts = {"real": 0, "fake": 0}
    for label in labels:
        counts[label] += 1
    return counts


def make_weighted_sampler(
    labels: Sequence[str], enabled: bool = True
) -> Optional[WeightedRandomSampler]:
    """Returns a WeightedRandomSampler that upsamples the minority class to
    ~parity per epoch, or None if disabled (caller should then use plain
    shuffling)."""
    if not enabled:
        return None

    counts = _class_counts(labels)
    weight_per_class = {label: 1.0 / count for label, count in counts.items() if count > 0}
    sample_weights = [weight_per_class[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def class_balanced_loss_weights(labels: Sequence[str], enabled: bool = True) -> torch.Tensor:
    """Returns a tensor [weight_real, weight_fake] for use as `pos_weight` /
    per-class loss weighting. Returns [1.0, 1.0] if disabled."""
    if not enabled:
        return torch.tensor([1.0, 1.0])

    counts = _class_counts(labels)
    total = counts["real"] + counts["fake"]
    weight_real = total / (2 * max(counts["real"], 1))
    weight_fake = total / (2 * max(counts["fake"], 1))
    return torch.tensor([weight_real, weight_fake])
