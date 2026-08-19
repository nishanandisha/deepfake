"""Regression tests for class-imbalance handling.

Sampling and loss weighting both correct the SAME imbalance. Applying both
double-corrects: on the real LAV-DF run (357 real / 1039 fake) the weighted
sampler rebalanced to ~50:50 and then a pos_weight of 0.344 pushed the
model into predicting every clip "real" -- macro-F1 pinned at exactly
0.1903 while AUC kept climbing, because only the operating point was
broken, not the ranking.
"""

import torch
from omegaconf import OmegaConf

from src.training.common import make_pos_weight

DEVICE = torch.device("cpu")
LABELS = ["real"] * 357 + ["fake"] * 1039  # the real lean-run train split


def _cfg(**data_kwargs):
    return OmegaConf.create({"data": data_kwargs})


def test_sampler_on_disables_loss_weighting():
    pos_weight = make_pos_weight(_cfg(use_class_balanced_sampler=True), LABELS, DEVICE)
    assert pos_weight.item() == 1.0, (
        "sampler already balances the classes; extra pos_weight double-corrects"
    )


def test_loss_weighting_applies_when_sampler_off():
    pos_weight = make_pos_weight(_cfg(use_class_balanced_sampler=False), LABELS, DEVICE)
    assert pos_weight.item() != 1.0
    # fake is the majority here, so it should be down-weighted
    assert pos_weight.item() < 1.0


def test_explicit_override_can_force_both():
    """Deliberately stacking them stays possible for ablations, but only
    when asked for explicitly."""
    cfg = _cfg(use_class_balanced_sampler=True, use_loss_weighting=True)
    assert make_pos_weight(cfg, LABELS, DEVICE).item() < 1.0


def test_explicit_override_can_disable_both():
    cfg = _cfg(use_class_balanced_sampler=False, use_loss_weighting=False)
    assert make_pos_weight(cfg, LABELS, DEVICE).item() == 1.0


def test_balanced_labels_give_neutral_weight_when_weighting_enabled():
    balanced = ["real"] * 500 + ["fake"] * 500
    cfg = _cfg(use_class_balanced_sampler=False)
    assert abs(make_pos_weight(cfg, balanced, DEVICE).item() - 1.0) < 1e-6


def test_default_when_key_absent_matches_sampler_default():
    """cfg.data may omit the key entirely; the sampler defaults to on, so
    loss weighting must default to off."""
    assert make_pos_weight(_cfg(), LABELS, DEVICE).item() == 1.0
