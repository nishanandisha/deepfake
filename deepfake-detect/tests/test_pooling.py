"""Tests for the temporal pooling strategies.

Mean pooling assumes evidence is spread across the whole clip. LAV-DF edits
average 0.65s inside ~8.5s clips, so the mean dilutes them to ~8% of the
pooled vector. These tests pin the behaviour that motivates offering
attention and max pooling as alternatives.
"""

import pytest
import torch

from src.models.common import AttentionPool, build_pooler, masked_max_pool, masked_mean_pool


def test_mean_pool_dilutes_a_short_spike():
    """The failure mode in one assertion: one strongly-activated timestep in
    16 survives max pooling intact but is averaged to 1/16th by the mean."""
    x = torch.zeros(1, 16, 4)
    x[0, 7] = 10.0  # a single 'manipulated' timestep

    assert masked_mean_pool(x)[0, 0].item() == pytest.approx(10.0 / 16)
    assert masked_max_pool(x)[0, 0].item() == pytest.approx(10.0)


def test_max_pool_respects_padding():
    x = torch.zeros(1, 5, 3)
    x[0, 4] = 99.0  # large value, but padded
    mask = torch.tensor([[False, False, False, False, True]])

    assert masked_max_pool(x, mask).max().item() == pytest.approx(0.0)


def test_max_pool_all_padded_is_finite():
    x = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    pooled = masked_max_pool(x, mask)
    assert torch.isfinite(pooled).all()


def test_attention_pool_output_shape_and_finiteness():
    pool = AttentionPool(embed_dim=16)
    x = torch.randn(3, 10, 16)
    pooled = pool(x)
    assert pooled.shape == (3, 16)
    assert torch.isfinite(pooled).all()


def test_attention_weights_sum_to_one_over_valid_steps():
    pool = AttentionPool(embed_dim=8)
    x = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 4:] = True

    w = pool.attention_weights(x, mask)
    assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)
    assert w[0, 4:].sum().item() == pytest.approx(0.0, abs=1e-6)


def test_attention_pool_can_learn_to_select_one_timestep():
    """The capability mean pooling lacks: concentrating on the timestep that
    carries the evidence."""
    torch.manual_seed(0)
    pool = AttentionPool(embed_dim=4)
    opt = torch.optim.Adam(pool.parameters(), lr=0.1)

    x = torch.zeros(8, 10, 4)
    x[:, 3, 0] = 1.0  # marker only at timestep 3
    target = torch.zeros(8, 4)
    target[:, 0] = 1.0  # want the pooled vector to recover that marker

    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(pool(x), target)
        loss.backward()
        opt.step()

    weights = pool.attention_weights(x)
    assert weights[:, 3].mean().item() > 0.5, "should concentrate on the informative timestep"
    assert loss.item() < masked_mean_pool(x).sub(target).pow(2).mean().item()


def test_build_pooler_dispatch():
    x = torch.randn(2, 5, 8)
    for name in ["mean", "max", "attention"]:
        assert build_pooler(name, 8)(x, None).shape == (2, 8)


def test_build_pooler_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown pooling"):
        build_pooler("median", 8)


def test_poolers_are_padding_consistent():
    """All three must ignore padded timesteps, or a batch's results would
    depend on how much padding its longest clip forced."""
    x = torch.randn(1, 8, 4)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 5:] = True

    for name in ["mean", "max", "attention"]:
        pooler = build_pooler(name, 4)
        torch.manual_seed(0)
        full = pooler(x, mask)
        x_changed = x.clone()
        x_changed[0, 5:] = 1e3  # only padded region changes
        torch.manual_seed(0)
        assert torch.allclose(full, pooler(x_changed, mask), atol=1e-4), f"{name} leaks padding"
