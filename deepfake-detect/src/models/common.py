"""Building blocks shared by the visual, acoustic, and (Stage 5) fusion
encoders: sinusoidal positional encoding and mask-aware temporal pooling."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d]
        return x + self.pe[: x.size(1)].unsqueeze(0)


def masked_mean_pool(x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
    """x: [B, T, d], padding_mask: [B, T] bool (True = padded). Mean over
    valid (non-padded) timesteps only."""
    if padding_mask is None:
        return x.mean(dim=1)
    valid = (~padding_mask).unsqueeze(-1).float()
    summed = (x * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp(min=1.0)
    return summed / counts


def masked_max_pool(x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
    """Max over valid timesteps. Unlike the mean, a single strongly-activated
    timestep survives pooling -- which is what a short localized manipulation
    produces."""
    if padding_mask is not None:
        x = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
    pooled = x.max(dim=1).values
    # A row that is entirely padding would be all -inf; fall back to zeros.
    return torch.nan_to_num(pooled, neginf=0.0)


class AttentionPool(nn.Module):
    """Learned attention pooling over the time axis.

    Mean pooling assumes the evidence is spread across the whole clip. That
    assumption fails badly on content-driven deepfakes: LAV-DF edits average
    0.65s inside ~8.5s clips, so averaging dilutes the manipulated region to
    roughly 8% of the pooled vector and the signal is largely washed out.

    Scoring each timestep and taking a softmax-weighted sum instead lets the
    model concentrate on the few frames that carry evidence, while staying
    differentiable and order-agnostic (unlike max pooling, which keeps only
    one timestep and discards how many were implicated).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 4, 32)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)  # [B, T]
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)

    def attention_weights(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        """Per-timestep weights, exposed so Stage 7 can report *which*
        frames the model actually attended to."""
        scores = self.score(x).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        return torch.softmax(scores, dim=1)


def build_pooler(pooling: str, embed_dim: int) -> nn.Module:
    """Returns a module with signature (x, padding_mask) -> [B, d].

    `pooling` comes straight from the model config: "mean" | "max" |
    "attention".
    """
    if pooling == "attention":
        return AttentionPool(embed_dim)

    class _Functional(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, padding_mask=None):
            return self.fn(x, padding_mask)

    if pooling == "mean":
        return _Functional(masked_mean_pool)
    if pooling == "max":
        return _Functional(masked_max_pool)
    raise ValueError(f"Unknown pooling {pooling!r}; expected mean, max or attention")
