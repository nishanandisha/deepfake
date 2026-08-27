"""Trainable classifier over frozen WavLM embeddings.

Attentive statistics pooling (Okabe et al., 2018) rather than a mean: a
learned per-frame weighting lets the model concentrate on the frames that
actually carry synthesis artifacts, and carrying the weighted *standard
deviation* alongside the mean preserves how much the evidence varies across
the clip -- a mean alone collapses "uniformly slightly odd" and "briefly very
odd" onto the same vector.

Kept small on purpose. The predecessor project documented a 4-layer d=512
encoder collapsing to a constant function on ~1,400 clips; this corpus is
1,866, so the same trap is one careless config edit away. ~1M parameters over
a frozen 94M frontend is the intended ratio.
"""

import torch
import torch.nn as nn


class AttentiveStatsPooling(nn.Module):
    """[B, T, D] (+ mask) -> [B, 2D] as concatenated weighted mean and std."""

    def __init__(self, input_dim: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, input_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Conv1d wants [B, D, T].
        h = x.transpose(1, 2)
        weights = self.attention(h)

        if padding_mask is not None:
            # True marks padding. Masked frames must not merely get a small
            # weight, they must get zero after softmax.
            weights = weights.masked_fill(padding_mask.unsqueeze(1), float("-inf"))

        weights = torch.softmax(weights, dim=2)

        mean = torch.sum(weights * h, dim=2)
        variance = torch.sum(weights * h.pow(2), dim=2) - mean.pow(2)
        std = torch.sqrt(variance.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)


class VoiceClassifierHead(nn.Module):
    """Frozen-embedding classifier emitting one logit: high = AI-generated."""

    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 256,
        bottleneck: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # LayerNorm first: WavLM activation scale varies a lot by layer, and
        # normalising here means the same head config works whichever layer
        # the sweep selects.
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pooling = AttentiveStatsPooling(proj_dim, bottleneck)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(proj_dim * 2),
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.proj(self.input_norm(x))
        pooled = self.pooling(h, padding_mask)
        return self.classifier(pooled).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
