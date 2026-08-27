"""The acoustic model: linear projection of named per-frame features +
sinusoidal positional encoding + Transformer encoder.

AcousticEncoder produces the sequence representation Ha [B, S, d].
AcousticClassifier wraps it with pooling + an MLP head, producing the single
manipulation logit this package predicts from.
"""

import torch
import torch.nn as nn

from src.models.common import SinusoidalPositionalEncoding, build_pooler


class AcousticEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Standardise each named descriptor before projecting.
        #
        # The raw feature columns span ~5 orders of magnitude -- measured on
        # real LAV-DF audio, short_time_energy has std 0.02 while
        # spectral_rolloff has std 1956 (a 114,000x range). Fed to a bare
        # Linear, the large-magnitude columns dominate the projection and
        # the small ones contribute no usable gradient; training loss then
        # sits at ln(2) forever because the model cannot even overfit.
        # The Transformer's internal LayerNorm cannot fix this: it acts
        # *after* input_proj has already mixed the columns together.
        #
        # BatchNorm1d over the feature axis learns per-descriptor running
        # statistics, which keeps every named feature on a comparable scale
        # without needing a separate precomputed-statistics artefact.
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """features: [B, S, input_dim]. padding_mask: [B, S] bool, True at
        PADDED (invalid) positions. Returns Ha: [B, S, embed_dim]."""
        # BatchNorm1d expects [B, C, S], so move the feature axis to dim 1
        # and back. Padded frames are zeros and do shift the running stats
        # slightly; that is acceptable here because every clip in a batch is
        # padded to the same length, so the bias is uniform across samples.
        x = self.input_norm(features.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        return self.transformer(x, src_key_padding_mask=padding_mask)


class AcousticClassifier(nn.Module):
    """Standalone acoustic branch: encoder + temporal mean pool + MLP head,
    producing a single manipulation-probability logit for this branch alone.
    """

    def __init__(self, encoder: AcousticEncoder, embed_dim: int = 512, pooling: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.pool = build_pooler(pooling, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns raw logits [B] (apply sigmoid for probability)."""
        ha = self.encoder(features, padding_mask=padding_mask)
        return self.head(self.pool(ha, padding_mask)).squeeze(-1)
