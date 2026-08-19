"""Stage 5: bidirectional cross-attention + gated fusion, the core
contribution. Reuses the Stage 2/3 encoder classes directly (not copies),
and can warm-start from their trained standalone checkpoints (see
`load_standalone_checkpoint_into`).

Architecture (mirrors the build plan):
  Hv [B,T,d] <- VisualEncoder,  Ha [B,S,d] <- AcousticEncoder
  H_v_to_a = LayerNorm(Hv + Attention(query=Hv, key=Ha, value=Ha))
  H_a_to_v = LayerNorm(Ha + Attention(query=Ha, key=Hv, value=Hv))
  pooled_v, pooled_a = masked_mean_pool(H_v_to_a), masked_mean_pool(H_a_to_v)
  gate = sigmoid(MLP([pooled_v; pooled_a]))          # audio/visual weighting
  z = gate * pooled_v + (1 - gate) * pooled_a
  y_hat = MLP_head(z)                                 # authenticity = 1 - sigmoid(y_hat)

Auxiliary heads (y_hat_visual, y_hat_acoustic) apply the same
standalone-head architecture directly to the *un-attended* Hv/Ha, so they
can be warm-started from the Stage 2/3 checkpoints and reflect each
branch's own unimodal signal, per the build plan's auxiliary-loss design.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.acoustic.encoder import AcousticEncoder
from src.models.common import build_pooler
from src.models.visual.encoder import VisualEncoder


def _make_standalone_head(embed_dim: int) -> nn.Sequential:
    """Matches VisualClassifier.head / AcousticClassifier.head exactly, so
    a Stage 2/3 checkpoint's head weights can be loaded directly."""
    return nn.Sequential(
        nn.Linear(embed_dim, embed_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(embed_dim // 2, 1),
    )


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        visual_encoder: VisualEncoder,
        acoustic_encoder: AcousticEncoder,
        embed_dim: int = 512,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        gate_hidden_dim: int = 128,
        pooling: str = "mean",
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.acoustic_encoder = acoustic_encoder
        # Four independent poolers: the two attended cross-modal streams and
        # the two un-attended streams feeding the auxiliary heads. Attention
        # pooling is parameterised, so they must not be shared.
        self.pool_v = build_pooler(pooling, embed_dim)
        self.pool_a = build_pooler(pooling, embed_dim)
        self.pool_hv = build_pooler(pooling, embed_dim)
        self.pool_ha = build_pooler(pooling, embed_dim)

        self.visual_aux_head = _make_standalone_head(embed_dim)
        self.acoustic_aux_head = _make_standalone_head(embed_dim)

        self.v_to_a_attention = nn.MultiheadAttention(
            embed_dim, cross_attention_heads, dropout=cross_attention_dropout, batch_first=True
        )
        self.a_to_v_attention = nn.MultiheadAttention(
            embed_dim, cross_attention_heads, dropout=cross_attention_dropout, batch_first=True
        )
        self.v_to_a_norm = nn.LayerNorm(embed_dim)
        self.a_to_v_norm = nn.LayerNorm(embed_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.fusion_head = _make_standalone_head(embed_dim)

    def encode(
        self,
        frames: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        acoustic_features: torch.Tensor,
        acoustic_padding_mask: torch.Tensor,
    ) -> dict:
        """Everything up to (but not including) the gate/fusion head.
        Split out from forward() so Stage 7's SHAP can attribute over the
        pooled per-modality vectors without re-running the encoders for
        every coalition sample."""
        hv = self.visual_encoder(frames, padding_mask=visual_padding_mask)  # [B,T,d]
        ha = self.acoustic_encoder(acoustic_features, padding_mask=acoustic_padding_mask)  # [B,S,d]

        attended_v, _ = self.v_to_a_attention(
            query=hv, key=ha, value=ha, key_padding_mask=acoustic_padding_mask
        )
        h_v_to_a = self.v_to_a_norm(hv + attended_v)

        attended_a, _ = self.a_to_v_attention(
            query=ha, key=hv, value=hv, key_padding_mask=visual_padding_mask
        )
        h_a_to_v = self.a_to_v_norm(ha + attended_a)

        return {
            "pooled_v": self.pool_v(h_v_to_a, visual_padding_mask),
            "pooled_a": self.pool_a(h_a_to_v, acoustic_padding_mask),
            "pooled_hv": self.pool_hv(hv, visual_padding_mask),
            "pooled_ha": self.pool_ha(ha, acoustic_padding_mask),
        }

    def fuse_from_pooled(self, pooled_v: torch.Tensor, pooled_a: torch.Tensor) -> tuple:
        """Gate + fusion head applied to already-pooled modality vectors.
        Returns (y_hat_logit, gate)."""
        gate = torch.sigmoid(self.gate_mlp(torch.cat([pooled_v, pooled_a], dim=-1))).squeeze(-1)
        z = gate.unsqueeze(-1) * pooled_v + (1 - gate).unsqueeze(-1) * pooled_a
        return self.fusion_head(z).squeeze(-1), gate

    def forward(
        self,
        frames: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        acoustic_features: torch.Tensor,
        acoustic_padding_mask: torch.Tensor,
    ) -> dict:
        encoded = self.encode(
            frames, visual_padding_mask, acoustic_features, acoustic_padding_mask
        )
        y_hat_logit, gate = self.fuse_from_pooled(encoded["pooled_v"], encoded["pooled_a"])

        return {
            "y_hat_logit": y_hat_logit,
            "y_hat_visual_logit": self.visual_aux_head(encoded["pooled_hv"]).squeeze(-1),
            "y_hat_acoustic_logit": self.acoustic_aux_head(encoded["pooled_ha"]).squeeze(-1),
            "gate": gate,  # per-sample visual weighting in [0,1]; 1-gate = acoustic weighting
            "pooled_v": encoded["pooled_v"],
            "pooled_a": encoded["pooled_a"],
        }


def load_standalone_checkpoint_into(
    encoder: nn.Module,
    aux_head: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> None:
    """Warm-starts a fusion sub-branch from a Stage 2/3 standalone
    VisualClassifier/AcousticClassifier checkpoint (keys "encoder.*" /
    "head.*") by splitting it into the fusion model's encoder and aux_head.
    """
    state_dict = torch.load(checkpoint_path, map_location=device or "cpu")
    encoder_state = {
        k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")
    }
    head_state = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
    encoder.load_state_dict(encoder_state)
    aux_head.load_state_dict(head_state)
