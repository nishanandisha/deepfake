"""Visual branch: per-frame CNN backbone + sinusoidal positional encoding +
Transformer encoder over the frame sequence, per Stage 2 of the build plan.

VisualEncoder produces Hv [B, T, d] (reused as-is by the Stage 5 fusion
module). VisualClassifier wraps it with the standalone pooling + MLP head
used for this branch's own auxiliary loss.
"""

import torch
import torch.nn as nn
import torchvision

from src.models.common import SinusoidalPositionalEncoding, build_pooler

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _build_backbone(name: str, pretrained: bool) -> tuple:
    """Returns (feature_extractor_module, output_dim)."""
    if name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = torchvision.models.efficientnet_b0(weights=weights)
        return nn.Sequential(net.features, net.avgpool, nn.Flatten(1)), 1280
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        modules = list(net.children())[:-1]  # drop fc
        return nn.Sequential(*modules, nn.Flatten(1)), 512
    raise ValueError(f"Unsupported visual backbone: {name!r}")


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        embed_dim: int = 512,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone, backbone_dim = _build_backbone(backbone, pretrained)
        self.input_proj = nn.Linear(backbone_dim, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.register_buffer("imagenet_mean", IMAGENET_MEAN.view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", IMAGENET_STD.view(1, 1, 3, 1, 1), persistent=False)

    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] float in [0, 1] -> ImageNet-normalized."""
        return (frames - self.imagenet_mean) / self.imagenet_std

    def forward(self, frames: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """frames: [B, T, 3, H, W] float in [0, 1].
        padding_mask: [B, T] bool, True at PADDED (invalid) positions.
        Returns Hv: [B, T, embed_dim].
        """
        batch_size, seq_len = frames.shape[:2]
        frames = self.normalize(frames)

        flat_frames = frames.reshape(batch_size * seq_len, *frames.shape[2:])
        features = self.backbone(flat_frames)
        features = self.input_proj(features).view(batch_size, seq_len, -1)

        features = self.pos_encoding(features)
        return self.transformer(features, src_key_padding_mask=padding_mask)


class VisualClassifier(nn.Module):
    """Standalone visual branch: encoder + temporal mean pool + MLP head,
    producing a single manipulation-probability logit for this branch alone.
    """

    def __init__(self, encoder: VisualEncoder, embed_dim: int = 512, pooling: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.pool = build_pooler(pooling, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, frames: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns raw logits [B] (apply sigmoid for probability)."""
        hv = self.encoder(frames, padding_mask=padding_mask)
        return self.head(self.pool(hv, padding_mask)).squeeze(-1)
