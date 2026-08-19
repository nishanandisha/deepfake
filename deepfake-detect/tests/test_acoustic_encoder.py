import torch

from src.models.acoustic.encoder import AcousticClassifier, AcousticEncoder
from src.models.common import masked_mean_pool


def _make_encoder(input_dim=47, embed_dim=32):
    return AcousticEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        transformer_depth=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
    )


def test_acoustic_encoder_output_shape():
    encoder = _make_encoder()
    features = torch.rand(2, 10, 47)

    ha = encoder(features)
    assert ha.shape == (2, 10, 32)


def test_acoustic_encoder_respects_padding_mask():
    encoder = _make_encoder()
    features = torch.rand(2, 10, 47)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 6:] = True

    ha = encoder(features, padding_mask=mask)
    assert ha.shape == (2, 10, 32)


def test_acoustic_classifier_output_shape():
    encoder = _make_encoder()
    classifier = AcousticClassifier(encoder, embed_dim=32)
    features = torch.rand(3, 10, 47)

    logits = classifier(features)
    assert logits.shape == (3,)


def test_masked_mean_pool_matches_manual_average():
    x = torch.arange(12).float().view(1, 4, 3)
    mask = torch.tensor([[False, False, False, True]])
    pooled = masked_mean_pool(x, mask)
    expected = x[0, :3].mean(dim=0, keepdim=True)
    assert torch.allclose(pooled, expected)
