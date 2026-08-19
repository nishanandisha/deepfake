import torch

from src.models.acoustic.encoder import AcousticEncoder
from src.models.fusion.cross_attention import CrossModalFusion
from src.models.visual.encoder import VisualEncoder


def _make_model(embed_dim=32):
    visual_encoder = VisualEncoder(
        backbone="efficientnet_b0", pretrained=False, embed_dim=embed_dim,
        transformer_depth=2, transformer_heads=4, transformer_ff_dim=64, dropout=0.0,
    )
    acoustic_encoder = AcousticEncoder(
        input_dim=47, embed_dim=embed_dim,
        transformer_depth=2, transformer_heads=4, transformer_ff_dim=64, dropout=0.0,
    )
    return CrossModalFusion(
        visual_encoder, acoustic_encoder, embed_dim=embed_dim,
        cross_attention_heads=4, cross_attention_dropout=0.0, gate_hidden_dim=16,
    )


def test_forward_output_shapes_and_keys():
    model = _make_model()
    frames = torch.rand(3, 5, 3, 64, 64)
    v_mask = torch.zeros(3, 5, dtype=torch.bool)
    features = torch.rand(3, 10, 47)
    a_mask = torch.zeros(3, 10, dtype=torch.bool)

    out = model(frames, v_mask, features, a_mask)

    # pooled_v/pooled_a are exposed for Stage 7's SHAP attribution; they are
    # [B, d] rather than per-sample scalars like the logits and the gate.
    scalar_keys = {"y_hat_logit", "y_hat_visual_logit", "y_hat_acoustic_logit", "gate"}
    assert scalar_keys <= set(out.keys())
    assert {"pooled_v", "pooled_a"} <= set(out.keys())

    for key in scalar_keys:
        assert out[key].shape == (3,)
    assert out["pooled_v"].shape == (3, 32)
    assert out["pooled_a"].shape == (3, 32)


def test_gate_is_bounded_in_unit_interval():
    model = _make_model()
    frames = torch.rand(4, 5, 3, 64, 64)
    v_mask = torch.zeros(4, 5, dtype=torch.bool)
    features = torch.rand(4, 10, 47)
    a_mask = torch.zeros(4, 10, dtype=torch.bool)

    out = model(frames, v_mask, features, a_mask)
    assert torch.all(out["gate"] >= 0.0)
    assert torch.all(out["gate"] <= 1.0)


def test_handles_padding_masks_without_error():
    model = _make_model()
    frames = torch.rand(2, 6, 3, 64, 64)
    v_mask = torch.zeros(2, 6, dtype=torch.bool)
    v_mask[0, 4:] = True
    features = torch.rand(2, 8, 47)
    a_mask = torch.zeros(2, 8, dtype=torch.bool)
    a_mask[1, 6:] = True

    out = model(frames, v_mask, features, a_mask)
    assert out["y_hat_logit"].shape == (2,)


def test_gradients_flow_to_both_encoders():
    model = _make_model()
    frames = torch.rand(2, 5, 3, 64, 64)
    v_mask = torch.zeros(2, 5, dtype=torch.bool)
    features = torch.rand(2, 10, 47)
    a_mask = torch.zeros(2, 10, dtype=torch.bool)

    out = model(frames, v_mask, features, a_mask)
    loss = (
        out["y_hat_logit"].sum()
        + out["y_hat_visual_logit"].sum()
        + out["y_hat_acoustic_logit"].sum()
    )
    loss.backward()

    def grad_norm(module):
        return sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)

    assert grad_norm(model.visual_encoder) > 0
    assert grad_norm(model.acoustic_encoder) > 0
