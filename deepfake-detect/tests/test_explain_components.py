"""Unit tests for the Stage 7 explanation components: Grad-CAM, the
report generator, and the SHAP helpers, exercised against tiny models so
they stay fast.
"""

from pathlib import Path

import numpy as np
import torch

from src.explain.cam_visual import GradCAM, most_implicated_frames, overlay_heatmap
from src.explain.report import build_report_html, write_report
from src.explain.shap_acoustic import top_k_features
from src.explain.shap_modality import explain_modality_split
from src.models.acoustic.encoder import AcousticEncoder
from src.models.fusion.cross_attention import CrossModalFusion
from src.models.visual.encoder import VisualEncoder
from src.utils.seed import set_seed


def _make_fusion_model(embed_dim=32):
    visual_encoder = VisualEncoder(
        backbone="efficientnet_b0", pretrained=False, embed_dim=embed_dim,
        transformer_depth=1, transformer_heads=4, transformer_ff_dim=32, dropout=0.0,
    )
    acoustic_encoder = AcousticEncoder(
        input_dim=47, embed_dim=embed_dim,
        transformer_depth=1, transformer_heads=4, transformer_ff_dim=32, dropout=0.0,
    )
    return CrossModalFusion(
        visual_encoder, acoustic_encoder, embed_dim=embed_dim,
        cross_attention_heads=4, cross_attention_dropout=0.0, gate_hidden_dim=16,
    )


def test_gradcam_produces_normalized_per_frame_heatmaps():
    set_seed(0, deterministic=False)
    model = _make_fusion_model()
    frames = torch.rand(1, 3, 3, 64, 64)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    cam = GradCAM(model.visual_encoder)
    heatmaps = cam(frames, mask, model.visual_aux_head)

    assert heatmaps.shape == (3, 64, 64)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0


def test_gradcam_works_on_frozen_model():
    """Regression: the real pipeline runs Grad-CAM on a checkpoint loaded
    via load_frozen_checkpoint (requires_grad=False everywhere), which used
    to fail with "cannot register a hook on a tensor that doesn't require
    gradient"."""
    set_seed(0, deterministic=False)
    model = _make_fusion_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    cam = GradCAM(model.visual_encoder)
    heatmaps = cam(
        torch.rand(1, 2, 3, 64, 64), torch.zeros(1, 2, dtype=torch.bool), model.visual_aux_head
    )

    assert heatmaps.shape == (2, 64, 64)
    # The freeze must survive the CAM pass -- Grad-CAM must not unfreeze the
    # model as a side effect, or later stages would silently start training.
    assert all(not p.requires_grad for p in model.parameters())


def test_gradcam_works_inside_no_grad_context():
    """Regression: the pipeline computes predictions under torch.no_grad();
    Grad-CAM must still be able to run its own backward pass."""
    set_seed(0, deterministic=False)
    model = _make_fusion_model()
    for param in model.parameters():
        param.requires_grad = False

    cam = GradCAM(model.visual_encoder)
    with torch.no_grad():
        heatmaps = cam(
            torch.rand(1, 2, 3, 64, 64), torch.zeros(1, 2, dtype=torch.bool),
            model.visual_aux_head,
        )

    assert heatmaps.shape == (2, 64, 64)


def test_gradcam_removes_hooks_after_use():
    set_seed(0, deterministic=False)
    model = _make_fusion_model()
    cam = GradCAM(model.visual_encoder)
    cam(torch.rand(1, 2, 3, 64, 64), torch.zeros(1, 2, dtype=torch.bool), model.visual_aux_head)

    # Leaked forward hooks would silently accumulate activations on every
    # later forward pass, so assert they're cleaned up.
    assert cam._handles == []


def test_most_implicated_frames_ranks_by_mean_saliency():
    heatmaps = np.zeros((4, 8, 8))
    heatmaps[2] = 1.0  # clearly the most salient frame
    heatmaps[0] = 0.5

    ranked = most_implicated_frames(heatmaps, top_k=2)

    assert ranked[0][0] == 2
    assert ranked[1][0] == 0


def test_overlay_heatmap_preserves_shape_and_dtype():
    frame = np.full((32, 32, 3), 120, dtype=np.uint8)
    heatmap = np.linspace(0, 1, 32 * 32).reshape(32, 32)

    overlaid = overlay_heatmap(frame, heatmap)

    assert overlaid.shape == frame.shape
    assert overlaid.dtype == np.uint8
    assert not np.array_equal(overlaid, frame)  # the overlay actually changed pixels


def test_top_k_features_sorts_by_absolute_value():
    attributions = {"f0": 0.1, "spectral_flatness": -0.9, "mfcc_0": 0.4, "zcr": 0.01}

    top = top_k_features(attributions, k=2)

    assert top[0] == ("spectral_flatness", -0.9)  # largest |value|, despite being negative
    assert top[1] == ("mfcc_0", 0.4)


def test_build_report_html_contains_key_sections(tmp_path: Path):
    modality_split = {
        "visual_share": 0.75, "acoustic_share": 0.25,
        "shap_visual": 0.6, "shap_acoustic": -0.2, "base_value": 0.1,
    }
    frames = [np.full((16, 16, 3), 200, dtype=np.uint8)]

    html = build_report_html(
        sample_id="clip_42",
        authenticity_score=0.31,
        decision="flag",
        modality_split=modality_split,
        top_acoustic_features=[("spectral_flatness", -0.9), ("f0", 0.3)],
        saliency_frames=frames,
        frame_indices=[7],
        ground_truth_label="fake",
        ground_truth_modality="video",
    )

    assert "clip_42" in html
    assert "FLAG" in html
    assert "0.3100" in html
    assert "spectral_flatness" in html
    assert "75.0" in html  # visual share percentage
    assert "data:image/png;base64," in html  # frame embedded, not a broken link
    assert "fake" in html

    output_path = tmp_path / "report.html"
    write_report(str(output_path), html)
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == html


def test_explain_modality_split_returns_normalized_shares():
    set_seed(0, deterministic=False)
    model = _make_fusion_model()
    device = torch.device("cpu")

    pooled_v = torch.rand(2, 32)
    pooled_a = torch.rand(2, 32)
    background_v = torch.zeros(32)
    background_a = torch.zeros(32)

    splits = explain_modality_split(
        model, pooled_v, pooled_a, background_v, background_a, device, n_samples=16
    )

    assert len(splits) == 2
    for split in splits:
        assert 0.0 <= split["visual_share"] <= 1.0
        assert abs(split["visual_share"] + split["acoustic_share"] - 1.0) < 1e-9
