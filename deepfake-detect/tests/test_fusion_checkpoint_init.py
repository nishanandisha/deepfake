from pathlib import Path

import torch

from src.models.fusion.cross_attention import CrossModalFusion, load_standalone_checkpoint_into
from src.training.train_acoustic import build_acoustic_model
from src.training.train_visual import build_visual_model


def _visual_model_cfg():
    return {
        "backbone": "efficientnet_b0",
        "pretrained": False,
        "embed_dim": 32,
        "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
    }


def _acoustic_model_cfg():
    return {
        "n_mfcc": 13,
        "embed_dim": 32,
        "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
    }


def test_visual_checkpoint_loads_into_fusion_submodules(tmp_path: Path):
    from omegaconf import OmegaConf

    standalone = build_visual_model(OmegaConf.create(_visual_model_cfg()))
    checkpoint_path = tmp_path / "visual_best.pt"
    torch.save(standalone.state_dict(), checkpoint_path)

    fusion = CrossModalFusion(
        visual_encoder=build_visual_model(OmegaConf.create(_visual_model_cfg())).encoder,
        acoustic_encoder=build_acoustic_model(OmegaConf.create(_acoustic_model_cfg())).encoder,
        embed_dim=32, cross_attention_heads=4, cross_attention_dropout=0.0, gate_hidden_dim=16,
    )

    load_standalone_checkpoint_into(fusion.visual_encoder, fusion.visual_aux_head, checkpoint_path)

    for (name, standalone_param), (_, fusion_param) in zip(
        standalone.encoder.state_dict().items(), fusion.visual_encoder.state_dict().items()
    ):
        assert torch.allclose(standalone_param, fusion_param), f"mismatch in {name}"
    for (name, standalone_param), (_, fusion_param) in zip(
        standalone.head.state_dict().items(), fusion.visual_aux_head.state_dict().items()
    ):
        assert torch.allclose(standalone_param, fusion_param), f"mismatch in {name}"


def test_acoustic_checkpoint_loads_into_fusion_submodules(tmp_path: Path):
    from omegaconf import OmegaConf

    standalone = build_acoustic_model(OmegaConf.create(_acoustic_model_cfg()))
    checkpoint_path = tmp_path / "acoustic_best.pt"
    torch.save(standalone.state_dict(), checkpoint_path)

    fusion = CrossModalFusion(
        visual_encoder=build_visual_model(OmegaConf.create(_visual_model_cfg())).encoder,
        acoustic_encoder=build_acoustic_model(OmegaConf.create(_acoustic_model_cfg())).encoder,
        embed_dim=32, cross_attention_heads=4, cross_attention_dropout=0.0, gate_hidden_dim=16,
    )

    load_standalone_checkpoint_into(
        fusion.acoustic_encoder, fusion.acoustic_aux_head, checkpoint_path
    )

    for (name, standalone_param), (_, fusion_param) in zip(
        standalone.encoder.state_dict().items(), fusion.acoustic_encoder.state_dict().items()
    ):
        assert torch.allclose(standalone_param, fusion_param), f"mismatch in {name}"
