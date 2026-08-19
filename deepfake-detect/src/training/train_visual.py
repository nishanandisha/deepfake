"""Standalone visual-branch training loop (Stage 2 of the build plan).

`train_visual(cfg)` is the reusable entry point (called by scripts/train.py
via Hydra, and directly by tests with a plain config object) so the
training logic isn't locked behind Hydra's CLI machinery.
"""

import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.visual.encoder import VisualClassifier, VisualEncoder
from src.preprocessing.cache import get_shared_cache
from src.preprocessing.dataset import VisualDataset
from src.training.common import run_standalone_training
from src.utils.logging import ExperimentLogger, get_logger


def build_visual_model(model_cfg) -> VisualClassifier:
    encoder = VisualEncoder(
        backbone=model_cfg.backbone,
        pretrained=model_cfg.pretrained,
        embed_dim=model_cfg.embed_dim,
        transformer_depth=model_cfg.transformer.depth,
        transformer_heads=model_cfg.transformer.heads,
        transformer_ff_dim=model_cfg.transformer.ff_dim,
        dropout=model_cfg.transformer.dropout,
    )
    return VisualClassifier(encoder, embed_dim=model_cfg.embed_dim,
                            pooling=model_cfg.get("pooling", "mean"))


def train_visual(cfg, logger: ExperimentLogger = None) -> Dict[str, float]:
    logger = logger or get_logger("visual_branch", log_dir=Path(cfg.output_dir))

    splits_dir = Path(cfg.data.splits_dir)
    num_frames = cfg.data.get("num_frames", 32)
    cache = get_shared_cache(cfg.data.get("cache_dir"), cfg.data.get("use_cache", True))
    train_dataset = VisualDataset(
        splits_dir / "train.csv",
        split="train",
        frame_rate=cfg.data.frame_rate,
        frame_size=cfg.data.frame_size,
        num_frames=num_frames,
        seed=cfg.seed,
        cache=cache,
    )
    val_dataset = VisualDataset(
        splits_dir / "val.csv",
        split="val",
        frame_rate=cfg.data.frame_rate,
        frame_size=cfg.data.frame_size,
        num_frames=num_frames,
        seed=cfg.seed,
        cache=cache,
    )

    model = build_visual_model(cfg.model)

    return run_standalone_training(
        cfg, logger, model, train_dataset, val_dataset, branch_name="Visual CNN + Transformer"
    )
