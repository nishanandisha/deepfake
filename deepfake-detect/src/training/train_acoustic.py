"""Standalone acoustic-branch training loop (Stage 3 of the build plan).

`train_acoustic(cfg)` is the reusable entry point (called by scripts/train.py
via Hydra, and directly by tests with a plain config object).
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.acoustic.encoder import AcousticClassifier, AcousticEncoder
from src.models.acoustic.features import extract_acoustic_features
from src.preprocessing.cache import get_shared_cache
from src.preprocessing.dataset import AcousticDataset
from src.training.common import run_standalone_training
from src.utils.logging import ExperimentLogger, get_logger

# Total feature dim = 3 * n_mfcc (mfcc + delta + delta2) + 8 named scalar
# features (f0, voicing_confidence, 4 spectral stats, zcr, energy) -- see
# src/models/acoustic/features.py::_feature_names for the exact ordering.
NUM_SCALAR_FEATURES = 8


def acoustic_input_dim(n_mfcc: int) -> int:
    return 3 * n_mfcc + NUM_SCALAR_FEATURES


def build_acoustic_model(model_cfg) -> AcousticClassifier:
    encoder = AcousticEncoder(
        input_dim=acoustic_input_dim(model_cfg.n_mfcc),
        embed_dim=model_cfg.embed_dim,
        transformer_depth=model_cfg.transformer.depth,
        transformer_heads=model_cfg.transformer.heads,
        transformer_ff_dim=model_cfg.transformer.ff_dim,
        dropout=model_cfg.transformer.dropout,
    )
    return AcousticClassifier(encoder, embed_dim=model_cfg.embed_dim,
                              pooling=model_cfg.get("pooling", "mean"))


def train_acoustic(cfg, logger: ExperimentLogger = None) -> Dict[str, float]:
    logger = logger or get_logger("acoustic_branch", log_dir=Path(cfg.output_dir))

    splits_dir = Path(cfg.data.splits_dir)
    num_frames = cfg.data.get("num_audio_frames", 300)
    cache = get_shared_cache(cfg.data.get("cache_dir"), cfg.data.get("use_cache", True))
    train_dataset = AcousticDataset(
        splits_dir / "train.csv",
        split="train",
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=cfg.model.n_mfcc,
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
        num_frames=num_frames,
        seed=cfg.seed,
        cache=cache,
    )
    val_dataset = AcousticDataset(
        splits_dir / "val.csv",
        split="val",
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=cfg.model.n_mfcc,
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
        num_frames=num_frames,
        seed=cfg.seed,
        cache=cache,
    )

    # Sanity check called out explicitly by Stage 3: feature_names order
    # must match the tensor column order now, since Stage 7's SHAP will
    # attribute the decision back to these names.
    _, feature_names = extract_acoustic_features(
        np.zeros(cfg.data.audio_sample_rate), n_mfcc=cfg.model.n_mfcc
    )
    assert len(feature_names) == acoustic_input_dim(cfg.model.n_mfcc)

    model = build_acoustic_model(cfg.model)

    return run_standalone_training(
        cfg, logger, model, train_dataset, val_dataset, branch_name="Audio Transformer"
    )
