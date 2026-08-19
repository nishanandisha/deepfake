"""Stage 4: evaluate the late-fusion (probability-averaging) baseline on
the val split, using frozen Stage 2/3 checkpoints. Neither branch is
modified here -- see src/models/fusion/late_fusion.py.
"""

import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.results import write_results_markdown
from src.models.fusion.late_fusion import (
    average_probabilities,
    load_frozen_checkpoint,
    predict_branch_probabilities,
    resolve_device,
)
from src.preprocessing.dataset import AcousticDataset, VisualDataset
from src.training.metrics import compute_binary_classification_metrics
from src.training.train_acoustic import build_acoustic_model
from src.training.train_visual import build_visual_model
from src.utils.logging import ExperimentLogger, get_logger


def evaluate_late_fusion(cfg, logger: ExperimentLogger = None) -> Dict[str, float]:
    logger = logger or get_logger("late_fusion", log_dir=Path(cfg.output_dir))
    device = resolve_device(cfg.device)

    splits_dir = Path(cfg.data.splits_dir)
    visual_dataset = VisualDataset(
        splits_dir / "val.csv",
        split="val",
        frame_rate=cfg.data.frame_rate,
        frame_size=cfg.data.frame_size,
        num_frames=cfg.data.get("num_frames", 32),
        seed=cfg.seed,
    )
    acoustic_dataset = AcousticDataset(
        splits_dir / "val.csv",
        split="val",
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=cfg.model.acoustic.n_mfcc,
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
        num_frames=cfg.data.get("num_audio_frames", 300),
        seed=cfg.seed,
    )
    assert len(visual_dataset) == len(acoustic_dataset), (
        "visual and acoustic val datasets must come from the same manifest rows "
        "for late fusion to pair probabilities correctly"
    )

    visual_model = load_frozen_checkpoint(
        build_visual_model(cfg.model.visual), cfg.visual_checkpoint, device
    )
    acoustic_model = load_frozen_checkpoint(
        build_acoustic_model(cfg.model.acoustic), cfg.acoustic_checkpoint, device
    )

    logger.info(f"Running frozen visual branch over {len(visual_dataset)} val samples")
    visual_probs = predict_branch_probabilities(
        visual_model, visual_dataset, device, batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    logger.info(f"Running frozen acoustic branch over {len(acoustic_dataset)} val samples")
    acoustic_probs = predict_branch_probabilities(
        acoustic_model, acoustic_dataset, device, batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    fused_probs = average_probabilities(visual_probs, acoustic_probs, cfg.visual_weight)
    y_true = [1 if label == "fake" else 0 for label in visual_dataset.df["label"]]

    metrics = compute_binary_classification_metrics(y_true, fused_probs)
    logger.info(f"Late fusion (avg.) val metrics: {metrics}")

    write_results_markdown(
        output_path=str(Path(cfg.output_dir) / "results.md"),
        branch_name="Late fusion (avg.)",
        metrics=metrics,
        data_source_note=(
            f"val split from {splits_dir}, frozen checkpoints "
            f"{cfg.visual_checkpoint} + {cfg.acoustic_checkpoint}, "
            f"visual_weight={cfg.visual_weight}"
        ),
    )
    return metrics
