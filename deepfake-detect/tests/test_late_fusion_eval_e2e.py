"""End-to-end sanity check for Stage 4: trains tiny visual + acoustic
models to convergence on a trivially separable synthetic multimodal
dataset (same rows carry both a video_path and an audio_path), then runs
the late-fusion evaluator against their frozen checkpoints and checks the
fused AUC is high and results.md is written. Mechanics-only, like the
Stage 2/3 overfit tests -- real Stage 4 validation needs real data.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from src.evaluation.late_fusion_eval import evaluate_late_fusion
from src.training.train_acoustic import train_acoustic
from src.training.train_visual import train_visual
from src.utils.seed import set_seed


def _write_video(path: Path, value: int, num_frames: int = 4, size=(48, 48)) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    if not writer.isOpened():
        return False
    for _ in range(num_frames):
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()
    return True


def _write_tone(path: Path, freq: float, amplitude: float, duration_s=0.3, sample_rate=16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate)


def _build_split_csv(tmp_path: Path, name: str, n_real: int, n_fake: int) -> Path:
    rows = []
    for i in range(n_real):
        video_path = tmp_path / f"{name}_real_{i}.mp4"
        audio_path = tmp_path / f"{name}_real_{i}.wav"
        if not _write_video(video_path, value=10):
            pytest.skip("mp4v codec unavailable in this environment")
        _write_tone(audio_path, freq=110, amplitude=0.02)
        rows.append(
            {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "label": "real",
                "identity_id": f"{name}_r{i}",
            }
        )
    for i in range(n_fake):
        video_path = tmp_path / f"{name}_fake_{i}.mp4"
        audio_path = tmp_path / f"{name}_fake_{i}.wav"
        if not _write_video(video_path, value=240):
            pytest.skip("mp4v codec unavailable in this environment")
        _write_tone(audio_path, freq=4000, amplitude=0.8)
        rows.append(
            {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "label": "fake",
                "identity_id": f"{name}_f{i}",
            }
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.slow
def test_late_fusion_eval_end_to_end(tmp_path: Path):
    set_seed(0, deterministic=False)  # keep results independent of test execution order
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _build_split_csv(splits_dir, "train", n_real=6, n_fake=6)
    _build_split_csv(splits_dir, "val", n_real=4, n_fake=4)

    visual_model_cfg = {
        "backbone": "efficientnet_b0",
        "pretrained": False,
        "embed_dim": 32,
        "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
    }
    acoustic_model_cfg = {
        "n_mfcc": 13,
        "embed_dim": 32,
        "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
    }
    training_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_epochs": 1,
        "max_epochs": 6,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 3,
    }

    visual_cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs" / "visual_branch"),
            "data": {
                "splits_dir": str(splits_dir),
                "frame_rate": 8,
                "frame_size": 48,
                "num_frames": 4,
                "batch_size": 4,
                "use_class_balanced_sampler": False,
                "num_workers": 0,
            },
            "model": visual_model_cfg,
            "training": {
                **training_cfg,
                "checkpoint_dir": str(tmp_path / "outputs" / "visual_branch" / "checkpoints"),
            },
        }
    )
    acoustic_cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs" / "acoustic_branch"),
            "data": {
                "splits_dir": str(splits_dir),
                "audio_sample_rate": 16000,
                "audio_frame_ms": 25,
                "audio_hop_ms": 10,
                "num_audio_frames": 40,
                "batch_size": 4,
                "use_class_balanced_sampler": False,
                "num_workers": 0,
            },
            "model": acoustic_model_cfg,
            "training": {
                **training_cfg,
                "checkpoint_dir": str(tmp_path / "outputs" / "acoustic_branch" / "checkpoints"),
            },
        }
    )

    train_visual(visual_cfg)
    train_acoustic(acoustic_cfg)

    late_fusion_cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs" / "late_fusion"),
            "visual_checkpoint": str(
                tmp_path / "outputs" / "visual_branch" / "checkpoints" / "best.pt"
            ),
            "acoustic_checkpoint": str(
                tmp_path / "outputs" / "acoustic_branch" / "checkpoints" / "best.pt"
            ),
            "visual_weight": 0.5,
            "data": {
                "splits_dir": str(splits_dir),
                "frame_rate": 8,
                "frame_size": 48,
                "num_frames": 4,
                "audio_sample_rate": 16000,
                "audio_frame_ms": 25,
                "audio_hop_ms": 10,
                "num_audio_frames": 40,
                "batch_size": 4,
                "num_workers": 0,
            },
            "model": {"visual": visual_model_cfg, "acoustic": acoustic_model_cfg},
        }
    )

    metrics = evaluate_late_fusion(late_fusion_cfg)

    assert metrics["auc"] > 0.9, f"expected near-perfect fused AUC, got {metrics}"
    assert (tmp_path / "outputs" / "late_fusion" / "results.md").exists()
