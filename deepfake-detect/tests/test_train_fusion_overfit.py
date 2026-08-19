"""End-to-end sanity check for Stage 5, mirroring the Stage 2/3/4 overfit
tests: trains the cross-modal fusion model from scratch (no Stage 2/3
checkpoints needed here -- see test_fusion_checkpoint_init.py for the
warm-start path) on a tiny synthetic multimodal dataset with a trivially
learnable signal, and asserts it learns to discriminate, that the gate
value is logged, and that a missing Stage 4 late-fusion comparison is
handled gracefully (reported as unavailable, not a crash).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from src.training.train_fusion import train_fusion
from src.utils.seed import set_seed


def _write_video(path: Path, value: int, num_frames=4, size=(48, 48)) -> bool:
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
            {"video_path": str(video_path), "audio_path": str(audio_path), "label": "real",
             "identity_id": f"{name}_r{i}"}
        )
    for i in range(n_fake):
        video_path = tmp_path / f"{name}_fake_{i}.mp4"
        audio_path = tmp_path / f"{name}_fake_{i}.wav"
        if not _write_video(video_path, value=240):
            pytest.skip("mp4v codec unavailable in this environment")
        _write_tone(audio_path, freq=4000, amplitude=0.8)
        rows.append(
            {"video_path": str(video_path), "audio_path": str(audio_path), "label": "fake",
             "identity_id": f"{name}_f{i}"}
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.slow
def test_fusion_training_loop_learns_trivial_signal(tmp_path: Path):
    set_seed(0, deterministic=False)  # keep results independent of test execution order
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _build_split_csv(splits_dir, "train", n_real=6, n_fake=6)
    _build_split_csv(splits_dir, "val", n_real=4, n_fake=4)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs" / "fusion"),
            "init_from_standalone_checkpoints": False,
            "late_fusion_results": str(tmp_path / "outputs" / "late_fusion" / "results.json"),
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
                "use_class_balanced_sampler": False,
                "num_workers": 0,
            },
            "model": {
                "visual": {
                    "backbone": "efficientnet_b0", "pretrained": False, "embed_dim": 32,
                    "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
                },
                "acoustic": {
                    "n_mfcc": 13, "embed_dim": 32,
                    "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
                },
                "cross_attention": {"heads": 4, "dropout": 0.0},
                "gate": {"hidden_dim": 16},
                "aux_loss": {"lambda_visual": 0.3, "lambda_acoustic": 0.3},
            },
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_epochs": 1,
                "max_epochs": 6,
                "grad_clip_norm": 1.0,
                "checkpoint_dir": str(tmp_path / "outputs" / "fusion" / "checkpoints"),
                "early_stopping_patience": 3,
            },
        }
    )

    metrics = train_fusion(cfg)

    assert metrics["auc"] > 0.9, f"expected near-perfect AUC on separable data, got {metrics}"
    assert 0.0 <= metrics["avg_gate_visual_weight"] <= 1.0
    assert (tmp_path / "outputs" / "fusion" / "checkpoints" / "best.pt").exists()

    results_text = (tmp_path / "outputs" / "fusion" / "results.md").read_text()
    # Stage 4 late-fusion results weren't produced in this isolated test.
    assert "not available" in results_text
