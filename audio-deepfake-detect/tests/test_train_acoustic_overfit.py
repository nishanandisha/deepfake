"""End-to-end sanity check for the training loop: trains on a tiny synthetic
dataset with a trivially learnable signal (quiet low-frequency "real" tones
vs. loud high-frequency "fake" tones) and asserts the model learns to
discriminate.

Confirms training mechanics only. It says nothing about whether the model
detects real deepfakes -- that takes a real dataset and scripts/evaluate.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from src.training.train_acoustic import train_acoustic
from src.utils.seed import set_seed


def _write_tone(path: Path, freq: float, amplitude: float, duration_s: float, sample_rate=16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), signal, sample_rate)


def _build_split_csv(tmp_path: Path, name: str, n_real: int, n_fake: int) -> Path:
    rows = []
    for i in range(n_real):
        path = tmp_path / f"{name}_real_{i}.wav"
        _write_tone(path, freq=110, amplitude=0.02, duration_s=0.3)
        rows.append({"audio_path": str(path), "label": "real", "identity_id": f"{name}_r{i}"})
    for i in range(n_fake):
        path = tmp_path / f"{name}_fake_{i}.wav"
        _write_tone(path, freq=4000, amplitude=0.8, duration_s=0.3)
        rows.append({"audio_path": str(path), "label": "fake", "identity_id": f"{name}_f{i}"})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.slow
def test_acoustic_training_loop_learns_trivial_signal(tmp_path: Path):
    set_seed(0, deterministic=False)  # keep results independent of test execution order
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _build_split_csv(splits_dir, "train", n_real=6, n_fake=6)
    _build_split_csv(splits_dir, "val", n_real=4, n_fake=4)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs"),
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
            "model": {
                "n_mfcc": 13,
                "embed_dim": 32,
                "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
            },
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_epochs": 1,
                "max_epochs": 6,
                "grad_clip_norm": 1.0,
                "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
                "early_stopping_patience": 3,
            },
        }
    )

    metrics = train_acoustic(cfg)

    assert metrics["auc"] > 0.9, f"expected near-perfect AUC on separable data, got {metrics}"
    assert (tmp_path / "outputs" / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "outputs" / "results.md").exists()
