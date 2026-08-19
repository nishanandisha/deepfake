"""End-to-end sanity check for the Stage 2 training loop: since real
FakeAVCeleb data isn't available yet, this trains on a tiny synthetic
dataset with a trivially learnable signal (near-black "real" clips vs.
near-white "fake" clips) and asserts the model actually learns to
discriminate. This proves the training mechanics (data loading, loss,
optimizer, scheduler, checkpointing, metrics) are wired correctly -- it is
NOT a substitute for the real Stage 2 exit criteria (AUC clearly above
chance on real FakeAVCeleb val data), which still requires real data.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

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


def _build_split_csv(tmp_path: Path, name: str, n_real: int, n_fake: int) -> Path:
    rows = []
    for i in range(n_real):
        path = tmp_path / f"{name}_real_{i}.mp4"
        if not _write_video(path, value=10):
            pytest.skip("mp4v codec unavailable in this environment")
        rows.append({"video_path": str(path), "label": "real", "identity_id": f"{name}_r{i}"})
    for i in range(n_fake):
        path = tmp_path / f"{name}_fake_{i}.mp4"
        if not _write_video(path, value=240):
            pytest.skip("mp4v codec unavailable in this environment")
        rows.append({"video_path": str(path), "label": "fake", "identity_id": f"{name}_f{i}"})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.slow
def test_visual_training_loop_learns_trivial_signal(tmp_path: Path):
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
                "frame_rate": 8,
                "frame_size": 48,
                "num_frames": 4,
                "batch_size": 4,
                "use_class_balanced_sampler": False,
                "num_workers": 0,
            },
            "model": {
                "backbone": "efficientnet_b0",
                "pretrained": False,
                "embed_dim": 32,
                "transformer": {"depth": 2, "heads": 4, "ff_dim": 64, "dropout": 0.0},
            },
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_epochs": 1,
                "max_epochs": 20,
                "grad_clip_norm": 1.0,
                "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
                "early_stopping_patience": 20,
            },
        }
    )

    metrics = train_visual(cfg)

    assert metrics["auc"] > 0.9, f"expected near-perfect AUC on separable data, got {metrics}"
    assert (tmp_path / "outputs" / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "outputs" / "results.md").exists()
