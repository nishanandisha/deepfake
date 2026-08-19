"""End-to-end sanity check for Stage 6: trains a tiny fusion model from
scratch on a synthetic three-way-split (train/val/calibration) dataset,
then runs calibration + threshold selection against it and checks the
artifacts (policy.json, results.md, reliability_diagram.png) come out
sane. Mechanics-only, like the other stages' overfit tests.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from src.evaluation.calibrate_and_select_policy import run_calibration_and_policy
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


def _model_and_data_cfg(splits_dir: Path):
    return {
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
    }


@pytest.mark.slow
def test_calibration_and_policy_end_to_end(tmp_path: Path):
    # Seed explicitly: torch/numpy global RNG state otherwise depends on
    # whatever tests ran before this one, making weight init (and therefore
    # the learned thresholds asserted below) order-dependent and flaky.
    set_seed(0, deterministic=False)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _build_split_csv(splits_dir, "train", n_real=6, n_fake=6)
    _build_split_csv(splits_dir, "val", n_real=4, n_fake=4)
    _build_split_csv(splits_dir, "calibration", n_real=4, n_fake=4)

    base_cfg = _model_and_data_cfg(splits_dir)

    fusion_cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs" / "fusion"),
            "init_from_standalone_checkpoints": False,
            "late_fusion_results": str(tmp_path / "outputs" / "late_fusion" / "results.json"),
            **base_cfg,
            "training": {
                # AUC saturates at 1.0 almost immediately on this trivially
                # separable data, but the logits stay low-magnitude (hence
                # weakly separated in absolute score space) until BCE loss
                # keeps being minimized for a while longer. A short patience
                # cuts training off right at that plateau, before the
                # calibrated scores are cleanly separated -- give it more
                # room so the threshold search below has a meaningful signal.
                "lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 1, "max_epochs": 15,
                "grad_clip_norm": 1.0,
                "checkpoint_dir": str(tmp_path / "outputs" / "fusion" / "checkpoints"),
                "early_stopping_patience": 15,
            },
        }
    )
    train_fusion(fusion_cfg)

    calibration_cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "auto",
            "output_dir": str(tmp_path / "outputs" / "calibration"),
            "fusion_checkpoint": str(tmp_path / "outputs" / "fusion" / "checkpoints" / "best.pt"),
            "false_suppression_ceiling": 0.02,
            "threshold_grid_size": 50,
            "tau_hi": None,
            "tau_lo": None,
            **base_cfg,
        }
    )

    artifact = run_calibration_and_policy(calibration_cfg)

    assert "temperature" in artifact
    assert 0.0 <= artifact["tau_lo"] <= artifact["tau_hi"] <= 1.0
    assert artifact["false_suppression_rate"] <= 0.02
    assert artifact["detection_recall"] > 0.5, (
        f"expected the policy to actually catch fakes on this well-separated "
        f"synthetic data, got {artifact}"
    )

    output_dir = tmp_path / "outputs" / "calibration"
    assert (output_dir / "policy.json").exists()
    assert (output_dir / "results.md").exists()
    assert (output_dir / "reliability_diagram.png").exists()
