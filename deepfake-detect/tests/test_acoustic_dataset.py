from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.preprocessing.dataset import AcousticDataset


def _write_tone(path: Path, freq: float, duration_s: float, sample_rate: int = 16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), signal, sample_rate)


def _make_manifest(tmp_path: Path) -> Path:
    real_path = tmp_path / "real.wav"
    fake_path = tmp_path / "fake.wav"
    _write_tone(real_path, freq=110, duration_s=0.3)
    _write_tone(fake_path, freq=110, duration_s=1.0)

    manifest = pd.DataFrame(
        [
            {"audio_path": str(real_path), "label": "real", "identity_id": "id0"},
            {"audio_path": str(fake_path), "label": "fake", "identity_id": "id1"},
        ]
    )
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path, index=False)
    return csv_path


def test_dataset_pads_short_clip(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = AcousticDataset(csv_path, split="val", n_mfcc=13, num_frames=200)

    features, mask, label = dataset[0]  # short "real" clip

    assert features.shape == (200, 47)
    assert mask.shape == (200,)
    assert mask.any()  # some padding present
    assert label.item() == 0.0
    assert dataset.feature_names is not None
    assert len(dataset.feature_names) == 47


def test_dataset_truncates_long_clip(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = AcousticDataset(csv_path, split="val", n_mfcc=13, num_frames=20)

    features, mask, label = dataset[1]  # long "fake" clip

    assert features.shape == (20, 47)
    assert not mask.any()
    assert label.item() == 1.0


def test_train_split_augments_by_default(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = AcousticDataset(csv_path, split="train", n_mfcc=13, num_frames=20)
    assert dataset.augment is True


def test_val_split_does_not_augment(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = AcousticDataset(csv_path, split="val", n_mfcc=13, num_frames=20)
    assert dataset.augment is False
