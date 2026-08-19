from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.dataset import VisualDataset


def _write_synthetic_video(
    path: Path, num_frames: int, fps: int = 10, size=(64, 64), value=20
) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        return False
    for _ in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return True


def _make_manifest(tmp_path: Path) -> Path:
    real_path = tmp_path / "real.mp4"
    fake_path = tmp_path / "fake.mp4"
    if not _write_synthetic_video(real_path, num_frames=3, value=10) or not _write_synthetic_video(
        fake_path, num_frames=8, value=240
    ):
        pytest.skip("mp4v codec unavailable in this environment")

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "real1",
                "video_path": str(real_path),
                "audio_path": str(real_path),
                "identity_id": "id0",
                "label": "real",
                "manipulated_modality": "none",
                "source_generator": None,
            },
            {
                "sample_id": "fake1",
                "video_path": str(fake_path),
                "audio_path": str(fake_path),
                "identity_id": "id1",
                "label": "fake",
                "manipulated_modality": "both",
                "source_generator": "fsgan",
            },
        ]
    )
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path, index=False)
    return csv_path


def test_dataset_pads_short_clip_and_masks_correctly(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = VisualDataset(csv_path, split="val", frame_rate=10, frame_size=32, num_frames=6)

    frames, mask, label = dataset[0]  # the 3-frame "real" clip

    assert frames.shape == (6, 3, 32, 32)
    assert mask.shape == (6,)
    assert mask[:3].sum() == 0  # first frames are real, not padded
    assert mask[3:].all()  # remaining are padding
    assert label.item() == 0.0


def test_dataset_truncates_long_clip(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = VisualDataset(csv_path, split="val", frame_rate=10, frame_size=32, num_frames=4)

    frames, mask, label = dataset[1]  # the 8-frame "fake" clip

    assert frames.shape == (4, 3, 32, 32)
    assert not mask.any()
    assert label.item() == 1.0


def test_dataset_val_split_does_not_augment(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = VisualDataset(csv_path, split="val", frame_rate=10, frame_size=32, num_frames=4)
    assert dataset.augment is False


def test_dataset_train_split_augments_by_default(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = VisualDataset(csv_path, split="train", frame_rate=10, frame_size=32, num_frames=4)
    assert dataset.augment is True
