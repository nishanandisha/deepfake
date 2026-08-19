from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.preprocessing.dataset import MultimodalDataset


def _write_video(path: Path, value: int, num_frames=4, size=(48, 48)) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    if not writer.isOpened():
        return False
    for _ in range(num_frames):
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()
    return True


def _write_tone(path: Path, freq: float, duration_s=0.3, sample_rate=16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate)


def _make_manifest(tmp_path: Path) -> Path:
    video_path = tmp_path / "clip.mp4"
    audio_path = tmp_path / "clip.wav"
    if not _write_video(video_path, value=100):
        pytest.skip("mp4v codec unavailable in this environment")
    _write_tone(audio_path, freq=220)

    manifest = pd.DataFrame(
        [{"video_path": str(video_path), "audio_path": str(audio_path), "label": "fake",
          "identity_id": "id0"}]
    )
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path, index=False)
    return csv_path


def test_multimodal_dataset_returns_paired_tensors(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = MultimodalDataset(
        csv_path,
        split="val",
        visual_kwargs={"frame_rate": 8, "frame_size": 32, "num_frames": 4},
        acoustic_kwargs={"n_mfcc": 13, "num_frames": 20},
    )

    frames, v_mask, features, a_mask, label = dataset[0]

    assert frames.shape == (4, 3, 32, 32)
    assert v_mask.shape == (4,)
    assert features.shape == (20, 47)
    assert a_mask.shape == (20,)
    assert label.item() == 1.0


def test_multimodal_dataset_length_matches_manifest(tmp_path: Path):
    csv_path = _make_manifest(tmp_path)
    dataset = MultimodalDataset(
        csv_path, split="val",
        visual_kwargs={"frame_rate": 8, "frame_size": 32, "num_frames": 4},
        acoustic_kwargs={"n_mfcc": 13, "num_frames": 20},
    )
    assert len(dataset) == 1
