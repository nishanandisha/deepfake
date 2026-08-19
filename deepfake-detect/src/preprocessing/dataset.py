"""PyTorch Datasets wrapping a split manifest CSV for the visual and
acoustic branches. Both load + preprocess raw media via
src/preprocessing/{video,audio,acoustic features}, apply train-only
augmentation, and pad/truncate every clip to a fixed length so batches can
be stacked without a custom collate_fn.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.acoustic.features import extract_acoustic_features
from src.preprocessing.audio import load_audio
from src.preprocessing.augmentations import (
    apply_clip_augmentations,
    audio_additive_noise,
    audio_codec_simulation,
)
from src.preprocessing.cache import PreprocessingCache  # noqa: F401 (type reference)
from src.preprocessing.video import preprocess_video

LABEL_TO_INT = {"real": 0, "fake": 1}


class VisualDataset(Dataset):
    def __init__(
        self,
        manifest_csv_path: str,
        split: str,
        frame_rate: float = 25,
        frame_size: int = 224,
        num_frames: int = 32,
        augment: bool = None,
        seed: int = 42,
        cache: "PreprocessingCache" = None,
    ):
        self.df = pd.read_csv(manifest_csv_path)
        self.split = split
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.num_frames = num_frames
        # Augmentation must never touch val/test/calibration data (Stage 1
        # rule) -- defaults to on for "train", off otherwise, unless the
        # caller explicitly overrides.
        self.augment = (split == "train") if augment is None else augment
        self._rng = np.random.default_rng(seed)
        # Cached frames are the *un-augmented* aligned faces; augmentation
        # still runs per-epoch below so cached samples keep varying.
        self.cache = cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.cache is not None:
            frames = self.cache.get_visual(
                row["video_path"], self.frame_rate, self.frame_size
            )
        else:
            frames = preprocess_video(
                row["video_path"], frame_rate=self.frame_rate, size=self.frame_size
            )
        if frames.shape[0] == 0:
            frames = np.zeros((1, self.frame_size, self.frame_size, 3), dtype=np.uint8)

        if self.augment:
            frames = apply_clip_augmentations(frames, output_size=self.frame_size, rng=self._rng)

        frames, padding_mask = _pad_or_truncate(frames, self.num_frames)

        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # [T,3,H,W]
        mask_tensor = torch.from_numpy(padding_mask)
        label_tensor = torch.tensor(float(LABEL_TO_INT[row["label"]]))

        return frames_tensor, mask_tensor, label_tensor


class AcousticDataset(Dataset):
    def __init__(
        self,
        manifest_csv_path: str,
        split: str,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_mfcc: int = 20,
        num_frames: int = 300,
        augment: bool = None,
        seed: int = 42,
        cache: "PreprocessingCache" = None,
        feature_jitter_std: float = 0.05,
        pitch_tracker: str = "yin",
    ):
        self.df = pd.read_csv(manifest_csv_path)
        self.split = split
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.n_mfcc = n_mfcc
        self.num_frames = num_frames
        self.pitch_tracker = pitch_tracker
        self.augment = (split == "train") if augment is None else augment
        self._rng = np.random.default_rng(seed)
        self.feature_names = None  # populated on first __getitem__ call
        self.cache = cache
        self.feature_jitter_std = feature_jitter_std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.cache is not None:
            # Signal-space augmentation (noise + codec sim) would change the
            # input to librosa.pyin, defeating the cache -- and pyin is the
            # single most expensive step in the pipeline (~1-3s/clip). So
            # when caching, features are extracted once from the clean
            # signal and the train split instead gets feature-space jitter.
            # A weaker regulariser than true codec simulation; the tradeoff
            # is what makes multi-epoch training feasible at all. Disable
            # the cache to get exact signal-space augmentation back.
            features, feature_names = self.cache.get_acoustic(
                row["audio_path"], self.sample_rate, self.frame_ms, self.hop_ms, self.n_mfcc,
                pitch_tracker=self.pitch_tracker,
            )
            if self.augment and self.feature_jitter_std > 0:
                scale = np.abs(features).mean(axis=0, keepdims=True) + 1e-8
                noise = self._rng.normal(0, self.feature_jitter_std, features.shape) * scale
                features = (features + noise).astype(np.float32)
        else:
            signal = load_audio(row["audio_path"], sample_rate=self.sample_rate)

            if self.augment:
                signal = audio_additive_noise(signal, snr_db=20.0, rng=self._rng)
                signal = audio_codec_simulation(signal, bit_depth=8)

            features, feature_names = extract_acoustic_features(
                signal,
                sample_rate=self.sample_rate,
                frame_ms=self.frame_ms,
                hop_ms=self.hop_ms,
                n_mfcc=self.n_mfcc,
                pitch_tracker=self.pitch_tracker,
            )
        self.feature_names = feature_names

        features, padding_mask = _pad_or_truncate(features, self.num_frames)

        features_tensor = torch.from_numpy(features).float()  # [S, D]
        mask_tensor = torch.from_numpy(padding_mask)
        label_tensor = torch.tensor(float(LABEL_TO_INT[row["label"]]))

        return features_tensor, mask_tensor, label_tensor


class MultimodalDataset(Dataset):
    """Pairs a VisualDataset and an AcousticDataset over the same manifest
    rows, for Stage 5's joint fusion training which needs both modalities
    in a single batch. Composition, not reimplementation -- each modality's
    loading/augmentation logic stays defined once, in its own dataset.
    """

    def __init__(
        self,
        manifest_csv_path: str,
        split: str,
        visual_kwargs: dict = None,
        acoustic_kwargs: dict = None,
        seed: int = 42,
        cache: "PreprocessingCache" = None,
    ):
        self.visual = VisualDataset(
            manifest_csv_path, split=split, seed=seed, cache=cache, **(visual_kwargs or {})
        )
        self.acoustic = AcousticDataset(
            manifest_csv_path, split=split, seed=seed, cache=cache, **(acoustic_kwargs or {})
        )
        assert len(self.visual) == len(self.acoustic), (
            "visual and acoustic datasets must be built from the same manifest rows"
        )
        self.df = self.visual.df

    def __len__(self) -> int:
        return len(self.visual)

    def __getitem__(self, idx: int):
        frames, visual_mask, label = self.visual[idx]
        features, acoustic_mask, label_check = self.acoustic[idx]
        assert label.item() == label_check.item(), (
            f"label mismatch at row {idx}: visual={label.item()} acoustic={label_check.item()}"
        )
        return frames, visual_mask, features, acoustic_mask, label


def _pad_or_truncate(frames: np.ndarray, num_frames: int):
    """Returns (frames [num_frames,H,W,3], padding_mask [num_frames] bool,
    True at padded positions)."""
    t = frames.shape[0]
    if t >= num_frames:
        return frames[:num_frames], np.zeros(num_frames, dtype=bool)

    pad_shape = (num_frames - t, *frames.shape[1:])
    padded = np.concatenate([frames, np.zeros(pad_shape, dtype=frames.dtype)], axis=0)
    mask = np.zeros(num_frames, dtype=bool)
    mask[t:] = True
    return padded, mask
