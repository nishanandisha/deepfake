"""Torch Dataset over cached WavLM embeddings.

Reads float16 `.npy` files written by `scripts/cache_embeddings.py`; no audio
decoding and no frontend pass happens here, which is what keeps an epoch to
seconds rather than minutes.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.embeddings import EmbeddingCache


class EmbeddingDataset(Dataset):
    """One row per (clip, variant). Returns [T, 768] float32 and a label.

    Augmented variants are cached alongside the clean encoding and enumerated
    here as extra rows, so a clip with two variants contributes three
    training examples that share a label and a group.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache: EmbeddingCache,
        variants: List[str] = None,
        max_frames: int = 400,
        random_crop: bool = False,
        spec_augment: bool = False,
        seed: int = 42,
    ):
        self.cache = cache
        self.max_frames = max_frames
        self.random_crop = random_crop
        self.spec_augment = spec_augment
        self.rng = np.random.default_rng(seed)

        variants = variants or ["clean"]
        rows = []
        for _, row in manifest.iterrows():
            for variant in variants:
                if cache.path_for(row["path"], variant).exists():
                    rows.append((row["path"], variant, int(row["label"]), row["source"]))

        if not rows:
            raise RuntimeError(
                "no cached embeddings found for this split -- "
                "run scripts/cache_embeddings.py first"
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, features: np.ndarray) -> np.ndarray:
        total = features.shape[0]
        if total <= self.max_frames:
            return features
        if self.random_crop:
            start = int(self.rng.integers(0, total - self.max_frames + 1))
        else:
            # Centre crop for eval: deterministic, and the middle of a clip is
            # likelier to be speech than either edge.
            start = (total - self.max_frames) // 2
        return features[start : start + self.max_frames]

    def _spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Time and channel masking on the embeddings.

        The waveform is unavailable by this point, so this supplies the
        variation that waveform augmentation cannot: it stops the head from
        keying on any single embedding channel or any one instant.
        """
        features = features.copy()
        frames, dims = features.shape

        for _ in range(2):
            width = int(self.rng.integers(0, max(2, frames // 10)))
            if width > 0 and frames > width:
                start = int(self.rng.integers(0, frames - width))
                features[start : start + width, :] = 0.0

        for _ in range(2):
            width = int(self.rng.integers(0, max(2, dims // 16)))
            if width > 0 and dims > width:
                start = int(self.rng.integers(0, dims - width))
                features[:, start : start + width] = 0.0

        return features

    def __getitem__(self, index: int):
        path, variant, label, _ = self.rows[index]
        features = self.cache.get(path, variant)
        if features is None:
            raise RuntimeError(f"cache entry vanished for {path} ({variant})")

        features = self._crop(features)
        if self.spec_augment:
            features = self._spec_augment(features)

        return torch.from_numpy(features), torch.tensor(float(label))

    def labels(self) -> np.ndarray:
        return np.array([row[2] for row in self.rows])

    def sources(self) -> List[str]:
        return [row[3] for row in self.rows]


def collate_padded(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to the longest clip in the batch and return a padding mask.

    Clips run 2.5-13s, so a fixed-length tensor would either truncate the long
    ones or pad the short ones by 5x. The mask is what stops attentive pooling
    from averaging over padding.
    """
    features, labels = zip(*batch)
    lengths = [f.shape[0] for f in features]
    longest = max(lengths)
    dim = features[0].shape[1]

    padded = torch.zeros(len(features), longest, dim)
    mask = torch.ones(len(features), longest, dtype=torch.bool)  # True = padding
    for i, (feature, length) in enumerate(zip(features, lengths)):
        padded[i, :length] = feature
        mask[i, :length] = False

    return padded, mask, torch.stack(labels)
