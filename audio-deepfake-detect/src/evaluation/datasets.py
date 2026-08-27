"""Dataset construction for evaluation and calibration runs.

Kept apart from the training loop's own dataset setup for one reason:
evaluation must never see augmented input. AcousticDataset already defaults
augmentation off for any split other than "train", and nothing here turns it
back on.
"""

from pathlib import Path

from src.preprocessing.cache import get_shared_cache
from src.preprocessing.dataset import AcousticDataset


def build_split_dataset(cfg, split: str, n_mfcc: int) -> AcousticDataset:
    """Builds the AcousticDataset for one named split of `cfg.data.splits_dir`.

    `n_mfcc` comes from the loaded model artefact rather than the config, so
    a mismatch between the config on disk and the weights being evaluated
    can't silently produce a differently-shaped feature matrix.
    """
    return AcousticDataset(
        Path(cfg.data.splits_dir) / f"{split}.csv",
        split=split,
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=n_mfcc,
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
        num_frames=cfg.data.get("num_audio_frames", 400),
        seed=cfg.seed,
        cache=get_shared_cache(cfg.data.get("cache_dir"), cfg.data.get("use_cache", True)),
    )
