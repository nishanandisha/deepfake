"""Train-split-only audio augmentations. Callers must gate these on
split=="train" themselves -- these functions apply unconditionally when
called, val/test/calibration data must never be passed through them.

The visual augmentations of the parent multimodal project (JPEG
compression, flips, resized crops) are deliberately absent here: they
operated on frames, and this package has no video path, which is also why
it carries no OpenCV dependency.
"""

from typing import Optional

import numpy as np


def audio_additive_noise(
    signal: np.ndarray, snr_db: float = 20.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal_power = np.mean(signal**2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, noise_power**0.5, signal.shape)
    return (signal + noise).astype(np.float32)


def audio_codec_simulation(signal: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Crude lossy-codec stand-in: quantize to a lower bit depth and back,
    to roughly mimic compression artifacts without needing a real codec
    dependency."""
    levels = 2**bit_depth
    quantized = np.round((signal + 1.0) / 2.0 * (levels - 1))
    quantized = np.clip(quantized, 0, levels - 1)
    return (quantized / (levels - 1) * 2.0 - 1.0).astype(np.float32)
