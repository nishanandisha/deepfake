"""Train-split-only augmentations. Callers must gate these on split=="train"
themselves -- these functions apply unconditionally when called, val/test/
calibration data must never be passed through them.
"""

from typing import Optional

import cv2
import numpy as np


def jpeg_compression(frame_rgb: np.ndarray, quality: int = 50) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return frame_rgb
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def gaussian_noise(
    frame_rgb: np.ndarray, std: float = 5.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, std, frame_rgb.shape)
    return np.clip(frame_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def horizontal_flip(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[:, ::-1, :])


def random_resized_crop(
    frame_rgb: np.ndarray,
    output_size: int,
    scale: tuple = (0.8, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    height, width = frame_rgb.shape[:2]

    area_fraction = rng.uniform(*scale)
    side = int(round((height * width * area_fraction) ** 0.5))
    side = min(side, height, width)

    y0 = rng.integers(0, height - side + 1)
    x0 = rng.integers(0, width - side + 1)
    crop = frame_rgb[y0 : y0 + side, x0 : x0 + side]

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def audio_additive_noise(
    signal: np.ndarray, snr_db: float = 20.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal_power = np.mean(signal**2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, noise_power**0.5, signal.shape)
    return (signal + noise).astype(np.float32)


def apply_clip_augmentations(
    frames: np.ndarray, output_size: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Apply flip/crop/compression/noise to a whole [T, H, W, 3] clip with
    shared random parameters across frames, so temporal consistency (the
    same crop window, the same flip decision) is preserved instead of each
    frame being independently perturbed."""
    rng = rng or np.random.default_rng()
    out = frames

    if rng.random() < 0.5:
        out = np.stack([horizontal_flip(f) for f in out])

    height, width = out.shape[1:3]
    area_fraction = rng.uniform(0.8, 1.0)
    side = min(int(round((height * width * area_fraction) ** 0.5)), height, width)
    y0 = int(rng.integers(0, height - side + 1))
    x0 = int(rng.integers(0, width - side + 1))
    out = np.stack(
        [
            cv2.resize(f[y0 : y0 + side, x0 : x0 + side], (output_size, output_size),
                       interpolation=cv2.INTER_AREA)
            for f in out
        ]
    )

    quality = int(rng.integers(30, 70))
    out = np.stack([jpeg_compression(f, quality=quality) for f in out])

    noise_std = rng.uniform(2, 8)
    out = np.stack([gaussian_noise(f, std=noise_std, rng=rng) for f in out])

    return out


def audio_codec_simulation(signal: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Crude lossy-codec stand-in: quantize to a lower bit depth and back,
    to roughly mimic compression artifacts without needing a real codec
    dependency."""
    levels = 2**bit_depth
    quantized = np.round((signal + 1.0) / 2.0 * (levels - 1))
    quantized = np.clip(quantized, 0, levels - 1)
    return (quantized / (levels - 1) * 2.0 - 1.0).astype(np.float32)
