"""Waveform augmentation, applied *before* the frozen frontend.

A frozen-and-cached frontend cannot be augmented on the fly: by training time
the waveform is long gone. So each training clip is encoded more than once --
clean plus a couple of perturbed variants -- and the training loader treats
the variants as additional examples of the same label.

The perturbations target the channel, not the content. What we want the model
to stop relying on is "this was recorded in a quiet room through a good mic
and never re-encoded", because that is a property of the corpus (933 human
clips scraped from 14 YouTube videos) rather than a property of human speech.
Every transform here is something a real upload might have been through:
bandwidth loss from a codec, a coloured channel, background noise, clipping.

Deliberately cheap and dependency-free -- numpy and torchaudio only, no
ffmpeg round-trip per variant, because this runs 2x over the training set.
"""

import numpy as np
import torch
import torchaudio.functional as AF

SAMPLE_RATE = 16000


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)) + 1e-12)


def add_noise(signal: np.ndarray, rng: np.random.Generator, snr_db_range=(8, 30)) -> np.ndarray:
    """Additive noise, coloured at random so it is not always flat white."""
    snr_db = rng.uniform(*snr_db_range)
    noise = rng.standard_normal(len(signal)).astype(np.float32)

    # A one-pole filter tilts the spectrum toward low (positive a) or high
    # (negative a) frequencies, which covers hum/rumble and hiss with one knob.
    a = rng.uniform(-0.8, 0.8)
    noise = np.asarray(AF.lfilter(
        torch.from_numpy(noise).unsqueeze(0),
        torch.tensor([1.0, -a], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        clamp=False,
    ).squeeze(0), dtype=np.float32)

    scale = _rms(signal) / (_rms(noise) * (10 ** (snr_db / 20)))
    return signal + scale * noise


def band_limit(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample down and back up: the bandwidth loss a lossy codec imposes.

    Cheaper than an actual MP3/AAC round-trip and captures the part that
    matters most here -- the high-frequency detail simply is not there any
    more, so a detector cannot lean on it.
    """
    target = int(rng.choice([8000, 11025, 12000, 16000]))
    if target >= SAMPLE_RATE:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    down = AF.resample(tensor, SAMPLE_RATE, target)
    up = AF.resample(down, target, SAMPLE_RATE)
    return up.squeeze(0).numpy().astype(np.float32)


def colour_channel(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random gentle EQ, standing in for microphone and room response."""
    tensor = torch.from_numpy(signal).unsqueeze(0)
    for _ in range(int(rng.integers(1, 3))):
        centre = float(rng.uniform(200, 6000))
        gain = float(rng.uniform(-8, 8))
        q = float(rng.uniform(0.5, 2.0))
        tensor = AF.equalizer_biquad(tensor, SAMPLE_RATE, centre, gain, q)
    return tensor.squeeze(0).numpy().astype(np.float32)


def random_gain(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Level change, occasionally hard enough to clip -- as uploads do."""
    signal = signal * float(rng.uniform(0.3, 1.8))
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


# Applied in this order; each fires with its own probability so variants are
# not all maximally corrupted.
_PIPELINE = [
    (colour_channel, 0.7),
    (band_limit, 0.5),
    (add_noise, 0.8),
    (random_gain, 0.6),
]


def augment(signal: np.ndarray, seed: int) -> np.ndarray:
    """One perturbed copy of `signal`. Deterministic in `seed` so a rerun of
    the caching script reproduces the same variants rather than quietly
    changing the training set."""
    rng = np.random.default_rng(seed)
    out = signal.astype(np.float32)
    for transform, probability in _PIPELINE:
        if rng.random() < probability:
            out = transform(out, rng)

    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)
