import numpy as np

from src.preprocessing.augmentations import audio_additive_noise, audio_codec_simulation


def test_audio_additive_noise_preserves_length():
    signal = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = audio_additive_noise(signal, snr_db=15, rng=np.random.default_rng(3))
    assert out.shape == signal.shape
    assert not np.array_equal(out, signal)


def test_audio_additive_noise_respects_snr():
    """A higher SNR must perturb the signal less -- otherwise the parameter
    is decorative and train-time augmentation strength is unknowable."""
    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)
    quiet = audio_additive_noise(signal, snr_db=40, rng=np.random.default_rng(0))
    loud = audio_additive_noise(signal, snr_db=5, rng=np.random.default_rng(0))

    assert np.abs(quiet - signal).mean() < np.abs(loud - signal).mean()


def test_audio_codec_simulation_quantizes_within_range():
    signal = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = audio_codec_simulation(signal, bit_depth=4)
    assert out.shape == signal.shape
    assert out.min() >= -1.01 and out.max() <= 1.01


def test_audio_codec_simulation_reduces_distinct_levels():
    signal = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)
    out = audio_codec_simulation(signal, bit_depth=4)
    assert len(np.unique(out)) <= 16
