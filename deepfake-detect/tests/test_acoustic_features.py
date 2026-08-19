import numpy as np

from src.models.acoustic.features import extract_acoustic_features


def _tone(freq: float, duration_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_count_matches_names():
    signal = _tone(220)
    features, names = extract_acoustic_features(signal, sample_rate=16000, n_mfcc=13)

    assert features.shape[1] == len(names)
    assert features.ndim == 2
    assert features.shape[0] > 1


def test_feature_names_ordering_and_uniqueness():
    _, names = extract_acoustic_features(_tone(220), n_mfcc=13)

    assert len(names) == len(set(names))
    assert names[0] == "mfcc_0"
    assert names[13] == "mfcc_delta_0"
    assert names[26] == "mfcc_delta2_0"
    assert names[39:] == [
        "f0",
        "voicing_confidence",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "zero_crossing_rate",
        "short_time_energy",
    ]


def test_no_nans_in_output():
    features, _ = extract_acoustic_features(_tone(440), n_mfcc=13)
    assert not np.isnan(features).any()


def test_short_signal_is_handled():
    tiny_signal = np.zeros(50, dtype=np.float32)
    features, names = extract_acoustic_features(tiny_signal, n_mfcc=13)
    assert features.shape[1] == len(names)
    assert features.shape[0] >= 1


def test_different_tones_give_different_spectral_centroid():
    low, names = extract_acoustic_features(_tone(110), n_mfcc=13)
    high, _ = extract_acoustic_features(_tone(4000), n_mfcc=13)

    centroid_idx = names.index("spectral_centroid")
    assert high[:, centroid_idx].mean() > low[:, centroid_idx].mean()
