from pathlib import Path

import numpy as np
import soundfile as sf

from src.preprocessing.cache import PreprocessingCache


def _write_tone(path: Path, freq=220.0, duration_s=0.3, sample_rate=16000):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate)


def test_acoustic_cache_roundtrip_preserves_feature_names(tmp_path: Path):
    audio_path = tmp_path / "clip.wav"
    _write_tone(audio_path)

    cache = PreprocessingCache(str(tmp_path / "cache"))
    features_1, names_1 = cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13)
    features_2, names_2 = cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13)

    assert np.allclose(features_1, features_2)
    assert names_1 == names_2
    assert len(names_1) == features_1.shape[1]
    assert cache.stats()["acoustic_entries"] == 1


def test_cache_key_changes_with_params(tmp_path: Path):
    """Different preprocessing params must not collide -- otherwise changing
    n_mfcc in the config would silently serve stale tensors."""
    audio_path = tmp_path / "clip.wav"
    _write_tone(audio_path)

    cache = PreprocessingCache(str(tmp_path / "cache"))
    cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13)
    cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 20)  # different n_mfcc

    assert cache.stats()["acoustic_entries"] == 2


def test_cache_key_changes_with_pitch_tracker(tmp_path: Path):
    """yin and pyin produce different voicing_confidence columns, so they
    must not share a cache entry."""
    audio_path = tmp_path / "clip.wav"
    _write_tone(audio_path)

    cache = PreprocessingCache(str(tmp_path / "cache"))
    cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13, pitch_tracker="yin")
    cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13, pitch_tracker="pyin")

    assert cache.stats()["acoustic_entries"] == 2


def test_disabled_cache_writes_nothing(tmp_path: Path):
    audio_path = tmp_path / "clip.wav"
    _write_tone(audio_path)

    cache = PreprocessingCache(str(tmp_path / "cache"), enabled=False)
    features, names = cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13)

    assert features.shape[1] == len(names)
    assert cache.stats() == {"enabled": False}


def test_corrupt_cache_entry_is_regenerated(tmp_path: Path):
    """An interrupted run can leave a truncated .npz; it must be rebuilt
    rather than crashing training hours in."""
    audio_path = tmp_path / "clip.wav"
    _write_tone(audio_path)

    cache = PreprocessingCache(str(tmp_path / "cache"))
    expected, _ = cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13)

    # The params dict must match get_acoustic's exactly -- including
    # pitch_tracker -- or this writes garbage to an unrelated key and the
    # test passes without ever exercising the recovery path.
    params = {"kind": "acoustic", "sample_rate": 16000, "frame_ms": 25.0,
              "hop_ms": 10.0, "n_mfcc": 13, "pitch_tracker": "yin"}
    corrupt_path = cache.acoustic_path(str(audio_path), params)
    assert corrupt_path.exists(), "params dict has drifted from get_acoustic's cache key"
    corrupt_path.write_bytes(b"corrupt not-an-npz")

    recovered, names = cache.get_acoustic(str(audio_path), 16000, 25.0, 10.0, 13)
    assert np.allclose(expected, recovered)
    assert len(names) == recovered.shape[1]
