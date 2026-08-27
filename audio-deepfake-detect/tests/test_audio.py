from pathlib import Path

import numpy as np
import soundfile as sf

from src.preprocessing.audio import frame_audio, load_audio, preprocess_audio


def _write_sine_wave(path: Path, sample_rate: int = 22050, duration_s: float = 0.5):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), signal, sample_rate)


def test_load_audio_resamples_to_target_rate(tmp_path: Path):
    path = tmp_path / "tone.wav"
    _write_sine_wave(path, sample_rate=22050)

    signal = load_audio(str(path), sample_rate=16000)
    assert signal.dtype == np.float32
    assert abs(len(signal) / 16000 - 0.5) < 0.01


def test_frame_audio_shapes():
    sample_rate = 16000
    signal = np.zeros(sample_rate)  # 1 second of silence

    frames, frame_length, hop_length = frame_audio(
        signal, sample_rate=sample_rate, frame_ms=25, hop_ms=10
    )

    assert frame_length == 400  # 25ms @ 16kHz
    assert hop_length == 160  # 10ms @ 16kHz
    assert frames.shape[1] == frame_length  # (num_frames, frame_length)
    assert frames.shape[0] > 1


def test_frame_audio_pads_short_signal():
    frames, frame_length, _ = frame_audio(np.zeros(10), sample_rate=16000, frame_ms=25, hop_ms=10)
    assert frames.shape[1] == frame_length


def test_preprocess_audio_end_to_end(tmp_path: Path):
    path = tmp_path / "tone.wav"
    _write_sine_wave(path)

    frames, frame_length, hop_length = preprocess_audio(str(path))
    assert frames.shape[1] == frame_length
