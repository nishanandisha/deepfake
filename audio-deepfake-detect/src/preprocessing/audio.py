"""Audio loading + framing for the acoustic branch: resample to 16kHz mono,
frame at 25ms window / 10ms hop.

Datasets like LAV-DF and DFDC mux audio inside .mp4 containers, which
soundfile cannot open -- librosa then falls back to audioread, which needs
an ffmpeg binary on PATH. Rather than require a system-wide ffmpeg install,
`_ensure_ffmpeg_on_path` points at the one bundled with imageio-ffmpeg.
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

_FFMPEG_READY = False


def _ensure_ffmpeg_on_path() -> None:
    """Puts a usable ffmpeg on PATH exactly once per process.

    Without this, decoding a .mp4 raises audioread.NoBackendError -- which,
    because warm_cache deliberately tolerates per-clip failures, previously
    showed up as a silently empty acoustic cache rather than an error.
    """
    global _FFMPEG_READY
    if _FFMPEG_READY:
        return

    try:
        import imageio_ffmpeg

        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass  # a system ffmpeg may still be present

    _FFMPEG_READY = True


def _decode_with_ffmpeg(path: str, sample_rate: int) -> np.ndarray:
    """Decodes audio to mono float32 by piping raw PCM out of ffmpeg.

    audioread's ffmpeg backend is fragile about the ffmpeg build it finds,
    so for containers this calls the binary directly -- one subprocess, no
    temp files, and it works with the bundled imageio-ffmpeg build.
    """
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio (or audio-embedded video) file as mono float32 at
    `sample_rate` Hz."""
    _ensure_ffmpeg_on_path()
    try:
        signal, _ = librosa.load(path, sr=sample_rate, mono=True)
        return signal.astype(np.float32)
    except Exception:
        # Containers (.mp4/.avi/.mov) routinely fail librosa's soundfile and
        # audioread paths; fall back to decoding via ffmpeg directly.
        return _decode_with_ffmpeg(path, sample_rate).astype(np.float32)


def frame_audio(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> Tuple[np.ndarray, int, int]:
    """Split a mono signal into overlapping frames.

    Returns (frames [num_frames, frame_length], frame_length, hop_length).
    """
    frame_length = int(round(sample_rate * frame_ms / 1000))
    hop_length = int(round(sample_rate * hop_ms / 1000))

    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))

    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)
    return frames, frame_length, hop_length


def preprocess_audio(
    path: str,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> Tuple[np.ndarray, int, int]:
    signal = load_audio(path, sample_rate=sample_rate)
    return frame_audio(signal, sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
