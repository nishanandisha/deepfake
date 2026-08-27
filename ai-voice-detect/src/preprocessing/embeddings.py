"""Frozen WavLM frontend plus an on-disk embedding cache.

Why frozen: fine-tuning a 94M-parameter frontend on an M1 would dominate every
run, and the training set here is 1,866 clips -- far too few to move that many
weights without overfitting. Running WavLM *once* per clip and caching the
result turns each training run into a few-minute job over a ~2M parameter
head, which is what makes iteration on this machine practical at all.

Why WavLM rather than MFCCs: mel-binning followed by DCT truncation discards
phase and fine spectral structure, which is exactly where vocoder and TTS
artifacts live. Measured on the predecessor project, an MFCC model scored ten
seconds of digital silence at 0.98 "fake" -- it had learned recording
conditions, because after MFCC that is most of what survives.

`torchaudio.pipelines.WAVLM_BASE_PLUS` is used in preference to a
HuggingFace checkpoint so the project depends only on torchaudio, which is
already required for audio I/O. There is no `transformers` dependency.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16000
EMBED_DIM = 768  # WavLM Base+ encoder width
NUM_LAYERS = 12  # transformer layers; extract_features can return 1..12


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _decode_with_ffmpeg(path: str, sample_rate: int) -> np.ndarray:
    """Last-resort decode through the ffmpeg binary.

    Needed for video containers (.mp4/.mov/.webm): soundfile rejects them
    outright, and librosa 1.0 dropped the audioread fallback that used to
    cover this, so neither can reach the audio track of a video. Since the
    model only ever sees the audio, a video upload is just a container to
    strip -- but without this it fails at the front door.
    """
    import shutil
    import subprocess

    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError(
            f"cannot decode {Path(path).name}: unsupported by soundfile and "
            "ffmpeg is not on PATH (brew install ffmpeg)"
        )

    result = subprocess.run(
        [binary, "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
         "-ar", str(sample_rate), "-"],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"no decodable audio track in {Path(path).name}"
            + (f": {message.splitlines()[-1]}" if message else "")
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 at `sample_rate`.

    Three tiers, cheapest first: soundfile handles the corpus (16 kHz mono
    FLAC) directly; librosa covers other bare audio formats and resampling;
    ffmpeg covers video containers, which neither of the first two can open.
    """
    try:
        signal, sr = sf.read(path, dtype="float32", always_2d=False)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        if sr == sample_rate:
            return signal.astype(np.float32)
    except Exception:  # noqa: BLE001 - fall through to the general decoder
        pass

    try:
        import librosa

        signal, _ = librosa.load(path, sr=sample_rate, mono=True)
        return signal.astype(np.float32)
    except Exception:  # noqa: BLE001 - containers land here
        return _decode_with_ffmpeg(path, sample_rate).astype(np.float32)


class WavLMFrontend:
    """Frozen WavLM Base+ returning per-frame hidden states at ~50 Hz."""

    def __init__(self, device: torch.device = None, layer: int = 6):
        self.device = device or pick_device()
        self.layer = layer
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def hidden_states(self, signal: np.ndarray, num_layers: int = None) -> List[torch.Tensor]:
        """Returns one [T, 768] tensor per transformer layer, on CPU."""
        waveform = torch.from_numpy(signal).float().unsqueeze(0).to(self.device)
        states, _ = self.model.extract_features(waveform, num_layers=num_layers)
        return [state.squeeze(0).cpu() for state in states]

    def embed(self, signal: np.ndarray, layer: int = None) -> np.ndarray:
        """Per-frame embedding [T, 768] float32 from a single layer.

        Layer choice matters: anti-spoofing cues concentrate in the middle of
        the stack, while the final layers drift toward phonetic/semantic
        content that is by design invariant to *how* the audio was produced.
        `scripts/sweep_layers.py` measures this rather than assuming it.
        """
        layer = self.layer if layer is None else layer
        states = self.hidden_states(signal, num_layers=layer)
        return states[layer - 1].numpy().astype(np.float32)


class EmbeddingCache:
    """Content-addressed store of per-clip embeddings, float16 on disk.

    float16 halves the footprint (~1.2 GB for this corpus at one layer) and
    costs nothing measurable in accuracy: these are activations feeding a
    LayerNorm, not weights being accumulated into.
    """

    def __init__(self, cache_dir: str, layer: int):
        self.cache_dir = Path(cache_dir) / f"wavlm_base_plus_L{layer}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.layer = layer

    def key(self, audio_path: str, variant: str) -> str:
        payload = json.dumps(
            {
                "source": str(audio_path),
                "model": "wavlm_base_plus",
                "layer": self.layer,
                "variant": variant,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def path_for(self, audio_path: str, variant: str = "clean") -> Path:
        return self.cache_dir / f"{self.key(audio_path, variant)}.npy"

    def get(self, audio_path: str, variant: str = "clean") -> Optional[np.ndarray]:
        path = self.path_for(audio_path, variant)
        if not path.exists():
            return None
        try:
            return np.load(path).astype(np.float32)
        except (OSError, ValueError):
            # A truncated file from an interrupted run should be regenerated,
            # not crash the job that finds it.
            path.unlink(missing_ok=True)
            return None

    def put(self, audio_path: str, features: np.ndarray, variant: str = "clean") -> None:
        path = self.path_for(audio_path, variant)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, features.astype(np.float16))
        tmp.replace(path)  # atomic, so a killed run leaves no partial file
