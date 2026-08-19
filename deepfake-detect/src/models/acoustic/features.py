"""Named acoustic feature extraction: MFCC (+ delta, delta-delta), F0 +
voicing confidence, spectral stats, ZCR/energy -- per frame, 25ms window /
10ms hop by default.

Deliberately hand-crafted, human-readable features (not a learned
embedding): `FEATURE_NAMES` gives one name per output dimension, in the
exact order features are concatenated, so Stage 7's SHAP explanations can
attribute the acoustic branch's decision back to specific descriptors (e.g.
"F0 variance", "spectral flatness").
"""

from typing import List, Tuple

import librosa
import numpy as np


def _feature_names(n_mfcc: int) -> List[str]:
    names = [f"mfcc_{i}" for i in range(n_mfcc)]
    names += [f"mfcc_delta_{i}" for i in range(n_mfcc)]
    names += [f"mfcc_delta2_{i}" for i in range(n_mfcc)]
    names += [
        "f0",
        "voicing_confidence",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "zero_crossing_rate",
        "short_time_energy",
    ]
    return names


def _track_pitch(
    signal: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    pitch_tracker: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (f0, voicing_confidence).

    "pyin" runs a full Viterbi decode and yields a true probabilistic
    voicing estimate, but costs ~17s for an 8.5s clip -- ~9.5 hours for a
    2,000-clip dataset, single-core. "yin" computes the same pitch track in
    ~0.1s (177x faster) but returns no voicing probability, so we derive a
    proxy from the two things that actually make a frame voiced: the pitch
    estimate sitting inside the plausible vocal range, and the frame
    carrying enough energy to be speech rather than silence.

    The proxy is coarser than pyin's posterior, and that is a real
    limitation to report -- but it is what makes multi-epoch training on
    full-length clips feasible on modest hardware.
    """
    if pitch_tracker == "pyin":
        f0, _voiced_flag, voiced_prob = librosa.pyin(
            signal, fmin=fmin, fmax=fmax, sr=sample_rate, hop_length=hop_length
        )
        return np.nan_to_num(f0, nan=0.0), np.nan_to_num(voiced_prob, nan=0.0)

    if pitch_tracker != "yin":
        raise ValueError(f"Unknown pitch_tracker {pitch_tracker!r}; expected 'pyin' or 'yin'")

    f0 = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sample_rate, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)

    rms = librosa.feature.rms(y=signal, hop_length=hop_length).reshape(-1)
    rms = rms[: len(f0)] if len(rms) >= len(f0) else np.pad(rms, (0, len(f0) - len(rms)))
    energy_score = rms / (rms.max() + 1e-8)

    # yin always emits *some* f0; frames pinned at the search bounds are
    # where it failed to find real periodicity.
    in_range = ((f0 > fmin * 1.05) & (f0 < fmax * 0.95)).astype(np.float32)
    voicing_confidence = (in_range * energy_score).astype(np.float32)

    return f0, voicing_confidence


def extract_acoustic_features(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mfcc: int = 20,
    fmin: float = 65.0,
    fmax: float = 2093.0,
    pitch_tracker: str = "yin",
) -> Tuple[np.ndarray, List[str]]:
    """Returns (features [num_frames, num_dims], feature_names), with
    feature_names[i] naming column i of features -- verify this ordering
    matches the tensor columns before wiring up SHAP in Stage 7.

    pitch_tracker: "yin" (default, fast) or "pyin" (slower, true voicing
    posterior). Both produce the same feature layout, so SHAP names stay
    aligned either way.
    """
    frame_length = int(round(sample_rate * frame_ms / 1000))
    hop_length = int(round(sample_rate * hop_ms / 1000))

    # librosa's minimum usable n_fft/frame_length is a handful of samples;
    # very short test clips are padded up to that floor so every feature
    # function below produces at least one frame.
    min_len = max(frame_length * 4, 512)
    if len(signal) < min_len:
        signal = np.pad(signal, (0, min_len - len(signal)))

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    f0, voiced_prob = _track_pitch(
        signal, sample_rate, hop_length, fmin, fmax, pitch_tracker
    )

    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )
    flatness = librosa.feature.spectral_flatness(
        y=signal, n_fft=frame_length, hop_length=hop_length
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=signal, frame_length=frame_length, hop_length=hop_length
    )
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)

    num_frames = min(
        mfcc.shape[1], len(f0), centroid.shape[1], bandwidth.shape[1],
        rolloff.shape[1], flatness.shape[1], zcr.shape[1], energy.shape[1],
    )

    features = np.concatenate(
        [
            mfcc[:, :num_frames].T,
            delta[:, :num_frames].T,
            delta2[:, :num_frames].T,
            f0[:num_frames, None],
            voiced_prob[:num_frames, None],
            centroid[:, :num_frames].T,
            bandwidth[:, :num_frames].T,
            rolloff[:, :num_frames].T,
            flatness[:, :num_frames].T,
            zcr[:, :num_frames].T,
            energy[:, :num_frames].T,
        ],
        axis=1,
    ).astype(np.float32)

    feature_names = _feature_names(n_mfcc)
    assert features.shape[1] == len(feature_names), (
        f"feature/name count mismatch: {features.shape[1]} columns vs "
        f"{len(feature_names)} names -- extraction order must match _feature_names()"
    )

    return features, feature_names
