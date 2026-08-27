"""Single-clip audio inference: the one function a caller needs.

preprocess -> named features -> acoustic Transformer -> (optional
calibration) -> decision -> SHAP explanation over the named descriptors.

Deliberately the only entry point a serving layer should touch, so nothing
above it reaches into training internals.

Calibration is OPTIONAL and off by default, because the exported acoustic
artefact carries no fitted policy of its own -- the parent multimodal
project fitted temperature and thresholds on the *fused* logit, and those
numbers do not transfer to this model. Run scripts/calibrate.py against a
calibration split to produce an audio-only policy.json, then pass it in.
Without one, `cScore` is the raw 1 - sigmoid(logit) and `calibrated` is
False; treat the decision as a plain 0.5 cut, not an operating point with a
measured false-suppression rate.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.evaluation.calibration import apply_temperature
from src.evaluation.policy import decide
from src.explain.shap_acoustic import explain_acoustic_features, top_k_features
from src.inference.loader import DEFAULT_MODEL_PATH, audio_settings_from, load_acoustic_model
from src.models.acoustic.features import extract_acoustic_features
from src.preprocessing.audio import load_audio
from src.preprocessing.dataset import _pad_or_truncate


@dataclass
class InferenceConfig:
    """Everything the pipeline needs that isn't the clip itself.

    The framing fields default to None, meaning "take it from the artefact's
    own data_config" -- which is almost always what you want, since features
    framed differently from training silently degrade the score.
    """

    model_path: Optional[str] = None
    policy_json: Optional[str] = None
    sample_rate: Optional[int] = None
    frame_ms: Optional[float] = None
    hop_ms: Optional[float] = None
    n_mfcc: Optional[int] = None
    num_audio_frames: Optional[int] = None
    pitch_tracker: Optional[str] = None
    top_k_features: int = 8
    shap_samples: int = 64
    explain: bool = True
    device: str = "auto"


def _waveform(signal: np.ndarray, sample_rate: int, peaks: int = 400) -> dict:
    """Downsampled peaks for a waveform display, plus the loudest regions.

    Note this is an energy heuristic for display only -- it is NOT the
    model's localization of the manipulation, which this architecture does
    not produce (clip-level classification only). Labelled as such so a UI
    never implies more than was measured.
    """
    duration = len(signal) / sample_rate
    if len(signal) == 0:
        return {"peaks": [], "durationSeconds": 0.0, "loudRegions": []}

    window = max(len(signal) // peaks, 1)
    trimmed = signal[: window * peaks]
    envelope = np.abs(trimmed.reshape(-1, window)).max(axis=1)
    envelope = envelope / (envelope.max() + 1e-9)

    threshold = float(np.percentile(envelope, 90))
    regions, start = [], None
    for i, value in enumerate(envelope):
        if value >= threshold and start is None:
            start = i
        elif value < threshold and start is not None:
            if i - start >= 2:
                regions.append({
                    "startSeconds": round(start / len(envelope) * duration, 3),
                    "endSeconds": round(i / len(envelope) * duration, 3),
                    "intensity": round(float(envelope[start:i].mean()), 3),
                })
            start = None

    return {
        "peaks": [round(float(v), 4) for v in envelope],
        "durationSeconds": round(duration, 3),
        "loudRegions": regions[:5],
    }


class AudioDeepfakePipeline:
    """Loads the model once, then serves many clips.

    Model loading dominates per-clip cost, so a serving layer should
    construct this once at startup rather than per request. `infer` is
    serialised by a lock so a threaded server can share one instance.
    """

    def __init__(self, cfg: InferenceConfig = None):
        self.cfg = cfg or InferenceConfig()
        self._lock = threading.Lock()
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.cfg.device == "auto"
            else self.cfg.device
        )

        loaded = load_acoustic_model(self.cfg.model_path or DEFAULT_MODEL_PATH, self.device)
        self.loaded = loaded
        self.model = loaded.model

        # Framing: explicit config wins, else whatever the model was trained
        # with, else the module defaults.
        settings = audio_settings_from(loaded.data_config)
        self.sample_rate = self.cfg.sample_rate or settings["sample_rate"]
        self.frame_ms = self.cfg.frame_ms or settings["frame_ms"]
        self.hop_ms = self.cfg.hop_ms or settings["hop_ms"]
        self.num_audio_frames = self.cfg.num_audio_frames or settings["num_audio_frames"]
        self.pitch_tracker = self.cfg.pitch_tracker or settings["pitch_tracker"]
        self.n_mfcc = self.cfg.n_mfcc or loaded.n_mfcc

        self.temperature, self.tau_lo, self.tau_hi = 1.0, 0.5, 0.5
        self.calibrated = False
        self.false_suppression_rate = None
        self.review_queue_rate = None
        if self.cfg.policy_json:
            self._load_policy(self.cfg.policy_json)

        _, self.feature_names = extract_acoustic_features(
            np.zeros(self.sample_rate), n_mfcc=self.n_mfcc, pitch_tracker=self.pitch_tracker
        )
        assert len(self.feature_names) == loaded.input_dim, (
            f"the loaded model expects {loaded.input_dim} feature columns but the "
            f"extractor produces {len(self.feature_names)} at n_mfcc={self.n_mfcc}"
        )
        self._background: Optional[torch.Tensor] = None

    def _load_policy(self, policy_json: str) -> None:
        import json

        policy = json.loads(Path(policy_json).read_text())
        self.temperature = float(policy["temperature"])
        self.tau_lo, self.tau_hi = float(policy["tau_lo"]), float(policy["tau_hi"])
        self.false_suppression_rate = policy.get("false_suppression_rate")
        self.review_queue_rate = policy.get("review_queue_rate")
        self.calibrated = True

    # -- preprocessing -----------------------------------------------------

    def _prepare(self, media_path: str):
        signal = load_audio(media_path, sample_rate=self.sample_rate)
        features, _ = extract_acoustic_features(
            signal,
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            n_mfcc=self.n_mfcc,
            pitch_tracker=self.pitch_tracker,
        )
        num_valid = features.shape[0]
        features, mask = _pad_or_truncate(features, self.num_audio_frames)
        return (
            torch.from_numpy(features).float().unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
            signal,
            num_valid,
        )

    # -- public API --------------------------------------------------------

    def infer(self, media_path: str, file_name: Optional[str] = None) -> Dict:
        """Scores one audio file (or a video container whose audio is muxed
        in -- .mp4 works, the video track is simply never decoded)."""
        with self._lock:
            return self._infer_locked(media_path, file_name)

    def _infer_locked(self, media_path: str, file_name: Optional[str] = None) -> Dict:
        features_t, mask, signal, num_valid = self._prepare(media_path)

        with torch.no_grad():
            logit = float(self.model(features_t, padding_mask=mask).item())

        p_fake = float(torch.sigmoid(torch.tensor(logit)).item())
        c_score = float(apply_temperature(np.array([logit]), self.temperature)[0])
        decision = decide(c_score, self.tau_lo, self.tau_hi)

        return {
            "sampleId": f"sample_{int(time.time() * 1000):x}",
            "fileName": file_name or Path(media_path).name,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "modality": "audio",
            "decision": decision,
            "calibrated": self.calibrated,
            "logit": round(logit, 5),
            "pFake": round(p_fake, 4),
            "cScore": round(c_score, 4),
            "tauLo": self.tau_lo,
            "tauHi": self.tau_hi,
            "temperature": self.temperature,
            "falseSuppressionRate": self.false_suppression_rate,
            "reviewQueueRate": self.review_queue_rate,
            "acousticShap": self._shap(features_t, mask) if self.cfg.explain else [],
            "waveform": _waveform(signal, self.sample_rate),
            "framesAnalysed": int(min(num_valid, self.num_audio_frames)),
            "framesAvailable": int(num_valid),
        }

    def _shap(self, features_t, mask) -> List[Dict]:
        if self._background is None:
            # Zero baseline: after the encoder's input BatchNorm this is the
            # per-feature mean, i.e. "an average clip" -- the right SHAP
            # reference without shipping a stored statistics artefact.
            self._background = torch.zeros_like(features_t[0])

        attributions = explain_acoustic_features(
            self.model, features_t, mask, self.feature_names,
            self._background, self.device, n_samples=self.cfg.shap_samples,
        )[0]
        return [
            {"feature": name, "value": round(value, 5)}
            for name, value in top_k_features(attributions, k=self.cfg.top_k_features)
        ]
