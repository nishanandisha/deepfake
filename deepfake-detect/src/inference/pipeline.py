"""Single-clip inference: the one function the UI calls.

Wraps Stages 1-7 end to end -- preprocess -> branches -> fusion ->
calibration -> policy -> explanation -- and returns exactly the shape
`deepfake-detect-ui/lib/types.ts::InferenceResult` expects, so the frontend
can swap its mock engine for this without touching component code.

Deliberately the *only* entry point the serving layer touches: keeping the
UI one function call away from the model means Stage 9 never reaches into
training internals.
"""

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from src.evaluation.calibration import apply_temperature
from src.evaluation.policy import decide
from src.explain.cam_visual import GradCAM, overlay_heatmap
from src.explain.shap_acoustic import explain_acoustic_features, top_k_features
from src.explain.shap_modality import explain_modality_split, implicated_modality
from src.models.acoustic.features import extract_acoustic_features
from src.preprocessing.audio import load_audio
from src.preprocessing.dataset import _pad_or_truncate
from src.preprocessing.video import preprocess_video
from src.training.train_fusion import build_fusion_model_for_inference


@dataclass
class InferenceConfig:
    """Everything the pipeline needs that isn't the clip itself."""

    fusion_checkpoint: str
    policy_json: str
    frame_rate: float = 2
    frame_size: int = 112
    num_frames: int = 16
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 20.0
    n_mfcc: int = 20
    num_audio_frames: int = 400
    pitch_tracker: str = "yin"
    top_k_features: int = 8
    top_k_frames: int = 6
    shap_samples_modality: int = 32
    shap_samples_acoustic: int = 64
    device: str = "auto"


def _to_data_uri(image_rgb: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii") if ok else ""


def _waveform(signal: np.ndarray, sample_rate: int, peaks: int = 400) -> dict:
    """Downsampled peaks for the UI's waveform panel, plus the loudest
    regions flagged as 'suspicious'. Note this is an energy heuristic for
    display only -- it is NOT the model's localization of the manipulation,
    which this architecture does not produce (clip-level classification
    only). Labelled as such so the UI never implies more than we measured.
    """
    duration = len(signal) / sample_rate
    if len(signal) == 0:
        return {"peaks": [], "durationSeconds": 0.0, "suspiciousRegions": []}

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
        "suspiciousRegions": regions[:5],
    }


class DeepfakeInferencePipeline:
    """Loads the model once, then serves many clips.

    Model loading takes seconds and dominates per-clip cost, so the serving
    layer should construct this once at startup rather than per request.
    """

    def __init__(self, cfg: InferenceConfig, model_cfg):
        self.cfg = cfg
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        )

        self.model = build_fusion_model_for_inference(model_cfg)
        state = torch.load(cfg.fusion_checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        policy = json.loads(Path(cfg.policy_json).read_text())
        self.temperature = policy["temperature"]
        self.tau_lo, self.tau_hi = policy["tau_lo"], policy["tau_hi"]
        self.false_suppression_rate = policy.get("false_suppression_rate", 0.0)
        self.review_queue_rate = policy.get("review_queue_rate", 0.0)

        _, self.feature_names = extract_acoustic_features(
            np.zeros(cfg.sample_rate), n_mfcc=cfg.n_mfcc, pitch_tracker=cfg.pitch_tracker
        )
        self._background_acoustic: Optional[torch.Tensor] = None

    # -- preprocessing -----------------------------------------------------

    def _prepare(self, media_path: str):
        cfg = self.cfg
        frames = preprocess_video(media_path, frame_rate=cfg.frame_rate, size=cfg.frame_size)
        if frames.shape[0] == 0:
            frames = np.zeros((1, cfg.frame_size, cfg.frame_size, 3), dtype=np.uint8)
        frames, v_mask = _pad_or_truncate(frames, cfg.num_frames)

        signal = load_audio(media_path, sample_rate=cfg.sample_rate)
        features, _ = extract_acoustic_features(
            signal, sample_rate=cfg.sample_rate, frame_ms=cfg.frame_ms,
            hop_ms=cfg.hop_ms, n_mfcc=cfg.n_mfcc, pitch_tracker=cfg.pitch_tracker,
        )
        features, a_mask = _pad_or_truncate(features, cfg.num_audio_frames)

        return (
            frames,
            torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device) / 255,
            torch.from_numpy(v_mask).unsqueeze(0).to(self.device),
            torch.from_numpy(features).float().unsqueeze(0).to(self.device),
            torch.from_numpy(a_mask).unsqueeze(0).to(self.device),
            signal,
        )

    # -- public API --------------------------------------------------------

    def infer(self, media_path: str, file_name: Optional[str] = None) -> Dict:
        cfg = self.cfg
        frames_uint8, frames_t, v_mask, features_t, a_mask, signal = self._prepare(media_path)

        with torch.no_grad():
            out = self.model(frames_t, v_mask, features_t, a_mask)

        logit = float(out["y_hat_logit"].item())
        c_score = float(apply_temperature(np.array([logit]), self.temperature)[0])
        decision = decide(c_score, self.tau_lo, self.tau_hi)

        modality_split = self._modality_split(out)
        acoustic_shap = self._acoustic_shap(features_t, a_mask)
        saliency = self._saliency(frames_t, v_mask, frames_uint8)

        return {
            "sampleId": f"sample_{int(time.time()*1000):x}",
            "fileName": file_name or Path(media_path).name,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "decision": decision,
            "cScore": round(c_score, 4),
            "tauLo": self.tau_lo,
            "tauHi": self.tau_hi,
            "gate": round(float(out["gate"].item()), 4),
            "yHatVisual": round(float(torch.sigmoid(out["y_hat_visual_logit"]).item()), 4),
            "yHatAcoustic": round(float(torch.sigmoid(out["y_hat_acoustic_logit"]).item()), 4),
            "yHatFused": round(float(torch.sigmoid(out["y_hat_logit"]).item()), 4),
            "acousticShap": acoustic_shap,
            "visualSaliency": saliency,
            "waveform": _waveform(signal, cfg.sample_rate),
            "falseSuppressionRate": self.false_suppression_rate,
            "reviewQueueRate": self.review_queue_rate,
            "manipulatedModalityGuess": implicated_modality(modality_split),
            "scenario": _scenario_for(decision, modality_split),
        }

    def _modality_split(self, out) -> Dict[str, float]:
        zeros = torch.zeros_like(out["pooled_v"][0])
        return explain_modality_split(
            self.model, out["pooled_v"], out["pooled_a"], zeros, zeros,
            self.device, n_samples=self.cfg.shap_samples_modality,
        )[0]

    def _acoustic_shap(self, features_t, a_mask) -> List[Dict]:
        if self._background_acoustic is None:
            # Zero baseline: after the encoder's input BatchNorm this is the
            # per-feature mean, i.e. "an average clip" -- the right SHAP
            # reference without shipping a stored statistics artefact.
            self._background_acoustic = torch.zeros_like(features_t[0])

        attributions = explain_acoustic_features(
            self.model, features_t, a_mask, self.feature_names,
            self._background_acoustic, self.device,
            n_samples=self.cfg.shap_samples_acoustic,
        )[0]
        return [
            {"feature": name, "value": round(value, 5)}
            for name, value in top_k_features(attributions, k=self.cfg.top_k_features)
        ]

    def _saliency(self, frames_t, v_mask, frames_uint8) -> List[Dict]:
        heatmaps = GradCAM(self.model.visual_encoder)(
            frames_t, v_mask, self.model.visual_aux_head
        )
        scores = heatmaps.reshape(heatmaps.shape[0], -1).mean(axis=1)
        order = np.argsort(scores)[::-1][: self.cfg.top_k_frames]

        span = float(scores.max() - scores.min()) or 1.0
        return [
            {
                "frameIndex": int(idx),
                "timestamp": round(float(idx) / self.cfg.frame_rate, 3),
                "salienceScore": round(float((scores[idx] - scores.min()) / span), 4),
                "thumbnailUrl": _to_data_uri(frames_uint8[idx]),
                "heatmapUrl": _to_data_uri(overlay_heatmap(frames_uint8[idx], heatmaps[idx])),
            }
            for idx in sorted(order)
        ]


def _scenario_for(decision: str, split: Dict[str, float]) -> str:
    """Maps a result onto the UI's ScenarioId vocabulary."""
    if decision == "approve":
        return "authentic"
    return {"video": "fake_video", "audio": "fake_audio", "both": "fake_both"}[
        implicated_modality(split) if implicated_modality(split) != "none" else "both"
    ]
