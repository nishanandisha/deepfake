"""End-to-end inference on an arbitrary audio file.

Two behaviours here exist because of failures measured in the predecessor
project, and both are deliberate:

**It abstains rather than guessing.** That model scored ten seconds of pure
digital silence at 0.98 "fake". Non-speech is not evidence of synthesis; it
is absence of evidence, and the honest output is "cannot analyse". The energy
gate below refuses to emit a verdict when there is not enough voiced audio.

**It scores long files in windows.** That model analysed only the first eight
seconds of any upload -- so a manipulated segment two minutes into a video was
never seen. Here a long file is cut into overlapping windows, each scored
independently, and both the mean and the max are reported. The max is what
catches a short synthetic insert inside otherwise genuine audio; the mean is
the stable summary.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.evaluation.calibration import apply_temperature
from src.models.head import VoiceClassifierHead
from src.preprocessing.embeddings import (
    SAMPLE_RATE,
    EmbeddingCache,
    WavLMFrontend,
    load_audio,
    pick_device,
)

WINDOW_SECONDS = 8.0
HOP_SECONDS = 4.0
MIN_SPEECH_SECONDS = 0.5


@dataclass
class Prediction:
    file_name: str
    verdict: str                 # "ai" | "human" | "abstain"
    probability: Optional[float]  # calibrated P(AI-generated)
    duration: float
    speech_seconds: float
    windows: List[Dict]
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "fileName": self.file_name,
            "verdict": self.verdict,
            "pAiGenerated": self.probability,
            "durationSeconds": round(self.duration, 2),
            "speechSeconds": round(self.speech_seconds, 2),
            "windows": self.windows,
            "reason": self.reason,
        }


def voiced_seconds(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Rough voiced-duration estimate from short-time energy.

    An adaptive threshold rather than a fixed one: a quiet recording is still
    speech, and a fixed floor would reject it. Frames above a fraction of the
    clip's own 95th-percentile energy count as voiced, which separates speech
    from silence and room tone without a heavyweight VAD dependency.
    """
    frame = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    if len(signal) < frame:
        return 0.0

    frames = np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop]
    energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    if not len(energy):
        return 0.0

    reference = np.percentile(energy, 95)
    if reference < 1e-4:  # essentially digital silence
        return 0.0
    return float(np.sum(energy > 0.15 * reference) * hop / sample_rate)


def _windows(signal: np.ndarray, sample_rate: int = SAMPLE_RATE):
    size = int(WINDOW_SECONDS * sample_rate)
    hop = int(HOP_SECONDS * sample_rate)
    if len(signal) <= size:
        yield 0.0, signal
        return
    for start in range(0, len(signal) - size + 1, hop):
        yield start / sample_rate, signal[start : start + size]


class VoicePredictor:
    def __init__(self, checkpoint: str, policy: Dict[str, float], layer: int = None,
                 device: torch.device = None):
        self.device = device or pick_device()
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = state["config"]
        self.layer = layer or cfg["layer"]
        self.max_frames = cfg["max_frames"]

        self.model = VoiceClassifierHead(
            input_dim=768, proj_dim=cfg["proj_dim"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.frontend = WavLMFrontend(device=self.device, layer=self.layer)
        self.temperature = policy.get("temperature", 1.0)
        self.threshold = policy.get("threshold", 0.5)

    @torch.no_grad()
    def _score(self, signal: np.ndarray) -> float:
        features = self.frontend.embed(signal)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        return float(self.model(tensor).item())

    def predict(self, path: str) -> Prediction:
        signal = load_audio(path)
        duration = len(signal) / SAMPLE_RATE
        speech = voiced_seconds(signal)
        name = Path(path).name

        if speech < MIN_SPEECH_SECONDS:
            return Prediction(
                name, "abstain", None, duration, speech, [],
                reason=(f"only {speech:.2f}s of voiced audio detected "
                        f"({MIN_SPEECH_SECONDS}s needed) -- nothing to analyse"),
            )

        windows = []
        for offset, chunk in _windows(signal):
            logit = self._score(chunk)
            windows.append({
                "startSeconds": round(offset, 2),
                "endSeconds": round(offset + len(chunk) / SAMPLE_RATE, 2),
                "pAiGenerated": round(float(apply_temperature(np.array([logit]), self.temperature)[0]), 4),
            })

        probabilities = np.array([w["pAiGenerated"] for w in windows])
        # Mean is the summary; max is reported alongside so a short synthetic
        # insert in a long genuine recording is not averaged into invisibility.
        aggregate = float(probabilities.mean())
        verdict = "ai" if aggregate >= self.threshold else "human"

        return Prediction(
            name, verdict, round(aggregate, 4), duration, speech, windows,
            reason=f"{len(windows)} window(s), max {probabilities.max():.4f}",
        )
