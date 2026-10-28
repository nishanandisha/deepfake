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

# Localisation. An 8s/4s window answers "is this clip synthetic"; it cannot say
# *where*, and for any clip under 8s it degenerates to a single window covering
# everything -- which is why a spliced file used to come back as one number.
#
# These were fitted on four splices (Hume, Kokoro, ElevenLabs, Polly inserted
# into genuine YouTube speech at different offsets) plus an all-real and an
# all-AI control: mean frame-IoU 0.753 against ground truth, no regions at all
# on the all-real control, 98% coverage on the all-AI one.
#
# REGION_THRESHOLD is far below the clip-level threshold (0.1058) on purpose.
# The head was trained on 4-6s clips with attentive pooling, so a 2s slice
# gives it much less evidence to aggregate and its scores sit lower -- inside a
# known-synthetic span they run 0.1-0.6 while genuine speech stays at 0.000-0.02.
# The separation is wide; the absolute values are simply compressed.
LOCALIZE_WINDOW_SECONDS = 2.0
LOCALIZE_HOP_SECONDS = 0.25
REGION_THRESHOLD = 0.10
REGION_MERGE_GAP_SECONDS = 0.5
MIN_REGION_SECONDS = 0.4
EMBED_CHUNK_SECONDS = 60.0

# Two gates on top of the raw regions, both there because the localiser is
# only trustworthy where the model is in-domain.
#
# Measured failure: on LAV-DF clips -- a corpus this model never trained on --
# genuinely human speech produced 2s-window scores of 0.46 and 0.59, as high
# as a real synthetic splice, while the clip-level score correctly stayed at
# 0.008. Short-window scores are not reliable out of domain, so unqualified
# regions would invent findings on authentic audio. That is the same trap the
# UI already guards against for visual salience.
#
# MIN_REGION_CONFIDENCE drops weak hits. MAX_REGION_COVERAGE encodes what
# "partial" means: a spliced insert is a minority of the clip with clean audio
# around it. Regions blanketing most of a file while the clip-level score says
# "human" are two signals disagreeing, and the clip-level one is the one with
# measured accuracy behind it.
MIN_REGION_CONFIDENCE = 0.30
MAX_REGION_COVERAGE = 0.60


@dataclass
class Prediction:
    file_name: str
    verdict: str                 # "ai" | "human" | "abstain"
    probability: Optional[float]  # calibrated P(AI-generated)
    duration: float
    speech_seconds: float
    windows: List[Dict]
    reason: str = ""
    probability_max: Optional[float] = None
    regions: List[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "fileName": self.file_name,
            "verdict": self.verdict,
            "pAiGenerated": self.probability,
            "durationSeconds": round(self.duration, 2),
            "speechSeconds": round(self.speech_seconds, 2),
            "windows": self.windows,
            "reason": self.reason,
            "pAiGeneratedMax": self.probability_max,
            "regions": self.regions or [],
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


def merge_regions(spans, duration, threshold=REGION_THRESHOLD,
                  gap=REGION_MERGE_GAP_SECONDS, min_length=MIN_REGION_SECONDS):
    """Scored overlapping windows -> contiguous suspect regions.

    Each instant takes the *maximum* over every window covering it, so a short
    synthetic insert is not averaged away by the genuine audio sharing its
    windows -- the failure that made an 8s mean return "human" for a clip with
    8.8s of synthetic speech in the middle of it.
    """
    if not spans:
        return []
    grid = np.arange(0.0, duration, 0.05)
    values = np.zeros_like(grid)
    for start, end, probability in spans:
        covered = (grid >= start) & (grid < end)
        values[covered] = np.maximum(values[covered], probability)

    runs, current = [], None
    for instant, value in zip(grid, values):
        if value >= threshold:
            if current is None:
                current = [instant, instant, value]
            else:
                current[1], current[2] = instant, max(current[2], value)
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= gap:
            merged[-1][1] = run[1]
            merged[-1][2] = max(merged[-1][2], run[2])
        else:
            merged.append(list(run))

    return [
        {"startSeconds": round(float(a), 2),
         "endSeconds": round(float(b), 2),
         "confidence": round(float(c), 4)}
        for a, b, c in merged if b - a >= min_length
    ]


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

    @torch.no_grad()
    def _embed_all(self, signal: np.ndarray) -> np.ndarray:
        """Frame embeddings for the whole signal, chunked to bound memory."""
        size = int(EMBED_CHUNK_SECONDS * SAMPLE_RATE)
        if len(signal) <= size:
            return self.frontend.embed(signal)
        return np.concatenate(
            [self.frontend.embed(signal[i : i + size])
             for i in range(0, len(signal), size)],
            axis=0,
        )

    @torch.no_grad()
    def _localize(self, signal: np.ndarray, duration: float):
        """Fine-grained window scores over one full-file embedding.

        The frozen frontend is the expensive part, so it runs once and the head
        slides over the resulting ~50 Hz frames. Re-embedding per window would
        repeat that pass for every 0.25s step.

        The coarse 8s/4s path above is deliberately left alone: its numbers are
        what the reported test metrics were measured on, and recomputing them
        from a sliced full-file embedding would shift them for no stated reason.
        """
        features = self._embed_all(signal)
        total = features.shape[0]
        if total == 0:
            return [], []
        fps = total / duration

        width = int(LOCALIZE_WINDOW_SECONDS * fps)
        hop = max(1, int(LOCALIZE_HOP_SECONDS * fps))
        if total <= width:
            batch = [features]
            bounds = [(0.0, duration)]
        else:
            batch, bounds = [], []
            for start in range(0, total - width + 1, hop):
                batch.append(features[start : start + width])
                bounds.append((start / fps, (start + width) / fps))

        logits = self.model(torch.from_numpy(np.stack(batch)).to(self.device)).cpu().numpy()
        probabilities = apply_temperature(logits, self.temperature)

        spans = [(a, b, float(x)) for (a, b), x in zip(bounds, probabilities)]
        detail = [{"startSeconds": round(a, 2), "endSeconds": round(b, 2),
                   "pAiGenerated": round(float(x), 4)} for a, b, x in spans]
        return spans, detail

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

        spans, _detail = self._localize(signal, duration)
        regions = merge_regions(spans, duration)
        regions = [r for r in regions if r["confidence"] >= MIN_REGION_CONFIDENCE]

        # Contradiction check: lots of "suspect" audio but a clip-level score
        # that says human means the fine-grained scores are not to be trusted
        # here. Report nothing rather than a guess.
        covered = sum(r["endSeconds"] - r["startSeconds"] for r in regions)
        if (duration > 0 and covered / duration > MAX_REGION_COVERAGE
                and aggregate < self.threshold):
            regions = []

        # A clip can be mostly genuine and still contain synthetic speech. The
        # mean cannot express that -- 8.8s of synthetic audio inside a 27s clip
        # averages to 0.08 and reads as "human" -- so a localised region
        # promotes the verdict even when the clip-level mean stays low. The
        # mean itself is left untouched: it is what the published metrics
        # measure, and the regions are reported alongside it rather than
        # silently redefining it.
        if regions and verdict == "human":
            verdict = "partial"

        if regions:
            spans_text = ", ".join(
                f"{r['startSeconds']:.1f}-{r['endSeconds']:.1f}s" for r in regions
            )
            reason = (f"{len(windows)} window(s), max {probabilities.max():.4f}; "
                      f"{len(regions)} suspect region(s): {spans_text}")
        else:
            reason = f"{len(windows)} window(s), max {probabilities.max():.4f}; no suspect regions"

        return Prediction(
            name, verdict, round(aggregate, 4), duration, speech, windows,
            reason=reason,
            probability_max=round(float(probabilities.max()), 4),
            regions=regions,
        )
