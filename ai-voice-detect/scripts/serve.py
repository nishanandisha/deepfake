"""HTTP server exposing the AI-voice model to the existing Next.js UI.

  POST /api/infer   multipart upload -> InferenceResult JSON
  GET  /api/health  readiness + which checkpoint is loaded

The UI (`../deepfake-detect-ui`) was written against the *old* multimodal
deepfake model, so its `InferenceResult` contract talks about video branches,
fusion gates and named-feature SHAP. This model answers a narrower question --
human or AI-generated speech, audio only -- so the mapping is deliberate and
documented per field in `_result()` below rather than faked:

  * `hasVideo` is always false. That flag already exists for voice-note
    uploads, and it is what makes the UI hide the video timeline and saliency
    filmstrip instead of rendering empty panels.
  * `acousticShap` is always empty. This project dropped named-feature SHAP
    when it moved from 68 hand-crafted descriptors to learned WavLM
    embeddings; there are no human-readable feature names left to attribute
    to. Returning [] is honest, inventing labels would not be.
  * `visualSaliency` is always empty for the same reason as `hasVideo`.

Where the UI *can* show real evidence, it gets real evidence: per-window
scores become `waveform.suspiciousRegions`, so a synthetic stretch inside a
longer recording is drawn on the timeline rather than averaged away.

Standard library only, so a demo needs no extra install. `cgi` is avoided
(removed in 3.13); multipart is parsed with `email`.
"""

import argparse
import json
import re
import sys
import tempfile
import time
import traceback
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import load_policy
from src.inference.predict import VoicePredictor, load_audio, SAMPLE_RATE

PREDICTOR: VoicePredictor = None
POLICY: dict = {}
CHECKPOINT_NAME = ""

# Measured on the held-out test split (outputs/run/test_results.json), so the
# UI's operating-point figures describe this model rather than the old one's.
TEST_METRICS = {"eer": 0.0200, "auc": 0.9979, "accuracy": 0.9747}

# The UI routes on an authenticity score with two thresholds. This model has
# one threshold on P(AI); `tau_hi` is its mirror in authenticity space, and
# `tau_lo` marks confident synthesis so borderline clips land in "flag"
# (the UI's uncertain band) instead of being asserted either way.
# tau_hi is the real operating point: above it the clip reads as human.
# tau_lo is set at even odds rather than at 1-threshold, so the UI reserves
# its strongest badge for clips the model puts past P(AI) = 0.5 and shows
# everything between the two as uncertain instead of asserting a verdict.
TAU_LO = 0.50
TAU_HI = 0.8942  # 1 - policy threshold (0.1058)


def _waveform(signal: np.ndarray, windows: list, peaks: int = 400) -> dict:
    """Downsampled peaks plus the windows that crossed the threshold."""
    duration = len(signal) / float(SAMPLE_RATE)
    if len(signal) == 0:
        return {"peaks": [], "durationSeconds": 0.0, "suspiciousRegions": []}

    step = max(len(signal) // peaks, 1)
    trimmed = signal[: step * peaks]
    envelope = np.abs(trimmed.reshape(-1, step)).max(axis=1)
    envelope = envelope / (envelope.max() + 1e-9)

    threshold = POLICY.get("threshold", 0.5)
    regions = [
        {
            "startSeconds": w["startSeconds"],
            "endSeconds": w["endSeconds"],
            "intensity": round(float(w["pAiGenerated"]), 4),
        }
        for w in windows
        if w["pAiGenerated"] >= threshold
    ]
    return {
        "peaks": [round(float(v), 4) for v in envelope],
        "durationSeconds": round(duration, 3),
        "suspiciousRegions": regions,
    }


def _decision(c_score: float) -> str:
    if c_score < TAU_LO:
        return "block"
    if c_score >= TAU_HI:
        return "approve"
    return "flag"


def _result(prediction, signal: np.ndarray) -> dict:
    abstained = prediction.verdict == "abstain"

    if abstained:
        # No speech means no evidence either way. The contract has no
        # "abstain", so this surfaces as the uncertain band with a neutral
        # score -- never as a confident "authentic", which is exactly the
        # failure this project was built to stop.
        p_ai, c_score, decision = 0.5, 0.5, "flag"
    else:
        p_ai = float(prediction.probability)
        c_score = 1.0 - p_ai
        decision = _decision(c_score)

    is_ai = (not abstained) and prediction.verdict == "ai"

    return {
        "sampleId": f"sample_{int(time.time()*1000):x}",
        "fileName": prediction.file_name,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "decision": decision,
        "cScore": round(c_score, 4),
        "tauLo": TAU_LO,
        "tauHi": TAU_HI,
        # Audio-only model: no visual branch exists, so the gate is
        # definitionally all-acoustic and there is no visual score to report.
        "gate": 0.0,
        "yHatVisual": None,
        "yHatAcoustic": round(p_ai, 4),
        "yHatFused": round(p_ai, 4),
        "hasVideo": False,
        # Learned embeddings have no human-readable feature names to attribute.
        "acousticShap": [],
        "visualSaliency": [],
        "waveform": _waveform(signal, prediction.windows),
        "falseSuppressionRate": TEST_METRICS["eer"],
        "reviewQueueRate": round(1.0 - TEST_METRICS["accuracy"], 4),
        "manipulatedModalityGuess": "audio" if is_ai else "none",
        "scenario": "fake_audio" if is_ai else "authentic",
        # Extra fields the old contract lacks. Unknown keys are ignored by the
        # UI, and they let `predict.py --json` and this endpoint agree.
        "verdict": prediction.verdict,
        "reason": prediction.reason,
        "speechSeconds": round(prediction.speech_seconds, 2),
        "windows": prediction.windows,
    }


def _parse_upload(handler) -> tuple:
    """Returns (filename, bytes) for the part named 'file'."""
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(length)

    if "multipart/form-data" not in content_type:
        raise ValueError("expected multipart/form-data")

    message = BytesParser(policy=email_default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "name=\"file\"" not in disposition:
            continue
        match = re.search(r'filename="([^"]*)"', disposition)
        return (match.group(1) if match else "upload"), part.get_payload(decode=True)

    raise ValueError("no file uploaded under field 'file'")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.rstrip("/")
        info = {
            "status": "ready",
            "model": "wavlm_base_plus + attentive-pooling head",
            "checkpoint": CHECKPOINT_NAME,
            "task": "human vs AI-generated speech (audio only)",
            "device": str(PREDICTOR.device),
            "threshold": POLICY.get("threshold"),
            "testMetrics": TEST_METRICS,
        }
        if path == "/api/health":
            self._send(200, info)
        elif path in ("", "/"):
            self._send(200, {**info, "endpoints": {
                "GET /api/health": "readiness",
                "POST /api/infer": "multipart upload (field 'file')",
            }, "ui": "http://localhost:3000"})
        else:
            self._send(404, {"error": "not found", "try": ["/api/health", "/api/infer"]})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/api/infer":
            self._send(404, {"error": "not found"})
            return

        temp_path = None
        try:
            filename, data = _parse_upload(self)
            if not data:
                self._send(400, {"error": "uploaded file was empty"})
                return

            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                temp_path = tmp.name

            prediction = PREDICTOR.predict(temp_path)
            prediction.file_name = filename
            signal = load_audio(temp_path)
            self._send(200, _result(prediction, signal))

        except Exception as error:  # noqa: BLE001 - surface to the client
            traceback.print_exc()
            self._send(500, {"error": str(error)})
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def main() -> None:
    global PREDICTOR, POLICY, CHECKPOINT_NAME

    parser = argparse.ArgumentParser(description="Serve the AI-voice detector.")
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--policy", default="outputs/run/policy.json")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    POLICY = load_policy(args.policy) if Path(args.policy).exists() else {}
    print(f"loading {args.checkpoint} ...", flush=True)
    PREDICTOR = VoicePredictor(args.checkpoint, POLICY, layer=args.layer)
    CHECKPOINT_NAME = str(Path(args.checkpoint))

    print(f"ready on http://localhost:{args.port}  "
          f"(device={PREDICTOR.device}, layer={PREDICTOR.layer}, "
          f"threshold={POLICY.get('threshold')})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
