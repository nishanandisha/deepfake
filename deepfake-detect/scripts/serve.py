"""HTTP server exposing the inference pipeline to the Next.js UI.

  POST /api/infer   multipart file upload -> InferenceResult JSON
  GET  /api/health  readiness + which checkpoint is loaded

Deliberately thin: all logic lives in src/inference/pipeline.py, so the UI
and the training code stay decoupled. Uses only the standard library so
there is no extra dependency to install before a demo.

Usage:
  python scripts/serve.py --fusion-checkpoint D:/.../fusion/checkpoints/best.pt \\
                          --policy-json      D:/.../calibration/policy.json
"""

import argparse
import cgi
import json
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir

from src.inference.pipeline import DeepfakeInferencePipeline, InferenceConfig

PIPELINE: DeepfakeInferencePipeline = None
CHECKPOINT_NAME = ""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The UI dev server runs on a different port, so CORS is required.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802 - BaseHTTPRequestHandler naming
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.rstrip("/")

        if path == "/api/health":
            self._send(200, {"status": "ready", "checkpoint": CHECKPOINT_NAME,
                             "device": str(PIPELINE.device)})
        elif path in ("", "/"):
            # This is an API, not a site. Hitting the root in a browser is a
            # natural thing to try, so say what this is and point at the UI
            # rather than returning a bare 404 that reads like a failure.
            self._send(200, {
                "service": "deepfake-detect inference API",
                "status": "ready",
                "checkpoint": CHECKPOINT_NAME,
                "device": str(PIPELINE.device),
                "endpoints": {
                    "GET /api/health": "readiness check",
                    "POST /api/infer": "multipart upload (field 'file') -> InferenceResult",
                },
                "ui": "http://localhost:3000",
            })
        else:
            self._send(404, {"error": "not found", "try": ["/api/health", "/api/infer"]})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/api/infer":
            self._send(404, {"error": "not found"})
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST",
                         "CONTENT_TYPE": self.headers["Content-Type"]},
            )
            item = form["file"] if "file" in form else None
            if item is None or not item.filename:
                self._send(400, {"error": "no file uploaded under field 'file'"})
                return

            suffix = Path(item.filename).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(item.file.read())
                temp_path = tmp.name

            try:
                result = PIPELINE.infer(temp_path, file_name=item.filename)
                self._send(200, result)
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception as error:  # noqa: BLE001 - surface to the client
            traceback.print_exc()
            self._send(500, {"error": str(error)})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def main() -> None:
    global PIPELINE, CHECKPOINT_NAME

    parser = argparse.ArgumentParser(description="Serve the deepfake inference pipeline.")
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--data", default="medium", help="configs/data/<name>.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--override", action="append", default=[],
        help="extra Hydra override, e.g. model.visual.pooling=mean (repeatable)",
    )
    args = parser.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=["model=fusion", f"data={args.data}", *args.override],
        )

    inference_cfg = InferenceConfig(
        fusion_checkpoint=args.fusion_checkpoint,
        policy_json=args.policy_json,
        frame_rate=cfg.data.frame_rate,
        frame_size=cfg.data.frame_size,
        num_frames=cfg.data.get("num_frames", 16),
        sample_rate=cfg.data.audio_sample_rate,
        frame_ms=cfg.data.audio_frame_ms,
        hop_ms=cfg.data.audio_hop_ms,
        n_mfcc=cfg.model.acoustic.n_mfcc,
        num_audio_frames=cfg.data.get("num_audio_frames", 400),
        pitch_tracker=cfg.data.get("pitch_tracker", "yin"),
    )

    print(f"Loading model from {args.fusion_checkpoint} ...", flush=True)
    PIPELINE = DeepfakeInferencePipeline(inference_cfg, cfg)
    CHECKPOINT_NAME = Path(args.fusion_checkpoint).parent.parent.name
    print(f"Ready on http://localhost:{args.port}  (device={PIPELINE.device})", flush=True)

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
