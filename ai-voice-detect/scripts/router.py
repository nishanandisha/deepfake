"""One endpoint that handles both audio and video uploads.

The two models cannot share a process: `deepfake-detect` and
`ai-voice-detect` both ship a top-level package called `src`, so importing
one shadows the other. They also need different dependency sets (OpenCV and
Hydra versus torchaudio). So each runs as its own server and this proxies
between them:

    UI :3000 -> router :8000 -+-> :8001  deepfake-detect  (visual branch)
                              +-> :8002  ai-voice-detect  (WavLM audio)

Routing is by probe, not by extension: an .mp4 can carry no video track, and
an .mp3 with embedded cover art exposes that artwork as a single-frame video
stream. Two decodable frames are required before a file is treated as video.

**Which branch decides the verdict.** The audio model drives it. That is not
arbitrary -- it is measured. The visual branch of the old model was tested
against degenerate input and scores a pure-black clip at 0.0039 where a real
clip scores 0.0038, while white, grey and noise frames all read as 0.84-0.98
"fake". Its Haar-cascade face detector misses on 0-69% of frames even on
clips from its own training distribution, and when it misses it silently
centre-crops background pixels. A verdict driven by that branch would flip on
whether a face happened to be found.

So the visual score is reported for display and explicitly marked advisory
via `visualAdvisory`; the headline `cScore` comes from the audio model, which
has a measured 2.00% EER on a held-out test split including two generators
withheld from training.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AUDIO_BACKEND = "http://localhost:8002"
VIDEO_BACKEND = "http://localhost:8001"
TIMEOUT = 180


def _has_video_stream(path: str) -> bool:
    """True when ffprobe reports at least two decodable video frames.

    Two, not one: an MP3/M4A carrying cover art exposes the artwork as a
    single-frame video stream, and treating that as a video sends the file
    down a path where the "video" is one still image.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip().splitlines()
        return bool(out) and int(out[0] or 0) >= 2
    except Exception:  # noqa: BLE001 - ffprobe absent or unhappy: assume audio
        return False


def _post_file(url: str, path: str, filename: str) -> dict:
    """Multipart POST without pulling in `requests`."""
    boundary = f"----router{uuid.uuid4().hex}"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        Path(path).read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{url}/api/infer", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def _backend_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=2) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def _merge(audio: dict, video: dict) -> dict:
    """Combine the two branches, escalating if *either* implicates the clip.

    The branches answer genuinely different questions, and neither subsumes
    the other:

      audio  -- "is this voice AI-generated?"   (synthetic speech)
      visual -- "was this face manipulated?"    (edited pixels)

    A face-swapped video of a real person's real voice is invisible to the
    audio model, and correctly so: the voice *is* human. Measured on LAV-DF
    clips whose manipulation is spliced real speech, the audio model returns
    0.0001 while the visual branch returns 0.9929. Letting audio alone decide
    would report those as authentic.

    So the fused score is the max: a clip is suspect if either branch says so.
    The cost is inherited from the visual branch's known failure mode -- it
    scores blank, grey and noise frames at 0.84-0.98, and its Haar face
    detector fails silently by centre-cropping background -- which means a
    video whose face detection fails can be escalated on visual evidence that
    is not real. `drivenBy` records which branch set the score so the UI can
    say why, and `visualAdvisory` carries the caveat.
    """
    merged = dict(audio)
    merged["hasVideo"] = True
    merged["visualSaliency"] = video.get("visualSaliency", [])

    p_audio = float(audio.get("yHatAcoustic", 0.0))
    p_visual = video.get("yHatVisual")
    merged["yHatVisual"] = p_visual

    if p_visual is None:
        return merged

    p_visual = float(p_visual)
    p_fused = max(p_audio, p_visual)
    driven_by = "visual" if p_visual > p_audio else "audio"

    merged["yHatFused"] = round(p_fused, 4)
    merged["cScore"] = round(1.0 - p_fused, 4)
    merged["decision"] = ("block" if merged["cScore"] < merged["tauLo"]
                          else "approve" if merged["cScore"] >= merged["tauHi"]
                          else "flag")
    merged["drivenBy"] = driven_by

    if merged["decision"] == "approve":
        merged["manipulatedModalityGuess"] = "none"
        merged["scenario"] = "authentic"
    elif p_visual > 0.5 and p_audio > 0.5:
        merged["manipulatedModalityGuess"] = "both"
        merged["scenario"] = "fake_both"
    elif driven_by == "visual":
        merged["manipulatedModalityGuess"] = "video"
        merged["scenario"] = "fake_video"
    else:
        merged["manipulatedModalityGuess"] = "audio"
        merged["scenario"] = "fake_audio"

    merged["visualAdvisory"] = {
        "yHatVisual": p_visual,
        "note": ("Visual branch is the old model and is unreliable in "
                 "isolation: it scores blank and noise frames at 0.84-0.98 "
                 "and its face detector fails silently. Treat a visual-driven "
                 "verdict as a prompt to review, not a conclusion."),
    }
    if video.get("waveform") and not merged.get("waveform", {}).get("peaks"):
        merged["waveform"] = video["waveform"]
    return merged


def _parse_upload(handler):
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
        if 'name="file"' not in disposition:
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
        audio_up, video_up = _backend_up(AUDIO_BACKEND), _backend_up(VIDEO_BACKEND)
        info = {
            # The UI only needs audio to be up; video degrades gracefully.
            "status": "ready" if audio_up else "unavailable",
            "checkpoint": "router",
            "backends": {
                "audio": {"url": AUDIO_BACKEND, "up": audio_up,
                          "role": "verdict (WavLM, 2.00% test EER)"},
                "video": {"url": VIDEO_BACKEND, "up": video_up,
                          "role": "advisory visual branch only"},
            },
        }
        if path == "/api/health":
            self._send(200 if audio_up else 503, info)
        elif path in ("", "/"):
            self._send(200, {**info, "endpoints": {
                "GET /api/health": "readiness",
                "POST /api/infer": "multipart upload (field 'file')",
            }})
        else:
            self._send(404, {"error": "not found"})

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

            suffix = Path(filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                temp_path = tmp.name

            is_video = _has_video_stream(temp_path)
            started = time.time()

            # The audio model runs for every upload, video included: a video's
            # audio track is exactly what it was validated on.
            audio = _post_file(AUDIO_BACKEND, temp_path, filename)

            if is_video and _backend_up(VIDEO_BACKEND):
                try:
                    video = _post_file(VIDEO_BACKEND, temp_path, filename)
                    result = _merge(audio, video)
                except Exception as error:  # noqa: BLE001
                    # A failing visual branch must not lose the audio verdict.
                    traceback.print_exc()
                    result = dict(audio)
                    result["visualAdvisory"] = {"error": str(error)}
            elif is_video:
                result = dict(audio)
                result["visualAdvisory"] = {
                    "note": "Video track present but the visual backend is not "
                            "running; verdict is from the audio track alone.",
                }
            else:
                result = audio

            result["routedAs"] = "video" if is_video else "audio"
            result["elapsedSeconds"] = round(time.time() - started, 2)
            self._send(200, result)

        except urllib.error.URLError as error:
            self._send(502, {"error": f"backend unreachable: {error}"})
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": str(error)})
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Route uploads to both models.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"router on http://localhost:{args.port}", flush=True)
    print(f"  audio (verdict)  -> {AUDIO_BACKEND}", flush=True)
    print(f"  video (advisory) -> {VIDEO_BACKEND}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
