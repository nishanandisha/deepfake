#!/usr/bin/env bash
# Bring up both models plus the router the UI talks to.
#
#   :8001  deepfake-detect   old multimodal model, visual branch (advisory)
#   :8002  ai-voice-detect   WavLM audio model (drives the verdict)
#   :8000  router            what the UI connects to
#
# The video backend is optional: if its checkpoint is missing the router
# still serves audio, and video uploads fall back to an audio-only verdict.
set -u
ROOT="/Users/nishanishmitha/Desktop/major /major_project"
AVD="$ROOT/ai-voice-detect"
DFD="$ROOT/deepfake-detect"
mkdir -p "$AVD/outputs/logs"

# Make ffmpeg reachable by the servers.
#
# Both backends shell out to ffmpeg for anything libsndfile cannot open --
# every .mp4/.mov upload, and voice-note formats like .opus and .m4a. The
# servers inherit this script's PATH, so when the script is launched from a
# shell whose PATH lacks Homebrew (a GUI launcher, a login shell that never
# sourced the profile, a restricted environment) ffmpeg vanishes and every
# such upload fails with a 500 that names decoding, not PATH.
#
# This is the project's bug #4 wearing different clothes: "no ffmpeg meant
# 100% of audio decodes failed", masked at the time by per-clip error
# tolerance. Resolve it here, once, rather than depending on the caller.
if ! command -v ffmpeg >/dev/null 2>&1; then
  for d in /opt/homebrew/bin /usr/local/bin /opt/local/bin; do
    [ -x "$d/ffmpeg" ] && { PATH="$d:$PATH"; break; }
  done
  export PATH
fi
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg                  $(command -v ffmpeg)"
else
  echo "WARNING: ffmpeg not found. .mp4/.mov/.opus/.m4a uploads will fail to"
  echo "         decode; .wav/.flac still work. Install with: brew install ffmpeg"
fi

for port in 8000 8001 8002; do lsof -ti:$port | xargs kill -9 2>/dev/null; done

export SSL_CERT_FILE="$("$AVD/.venv/bin/python" -c 'import certifi;print(certifi.where())')"

echo "starting audio backend  :8002"
( cd "$AVD" && nohup "$AVD/.venv/bin/python" "$AVD/scripts/serve.py" --layer 12 --port 8002 \
  > "$AVD/outputs/logs/audio.log" 2>&1 & )

CKPT="$DFD/outputs/exported/fusion/checkpoints/best.pt"
POLICY="$DFD/outputs/exported/calibration/policy.json"
if [ -f "$CKPT" ] && [ -f "$POLICY" ]; then
  echo "starting video backend  :8001"
  ( cd "$DFD" && nohup "$DFD/.venv/bin/python" scripts/serve.py \
      --fusion-checkpoint "$CKPT" --policy-json "$POLICY" --port 8001 \
      > "$AVD/outputs/logs/video.log" 2>&1 & )
else
  echo "video backend skipped (no checkpoint at $CKPT)"
fi

echo "starting router         :8000"
nohup "$AVD/.venv/bin/python" "$AVD/scripts/router.py" --port 8000 \
  > "$AVD/outputs/logs/router.log" 2>&1 &

# Wait for the router AND for every backend it proxies to report up. The
# router answers /api/health as soon as it is listening, so polling it alone
# returns `video: up:false` while the visual backend is still loading its
# torch checkpoint -- which reads as a failure when it is only slow.
echo "waiting for backends ..."
for _ in $(seq 1 90); do
  health=$(curl -sf http://localhost:8000/api/health 2>/dev/null) || { sleep 2; continue; }
  if [ ! -f "$CKPT" ] || [ ! -f "$POLICY" ]; then
    case "$health" in *'"audio"'*'"up": true'*) break ;; esac
  else
    case "$health" in *'"up": false'*) ;; *'"up": true'*) break ;; esac
  fi
  sleep 2
done
curl -s http://localhost:8000/api/health
echo
