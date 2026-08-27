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

for port in 8000 8001 8002; do lsof -ti:$port | xargs kill -9 2>/dev/null; done

export SSL_CERT_FILE="$("$AVD/.venv/bin/python" -c 'import certifi;print(certifi.where())')"

echo "starting audio backend  :8002"
nohup "$AVD/.venv/bin/python" "$AVD/scripts/serve.py" --layer 12 --port 8002 \
  > "$AVD/outputs/logs/audio.log" 2>&1 &

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

echo "waiting for backends ..."
for _ in $(seq 1 60); do
  curl -sf http://localhost:8000/api/health >/dev/null 2>&1 && break
  sleep 2
done
curl -s http://localhost:8000/api/health
echo
