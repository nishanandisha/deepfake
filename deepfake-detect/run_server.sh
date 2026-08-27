#!/usr/bin/env bash
# Starts the inference API using the settings in .env
set -euo pipefail

cd "$(dirname "$0")"
set -a; source .env; set +a

export SSL_CERT_FILE="$(./.venv/bin/python -c 'import certifi; print(certifi.where())')"

exec ./.venv/bin/python scripts/serve.py \
  --fusion-checkpoint "$FUSION_CHECKPOINT" \
  --policy-json "$POLICY_JSON" \
  --data "$DATA_CONFIG" \
  --port "$PORT"
