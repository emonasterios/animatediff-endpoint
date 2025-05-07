#!/usr/bin/env bash
set -ex

echo "[entrypoint] Starting entrypoint with HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}"

# Descarga los modelos (toma token de la env var)
python3 ./scripts/download_models.py --token "$HUGGINGFACE_TOKEN" --no-fail || true

# Levanta el server
exec python3 -u ./server.py
