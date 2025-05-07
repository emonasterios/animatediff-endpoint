#!/usr/bin/env bash
set -e

# Descarga los modelos (toma token de la env var)
python3 ./scripts/download_models.py --token "$HUGGINGFACE_TOKEN"

# Levanta el server
exec python3 -u ./server.py
