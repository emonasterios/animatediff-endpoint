#!/bin/bash
# Script para probar el endpoint de RunPod para AnimateDiff

# Verificar si se proporcionaron los argumentos necesarios
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Uso: $0 <endpoint_id> <api_key> [prompt]"
    echo "Ejemplo: $0 abc123 sk_1234567890abcdef \"A cat walking\""
    exit 1
fi

ENDPOINT_ID=$1
API_KEY=$2
PROMPT=${3:-"A cat walking"}

# Ejecutar el script de Python
python3 $(dirname "$0")/test_runpod.py --endpoint-id "$ENDPOINT_ID" --api-key "$API_KEY" --prompt "$PROMPT"