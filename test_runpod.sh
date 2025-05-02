#!/bin/bash
# Script para probar el endpoint de RunPod de AnimateDiff

# Verificar si se proporcionaron los argumentos necesarios
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Uso: $0 <endpoint_id> <api_key> [prompt] [output_dir]"
    echo "Ejemplo: $0 abc123def456 runpod_api_key_12345 'A cat walking' ./output"
    exit 1
fi

ENDPOINT_ID=$1
API_KEY=$2
PROMPT=${3:-"A cat walking"}
OUTPUT_DIR=${4:-"./output"}

# Ejecutar el script de Python
python test_runpod.py --endpoint-id "$ENDPOINT_ID" --api-key "$API_KEY" --prompt "$PROMPT" --output-dir "$OUTPUT_DIR"

# Verificar el resultado
if [ $? -eq 0 ]; then
    echo "La prueba se completó correctamente"
    exit 0
else
    echo "La prueba falló"
    exit 1
fi