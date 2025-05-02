#!/bin/bash
# Script para probar el endpoint de RunPod con AnimateDiff usando curl
# Reemplaza 'TU_API_KEY' con tu API key de RunPod y 'TU_ENDPOINT_ID' con el ID de tu endpoint

API_KEY="TU_API_KEY"
ENDPOINT_ID="TU_ENDPOINT_ID"
API_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/run"

# Datos de prueba
JSON_DATA='{
  "input": {
    "prompt": "a beautiful landscape with mountains and a lake, best quality, masterpiece",
    "steps": 20,
    "width": 512,
    "height": 512,
    "n_prompt": "low quality, bad anatomy, worst quality, low resolution",
    "guidance_scale": 7.5,
    "seed": 42
  }
}'

echo "Enviando solicitud a ${API_URL}..."
echo "Datos de entrada: ${JSON_DATA}"

# Enviar solicitud
RESPONSE=$(curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "${JSON_DATA}")

echo "Respuesta: ${RESPONSE}"

# Extraer ID de la solicitud
REQUEST_ID=$(echo ${RESPONSE} | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -z "${REQUEST_ID}" ]; then
  echo "No se pudo obtener el ID de la solicitud"
  exit 1
fi

echo "ID de la solicitud: ${REQUEST_ID}"

# Verificar estado
STATUS_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${REQUEST_ID}"

echo "Verificando estado en ${STATUS_URL}..."

while true; do
  STATUS_RESPONSE=$(curl -s -X GET "${STATUS_URL}" \
    -H "Authorization: Bearer ${API_KEY}")
  
  STATUS=$(echo ${STATUS_RESPONSE} | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
  
  echo "Estado actual: ${STATUS}"
  
  if [ "${STATUS}" = "COMPLETED" ]; then
    echo "¡Solicitud completada!"
    echo "Resultado: ${STATUS_RESPONSE}"
    break
  elif [ "${STATUS}" = "FAILED" ]; then
    echo "La solicitud falló"
    echo "Error: ${STATUS_RESPONSE}"
    break
  elif [ "${STATUS}" = "IN_QUEUE" ] || [ "${STATUS}" = "IN_PROGRESS" ]; then
    echo "La solicitud está ${STATUS}, esperando 5 segundos..."
    sleep 5
  else
    echo "Estado desconocido: ${STATUS}"
    break
  fi
done