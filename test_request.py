#!/usr/bin/env python3
"""
Script para probar el endpoint de RunPod con AnimateDiff.
Reemplaza 'TU_API_KEY' con tu API key de RunPod y 'TU_ENDPOINT_ID' con el ID de tu endpoint.
"""

import requests
import json
import time
import os
import base64
from datetime import datetime

# Configuración
RUNPOD_API_KEY = "TU_API_KEY"  # Reemplaza con tu API key de RunPod
ENDPOINT_ID = "TU_ENDPOINT_ID"  # Reemplaza con el ID de tu endpoint
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

# Datos de prueba
test_input = {
    "input": {
        "prompt": "a beautiful landscape with mountains and a lake, best quality, masterpiece",
        "steps": 20,
        "width": 512,
        "height": 512,
        "n_prompt": "low quality, bad anatomy, worst quality, low resolution",
        "guidance_scale": 7.5,
        "seed": 42
    }
}

# Función para enviar la solicitud
def send_request():
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Enviando solicitud a {API_URL}...")
    print(f"Datos de entrada: {json.dumps(test_input, indent=2)}")
    
    response = requests.post(API_URL, headers=headers, json=test_input)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Solicitud enviada correctamente. ID: {data.get('id')}")
        return data.get('id')
    else:
        print(f"Error al enviar la solicitud: {response.status_code}")
        print(f"Respuesta: {response.text}")
        return None

# Función para verificar el estado de la solicitud
def check_status(request_id):
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{request_id}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    print(f"Verificando estado de la solicitud {request_id}...")
    
    while True:
        response = requests.get(status_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            
            print(f"Estado actual: {status}")
            
            if status == "COMPLETED":
                print("¡Solicitud completada!")
                return data.get('output')
            elif status == "FAILED":
                print(f"La solicitud falló: {data.get('error')}")
                return None
            elif status == "IN_QUEUE" or status == "IN_PROGRESS":
                print(f"La solicitud está {status}, esperando 5 segundos...")
                time.sleep(5)
            else:
                print(f"Estado desconocido: {status}")
                return None
        else:
            print(f"Error al verificar el estado: {response.status_code}")
            print(f"Respuesta: {response.text}")
            return None

# Función para guardar el video
def save_video(video_url):
    if not video_url:
        print("No se recibió URL de video")
        return
    
    print(f"Descargando video desde {video_url}...")
    
    response = requests.get(video_url)
    
    if response.status_code == 200:
        # Crear directorio para guardar el video si no existe
        os.makedirs("output", exist_ok=True)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/animatediff_{timestamp}.mp4"
        
        # Guardar el video
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"Video guardado como {filename}")
    else:
        print(f"Error al descargar el video: {response.status_code}")

# Función principal
def main():
    print("=== Prueba de RunPod con AnimateDiff ===")
    
    # Enviar solicitud
    request_id = send_request()
    
    if not request_id:
        print("No se pudo obtener el ID de la solicitud")
        return
    
    # Verificar estado
    output = check_status(request_id)
    
    if output:
        print(f"Resultado: {json.dumps(output, indent=2)}")
        
        # Si el resultado contiene una URL de video, guardarla
        if isinstance(output, dict) and "video_url" in output:
            save_video(output["video_url"])
        elif isinstance(output, str) and output.startswith("http"):
            save_video(output)
        else:
            print("No se encontró URL de video en la respuesta")

if __name__ == "__main__":
    main()