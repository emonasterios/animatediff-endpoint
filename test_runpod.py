#!/usr/bin/env python3
"""
Script para probar el endpoint de RunPod de AnimateDiff.
Este script envía una solicitud de prueba al endpoint y muestra la respuesta.
"""

import os
import sys
import json
import time
import argparse
import requests
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test-runpod")

def test_endpoint(endpoint_id, api_key, prompt="A cat walking", output_dir="./output"):
    """
    Prueba el endpoint de RunPod con una solicitud simple.
    
    Args:
        endpoint_id: ID del endpoint de RunPod
        api_key: API key de RunPod
        prompt: Prompt para la generación de video
        output_dir: Directorio donde guardar el video generado
    
    Returns:
        dict: Respuesta del endpoint
    """
    # Crear directorio de salida si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Preparar la solicitud
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    payload = {
        "input": {
            "prompt": prompt,
            "steps": 25,  # Menos pasos para una prueba rápida
            "width": 512,
            "height": 512,
            "guidance_scale": 7.5,
            "seed": 42
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Enviar la solicitud
    logger.info(f"Enviando solicitud a {url}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Obtener el ID del trabajo
        job_data = response.json()
        job_id = job_data.get("id")
        
        if not job_id:
            logger.error(f"No se pudo obtener el ID del trabajo. Respuesta: {job_data}")
            return None
        
        logger.info(f"Solicitud enviada correctamente. ID del trabajo: {job_id}")
        
        # Esperar a que el trabajo se complete
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        
        logger.info(f"Esperando a que el trabajo se complete...")
        
        max_attempts = 60  # 5 minutos máximo (5 segundos por intento)
        for attempt in range(max_attempts):
            time.sleep(5)
            
            status_response = requests.get(status_url, headers=headers)
            status_data = status_response.json()
            
            status = status_data.get("status")
            logger.info(f"Estado del trabajo: {status}")
            
            if status == "COMPLETED":
                # Trabajo completado, obtener el resultado
                output = status_data.get("output")
                
                if output and "data" in output and "filename" in output:
                    # Guardar el video
                    video_data = output["data"]
                    filename = output["filename"]
                    output_file = output_path / filename
                    
                    import base64
                    with open(output_file, "wb") as f:
                        f.write(base64.b64decode(video_data))
                    
                    logger.info(f"Video guardado en {output_file}")
                    return output
                else:
                    if "error" in output:
                        logger.error(f"Error en el trabajo: {output['error']}")
                    else:
                        logger.error(f"Formato de salida inesperado: {output}")
                    return output
            
            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                logger.error(f"El trabajo falló: {error}")
                return {"error": error}
            
            # Seguir esperando para otros estados (IN_QUEUE, IN_PROGRESS, etc.)
        
        logger.error(f"Tiempo de espera agotado después de {max_attempts * 5} segundos")
        return {"error": "Timeout waiting for job completion"}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al enviar la solicitud: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Probar el endpoint de RunPod de AnimateDiff")
    parser.add_argument("--endpoint-id", required=True, help="ID del endpoint de RunPod")
    parser.add_argument("--api-key", required=True, help="API key de RunPod")
    parser.add_argument("--prompt", default="A cat walking", help="Prompt para la generación de video")
    parser.add_argument("--output-dir", default="./output", help="Directorio donde guardar el video generado")
    args = parser.parse_args()
    
    result = test_endpoint(
        endpoint_id=args.endpoint_id,
        api_key=args.api_key,
        prompt=args.prompt,
        output_dir=args.output_dir
    )
    
    if result:
        if "error" in result:
            logger.error(f"La prueba falló: {result['error']}")
            sys.exit(1)
        else:
            logger.info("La prueba se completó correctamente")
            sys.exit(0)
    else:
        logger.error("La prueba falló sin resultado")
        sys.exit(1)

if __name__ == "__main__":
    main()