#!/usr/bin/env python3
"""
Script para verificar si los modelos se cargan correctamente.
Este script intenta cargar los modelos necesarios para AnimateDiff y reporta cualquier error.
"""

import os
import sys
import logging
import argparse
import traceback
import json
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("check-model-loading")

# Añadir directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_model_files():
    """Verifica si los archivos de modelo existen."""
    logger.info("=== Verificando archivos de modelo ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Verificar modelo de Stable Diffusion
    sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
    if os.path.exists(sd_dir) and os.listdir(sd_dir):
        logger.info(f"Modelo Stable Diffusion encontrado en {sd_dir}")
        # Listar archivos
        files = os.listdir(sd_dir)
        logger.info(f"Archivos: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
    else:
        logger.error(f"Modelo Stable Diffusion no encontrado en {sd_dir}")
        return False
    
    # Verificar módulos de movimiento
    motion_module_dir = os.path.join(base_dir, "models", "Motion_Module")
    
    mm_v1_path = os.path.join(motion_module_dir, "mm_sd_v15_v1-fp16.safetensors")
    mm_v2_path = os.path.join(motion_module_dir, "mm_sd_v15_v2-fp16.safetensors")
    
    if os.path.exists(mm_v1_path):
        size_mb = os.path.getsize(mm_v1_path) / (1024 * 1024)
        logger.info(f"Módulo de movimiento v1 encontrado en {mm_v1_path} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"Módulo de movimiento v1 no encontrado en {mm_v1_path}")
    
    if os.path.exists(mm_v2_path):
        size_mb = os.path.getsize(mm_v2_path) / (1024 * 1024)
        logger.info(f"Módulo de movimiento v2 encontrado en {mm_v2_path} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"Módulo de movimiento v2 no encontrado en {mm_v2_path}")
    
    return True

def try_load_diffusers():
    """Intenta cargar el modelo de Stable Diffusion usando diffusers."""
    logger.info("=== Intentando cargar modelo con diffusers ===")
    
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
        
        logger.info(f"Cargando modelo desde {sd_dir}")
        
        # Intentar cargar el modelo
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            logger.info("Moviendo modelo a GPU...")
            pipe = pipe.to("cuda")
        
        logger.info("Modelo cargado correctamente con diffusers")
        return True
    except Exception as e:
        logger.error(f"Error al cargar el modelo con diffusers: {e}")
        traceback.print_exc()
        return False

def try_load_animatediff():
    """Intenta cargar AnimateDiff."""
    logger.info("=== Intentando cargar AnimateDiff ===")
    
    try:
        # Importar AnimateDiff
        from inference_util import AnimateDiff
        
        logger.info("Inicializando AnimateDiff...")
        
        # Inicializar AnimateDiff
        animatediff = AnimateDiff()
        
        logger.info("AnimateDiff inicializado correctamente")
        
        # Intentar una inferencia simple
        logger.info("Intentando una inferencia simple...")
        
        try:
            save_path = animatediff.inference(
                prompt="A cat walking",
                steps=5,  # Menos pasos para una prueba rápida
                width=512,
                height=512,
                n_prompt="bad quality, blurry",
                guidance_scale=7.5,
                seed=42
            )
            
            logger.info(f"Inferencia completada correctamente. Video guardado en {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error durante la inferencia: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        logger.error(f"Error al cargar AnimateDiff: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Verificar la carga de modelos para AnimateDiff")
    parser.add_argument("--skip-inference", action="store_true", help="Omitir la prueba de inferencia")
    parser.add_argument("--json", action="store_true", help="Mostrar resultados en formato JSON")
    parser.add_argument("--output", help="Guardar resultados en un archivo")
    args = parser.parse_args()
    
    results = {
        "model_files_exist": False,
        "diffusers_load": False,
        "animatediff_load": False,
        "inference": False
    }
    
    # Verificar archivos de modelo
    results["model_files_exist"] = check_model_files()
    
    # Intentar cargar con diffusers
    results["diffusers_load"] = try_load_diffusers()
    
    # Intentar cargar AnimateDiff
    results["animatediff_load"] = try_load_animatediff()
    
    # Mostrar resultados
    logger.info("=== Resultados ===")
    logger.info(f"Archivos de modelo existen: {results['model_files_exist']}")
    logger.info(f"Carga con diffusers: {results['diffusers_load']}")
    logger.info(f"Carga de AnimateDiff: {results['animatediff_load']}")
    
    # Mostrar resultados en formato JSON si se solicita
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Guardar resultados en un archivo si se especifica
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Resultados guardados en {args.output}")
        except Exception as e:
            logger.error(f"Error al guardar resultados: {e}")
    
    # Determinar éxito
    success = results["model_files_exist"] and results["diffusers_load"] and results["animatediff_load"]
    
    if success:
        logger.info("Todos los modelos se cargaron correctamente")
    else:
        logger.error("Hubo errores al cargar los modelos. Revise los mensajes anteriores.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)