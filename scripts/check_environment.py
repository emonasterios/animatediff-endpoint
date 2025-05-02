#!/usr/bin/env python3
"""
Script to check the system environment and dependencies for AnimateDiff.
Run this script to diagnose issues with the RunPod worker.
"""

import os
import sys
import platform
import subprocess
import traceback
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("environment-check")

def check_python_version():
    """Check Python version"""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python path: {sys.path}")

def check_system_info():
    """Check system information"""
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    
    # Check memory
    try:
        import psutil
        vm = psutil.virtual_memory()
        logger.info(f"Total memory: {vm.total / (1024**3):.2f} GB")
        logger.info(f"Available memory: {vm.available / (1024**3):.2f} GB")
        logger.info(f"Used memory: {vm.used / (1024**3):.2f} GB")
        logger.info(f"Memory percent: {vm.percent}%")
    except ImportError:
        logger.warning("psutil not installed, skipping memory check")
        
    # Check disk space
    try:
        total, used, free = os.statvfs('/').f_blocks, os.statvfs('/').f_bfree, os.statvfs('/').f_bavail
        logger.info(f"Disk space - total: {total * os.statvfs('/').f_frsize / (1024**3):.2f} GB, "
                   f"used: {(total - free) * os.statvfs('/').f_frsize / (1024**3):.2f} GB, "
                   f"free: {free * os.statvfs('/').f_frsize / (1024**3):.2f} GB")
    except Exception as e:
        logger.warning(f"Error checking disk space: {e}")

def check_cuda():
    """Check CUDA availability and version"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
                logger.info(f"CUDA device {i} capability: {torch.cuda.get_device_capability(i)}")
                logger.info(f"CUDA device {i} memory - allocated: {torch.cuda.memory_allocated(i) / (1024**2):.2f} MB, "
                           f"reserved: {torch.cuda.memory_reserved(i) / (1024**2):.2f} MB")
    except ImportError:
        logger.warning("PyTorch not installed, skipping CUDA check")
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        traceback.print_exc()

def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = [
        "torch", "runpod", "imageio", "omegaconf", "safetensors", "einops", 
        "transformers", "accelerate", "diffusers", "huggingface_hub"
    ]
    
    for dep in dependencies:
        try:
            spec = importlib.util.find_spec(dep)
            if spec is not None:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                logger.info(f"{dep}: installed (version: {version})")
            else:
                logger.warning(f"{dep}: not installed")
        except ImportError:
            logger.warning(f"{dep}: not installed")
        except Exception as e:
            logger.error(f"Error checking {dep}: {e}")

def check_model_files():
    """Check if required model files exist"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for motion module
    motion_module_dir = os.path.join(base_dir, "models/Motion_Module")
    if os.path.exists(motion_module_dir):
        logger.info(f"Motion module directory exists: {motion_module_dir}")
        motion_files = os.listdir(motion_module_dir)
        logger.info(f"Motion module files: {motion_files}")
        
        # Check file sizes
        for file in motion_files:
            file_path = os.path.join(motion_module_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  {file}: {size_mb:.2f} MB")
    else:
        logger.warning(f"Motion module directory does not exist: {motion_module_dir}")
    
    # Check for base model
    base_model_dir = os.path.join(base_dir, "models/StableDiffusion")
    if os.path.exists(base_model_dir):
        logger.info(f"Base model directory exists: {base_model_dir}")
        for model_dir in os.listdir(base_model_dir):
            model_path = os.path.join(base_model_dir, model_dir)
            if os.path.isdir(model_path):
                logger.info(f"Model directory: {model_dir}")
                try:
                    model_files = os.listdir(model_path)
                    logger.info(f"  Files: {model_files}")
                except Exception as e:
                    logger.error(f"  Error listing files: {e}")
    else:
        logger.warning(f"Base model directory does not exist: {base_model_dir}")
    
    # Check for LoRA models
    lora_dir = os.path.join(base_dir, "models/DreamBooth_LoRA")
    if os.path.exists(lora_dir):
        logger.info(f"LoRA directory exists: {lora_dir}")
        lora_files = os.listdir(lora_dir)
        logger.info(f"LoRA files count: {len(lora_files)}")
        logger.info(f"Sample LoRA files: {lora_files[:5] if len(lora_files) > 5 else lora_files}")
    else:
        logger.warning(f"LoRA directory does not exist: {lora_dir}")

def check_runpod_environment():
    """Check RunPod environment variables"""
    runpod_vars = [var for var in os.environ if var.startswith("RUNPOD_")]
    
    if runpod_vars:
        logger.info("RunPod environment variables:")
        for var in runpod_vars:
            # Don't log sensitive values
            if "TOKEN" in var or "SECRET" in var or "KEY" in var:
                logger.info(f"  {var}: [REDACTED]")
            else:
                logger.info(f"  {var}: {os.environ.get(var)}")
    else:
        logger.warning("No RunPod environment variables found")

def main():
    """Main function"""
    logger.info("=== AnimateDiff Environment Check ===")
    
    try:
        check_python_version()
        check_system_info()
        check_cuda()
        check_dependencies()
        check_model_files()
        check_runpod_environment()
        
        logger.info("Environment check completed successfully")
    except Exception as e:
        logger.error(f"Error during environment check: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()