#!/usr/bin/env python3
"""
Script to download the required model files for AnimateDiff.
This script downloads:
1. The base Stable Diffusion v1.5 model
2. The motion modules for AnimateDiff
3. Optional example LoRA files
"""

import os
import sys
import requests
import argparse
import logging
import subprocess
from tqdm import tqdm
import gdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download-models")

# Add parent directory to path to import from parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_file(url, destination):
    """
    Download a file from a URL to a destination with a progress bar.
    """
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}")
        return
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    logger.info(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_from_gdrive(file_id, destination):
    """
    Download a file from Google Drive to a destination.
    """
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}")
        return
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    logger.info(f"Downloading from Google Drive to {destination}")
    try:
        gdown.download(id=file_id, output=destination, quiet=False)
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {e}")
        logger.error("Try reinstalling gdown with: pip install --upgrade --no-cache-dir gdown")
        sys.exit(1)
        
def download_stable_diffusion():
    """
    Download the Stable Diffusion v1.5 model using the Hugging Face CLI.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
    
    if os.path.exists(sd_dir) and os.listdir(sd_dir):
        logger.info(f"Stable Diffusion model already exists at {sd_dir}")
        return
    
    os.makedirs(sd_dir, exist_ok=True)
    
    logger.info("Downloading Stable Diffusion v1.5 model...")
    
    try:
        # Try using huggingface_hub if available
        try:
            import huggingface_hub
            logger.info("Using huggingface_hub to download the model")
            huggingface_hub.snapshot_download(
                repo_id="runwayml/stable-diffusion-v1-5",
                local_dir=sd_dir,
                local_dir_use_symlinks=False
            )
        except ImportError:
            # Fall back to using the CLI
            logger.info("huggingface_hub not available, using CLI")
            cmd = [
                "python", "-m", "huggingface_hub", "download",
                "--repo-id=runwayml/stable-diffusion-v1-5",
                f"--local-dir={sd_dir}",
                "--local-dir-use-symlinks=False"
            ]
            subprocess.run(cmd, check=True)
        
        logger.info(f"Stable Diffusion model downloaded successfully to {sd_dir}")
    except Exception as e:
        logger.error(f"Error downloading Stable Diffusion model: {e}")
        logger.error("You may need to install the Hugging Face CLI: pip install -U huggingface_hub")
        logger.error("Or log in with: huggingface-cli login")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download required model files for AnimateDiff")
    parser.add_argument("--version", choices=["v1", "v2", "both"], default="both", 
                        help="Which version of the motion module to download (default: both)")
    parser.add_argument("--example-lora", action="store_true", 
                        help="Download example LoRA file")
    parser.add_argument("--skip-sd", action="store_true",
                        help="Skip downloading the Stable Diffusion model")
    parser.add_argument("--force", action="store_true",
                        help="Force download even if files already exist")
    args = parser.parse_args()
    
    # Base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we're running in a container
    in_container = os.path.exists("/.dockerenv")
    if in_container:
        logger.info("Running inside a Docker container")
    
    # Download Stable Diffusion model if not skipped
    if not args.skip_sd:
        download_stable_diffusion()
    else:
        logger.info("Skipping Stable Diffusion model download")
    
    # Motion module files
    motion_module_dir = os.path.join(base_dir, "models", "Motion_Module")
    os.makedirs(motion_module_dir, exist_ok=True)
    
    # Google Drive file IDs for motion modules
    motion_module_v1_id = "1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdl"  # v1
    motion_module_v2_id = "1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu"  # v2
    
    # Download motion modules
    if args.version in ["v1", "both"]:
        download_from_gdrive(
            motion_module_v1_id,
            os.path.join(motion_module_dir, "mm_sd_v15_v1-fp16.safetensors")
        )
    
    if args.version in ["v2", "both"]:
        download_from_gdrive(
            motion_module_v2_id,
            os.path.join(motion_module_dir, "mm_sd_v15_v2-fp16.safetensors")
        )
    
    # Download example LoRA if requested
    if args.example_lora:
        lora_dir = os.path.join(base_dir, "models", "DreamBooth_LoRA")
        os.makedirs(lora_dir, exist_ok=True)
        
        # Example LoRA file ID (replace with actual ID if available)
        example_lora_id = "1EqXxQDG_X7RdVG8bRtGxCi76HEJcQ5FC"  # example_add_detail.safetensors
        
        download_from_gdrive(
            example_lora_id,
            os.path.join(lora_dir, "example_add_detail.safetensors")
        )
    
    # Verify all required files exist
    logger.info("Verifying downloaded files...")
    
    # Check Stable Diffusion model
    sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
    if not os.path.exists(sd_dir) or not os.listdir(sd_dir):
        logger.warning(f"Stable Diffusion model not found at {sd_dir}")
    else:
        logger.info(f"Stable Diffusion model found at {sd_dir}")
    
    # Check motion modules
    if args.version in ["v1", "both"]:
        mm_v1_path = os.path.join(motion_module_dir, "mm_sd_v15_v1-fp16.safetensors")
        if not os.path.exists(mm_v1_path):
            logger.warning(f"Motion module v1 not found at {mm_v1_path}")
        else:
            logger.info(f"Motion module v1 found at {mm_v1_path}")
    
    if args.version in ["v2", "both"]:
        mm_v2_path = os.path.join(motion_module_dir, "mm_sd_v15_v2-fp16.safetensors")
        if not os.path.exists(mm_v2_path):
            logger.warning(f"Motion module v2 not found at {mm_v2_path}")
        else:
            logger.info(f"Motion module v2 found at {mm_v2_path}")
    
    logger.info("Download completed!")

if __name__ == "__main__":
    main()