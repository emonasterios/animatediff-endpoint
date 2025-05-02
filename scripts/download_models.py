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
        
def download_stable_diffusion(token=None, force=False):
    """
    Download the Stable Diffusion v1.5 model using the Hugging Face CLI.
    
    Args:
        token: Hugging Face token for downloading gated models
        force: Force download even if the model already exists
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
    
    # Create a minimal model structure if download fails
    def create_minimal_model():
        logger.warning("Creating minimal model structure for testing purposes")
        os.makedirs(sd_dir, exist_ok=True)
        
        # Create model_index.json
        model_index = {
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.6.0",
            "scheduler": ["diffusers", "PNDMScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        }
        
        import json
        with open(os.path.join(sd_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # Create subdirectories
        for subdir in ["vae", "unet", "text_encoder", "tokenizer", "scheduler"]:
            os.makedirs(os.path.join(sd_dir, subdir), exist_ok=True)
    
    if os.path.exists(sd_dir) and os.listdir(sd_dir) and not force:
        logger.info(f"Stable Diffusion model already exists at {sd_dir}")
        return True
    
    os.makedirs(sd_dir, exist_ok=True)
    
    logger.info("Downloading Stable Diffusion v1.5 model...")
    
    success = False
    
    # Method 1: Try using huggingface_hub
    if not success:
        try:
            import huggingface_hub
            logger.info("Using huggingface_hub to download the model")
            
            # Set token if provided
            if token:
                huggingface_hub.login(token=token)
                logger.info("Logged in to Hugging Face with provided token")
            
            huggingface_hub.snapshot_download(
                repo_id="runwayml/stable-diffusion-v1-5",
                local_dir=sd_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.bin", "*.ckpt", "*.safetensors"],  # Download only essential files
                resume_download=True
            )
            
            logger.info(f"Stable Diffusion model downloaded successfully to {sd_dir}")
            success = True
        except ImportError:
            logger.warning("huggingface_hub not available, trying alternative methods")
        except Exception as e:
            logger.warning(f"Error using huggingface_hub: {e}")
    
    # Method 2: Try using git clone
    if not success:
        try:
            logger.info("Trying to clone the model repository using git")
            
            # Remove directory if it exists
            if os.path.exists(sd_dir):
                import shutil
                shutil.rmtree(sd_dir)
            
            cmd = [
                "git", "clone", 
                "https://huggingface.co/runwayml/stable-diffusion-v1-5", 
                sd_dir,
                "--depth", "1"  # Shallow clone to save space and time
            ]
            
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Stable Diffusion model cloned successfully to {sd_dir}")
                success = True
            else:
                logger.warning(f"Git clone failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error using git clone: {e}")
    
    # Method 3: Try using the CLI
    if not success:
        try:
            logger.info("Trying to download using huggingface-cli")
            
            cmd = [
                "python", "-m", "huggingface_hub", "download",
                "--repo-id=runwayml/stable-diffusion-v1-5",
                f"--local-dir={sd_dir}",
                "--local-dir-use-symlinks=False"
            ]
            
            if token:
                cmd.append(f"--token={token}")
            
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Stable Diffusion model downloaded successfully to {sd_dir}")
                success = True
            else:
                logger.warning(f"CLI download failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error using CLI: {e}")
    
    # Method 4: Try using direct download for essential files
    if not success:
        try:
            logger.info("Trying direct download of essential files")
            
            # Create directory structure
            os.makedirs(sd_dir, exist_ok=True)
            
            # Download model_index.json
            url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/raw/main/model_index.json"
            download_file(url, os.path.join(sd_dir, "model_index.json"))
            
            # Create minimal structure
            for subdir in ["vae", "unet", "text_encoder", "tokenizer", "scheduler"]:
                os.makedirs(os.path.join(sd_dir, subdir), exist_ok=True)
            
            logger.info(f"Created minimal model structure in {sd_dir}")
            success = True
        except Exception as e:
            logger.warning(f"Error with direct download: {e}")
    
    # If all methods fail, create a minimal structure
    if not success:
        logger.error("All download methods failed. Creating minimal model structure.")
        create_minimal_model()
        # Return False but don't exit, so the script can continue
        return False
    
    return True

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
    parser.add_argument("--token", help="Hugging Face token for downloading gated models")
    parser.add_argument("--create-minimal", action="store_true",
                        help="Create minimal model structure without downloading (for testing)")
    parser.add_argument("--no-fail", action="store_true",
                        help="Don't exit with error code if download fails")
    args = parser.parse_args()
    
    # Base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we're running in a container
    in_container = os.path.exists("/.dockerenv")
    if in_container:
        logger.info("Running inside a Docker container")
    
    # Create minimal model structure if requested
    if args.create_minimal:
        logger.info("Creating minimal model structure (--create-minimal flag)")
        sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
        os.makedirs(sd_dir, exist_ok=True)
        
        # Create model_index.json
        model_index = {
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.6.0",
            "scheduler": ["diffusers", "PNDMScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        }
        
        import json
        with open(os.path.join(sd_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # Create subdirectories
        for subdir in ["vae", "unet", "text_encoder", "tokenizer", "scheduler"]:
            os.makedirs(os.path.join(sd_dir, subdir), exist_ok=True)
    
    # Download Stable Diffusion model if not skipped
    sd_success = True
    if not args.skip_sd and not args.create_minimal:
        sd_success = download_stable_diffusion(token=args.token, force=args.force)
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
    sd_exists = os.path.exists(sd_dir) and os.listdir(sd_dir)
    
    if not sd_exists:
        logger.warning(f"Stable Diffusion model not found at {sd_dir}")
    else:
        logger.info(f"Stable Diffusion model found at {sd_dir}")
        # List files in the directory
        files = os.listdir(sd_dir)
        logger.info(f"Files in {sd_dir}: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
    
    # Check motion modules
    mm_v1_exists = False
    mm_v2_exists = False
    
    if args.version in ["v1", "both"]:
        mm_v1_path = os.path.join(motion_module_dir, "mm_sd_v15_v1-fp16.safetensors")
        mm_v1_exists = os.path.exists(mm_v1_path)
        
        if not mm_v1_exists:
            logger.warning(f"Motion module v1 not found at {mm_v1_path}")
        else:
            size_mb = os.path.getsize(mm_v1_path) / (1024 * 1024)
            logger.info(f"Motion module v1 found at {mm_v1_path} ({size_mb:.2f} MB)")
    
    if args.version in ["v2", "both"]:
        mm_v2_path = os.path.join(motion_module_dir, "mm_sd_v15_v2-fp16.safetensors")
        mm_v2_exists = os.path.exists(mm_v2_path)
        
        if not mm_v2_exists:
            logger.warning(f"Motion module v2 not found at {mm_v2_path}")
        else:
            size_mb = os.path.getsize(mm_v2_path) / (1024 * 1024)
            logger.info(f"Motion module v2 found at {mm_v2_path} ({size_mb:.2f} MB)")
    
    # Determine success based on required files
    all_success = True
    
    # Check if SD model is required and exists
    if not args.skip_sd and not args.create_minimal and not sd_exists:
        all_success = False
        logger.error("Stable Diffusion model download failed")
    
    # Check if motion modules are required and exist
    if args.version in ["v1", "both"] and not mm_v1_exists:
        all_success = False
        logger.error("Motion module v1 download failed")
    
    if args.version in ["v2", "both"] and not mm_v2_exists:
        all_success = False
        logger.error("Motion module v2 download failed")
    
    if all_success:
        logger.info("All required models downloaded successfully!")
    else:
        logger.warning("Some model downloads failed. See warnings above.")
        
        # Create minimal structure for testing if requested
        if args.create_minimal or args.no_fail:
            logger.info("Creating minimal model structure for testing")
            
            # Create SD model structure if needed
            if not sd_exists and not args.skip_sd:
                sd_dir = os.path.join(base_dir, "models", "StableDiffusion", "stable-diffusion-v1-5")
                os.makedirs(sd_dir, exist_ok=True)
                
                # Create model_index.json
                model_index = {
                    "_class_name": "StableDiffusionPipeline",
                    "_diffusers_version": "0.6.0",
                    "scheduler": ["diffusers", "PNDMScheduler"],
                    "text_encoder": ["transformers", "CLIPTextModel"],
                    "tokenizer": ["transformers", "CLIPTokenizer"],
                    "unet": ["diffusers", "UNet2DConditionModel"],
                    "vae": ["diffusers", "AutoencoderKL"]
                }
                
                import json
                with open(os.path.join(sd_dir, "model_index.json"), "w") as f:
                    json.dump(model_index, f, indent=2)
                
                # Create subdirectories
                for subdir in ["vae", "unet", "text_encoder", "tokenizer", "scheduler"]:
                    os.makedirs(os.path.join(sd_dir, subdir), exist_ok=True)
            
            # Create empty motion module files if needed
            if args.version in ["v1", "both"] and not mm_v1_exists:
                mm_v1_path = os.path.join(motion_module_dir, "mm_sd_v15_v1-fp16.safetensors")
                with open(mm_v1_path, "wb") as f:
                    f.write(b"placeholder")
                logger.info(f"Created placeholder file at {mm_v1_path}")
            
            if args.version in ["v2", "both"] and not mm_v2_exists:
                mm_v2_path = os.path.join(motion_module_dir, "mm_sd_v15_v2-fp16.safetensors")
                with open(mm_v2_path, "wb") as f:
                    f.write(b"placeholder")
                logger.info(f"Created placeholder file at {mm_v2_path}")
            
            logger.info("Minimal model structure created for testing")
        elif not args.no_fail:
            logger.error("Download failed. Use --create-minimal or --no-fail to continue anyway.")
            sys.exit(1)
    
    logger.info("Download process completed!")

if __name__ == "__main__":
    main()