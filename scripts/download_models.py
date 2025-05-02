#!/usr/bin/env python3
"""
Script to download the required model files for AnimateDiff.
"""

import os
import sys
import requests
import argparse
from tqdm import tqdm
import gdown

# Add parent directory to path to import from parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_file(url, destination):
    """
    Download a file from a URL to a destination with a progress bar.
    """
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading {url} to {destination}")
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
        print(f"File already exists: {destination}")
        return
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading from Google Drive to {destination}")
    try:
        gdown.download(id=file_id, output=destination, quiet=False)
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        print("Try reinstalling gdown with: pip install --upgrade --no-cache-dir gdown")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download required model files for AnimateDiff")
    parser.add_argument("--version", choices=["v1", "v2", "both"], default="both", 
                        help="Which version of the motion module to download (default: both)")
    parser.add_argument("--example-lora", action="store_true", 
                        help="Download example LoRA file")
    args = parser.parse_args()
    
    # Base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
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
    
    print("Download completed!")

if __name__ == "__main__":
    main()