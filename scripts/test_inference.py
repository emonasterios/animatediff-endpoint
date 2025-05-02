#!/usr/bin/env python3
"""
Script to run a simple test inference with AnimateDiff.
This can help diagnose issues with the RunPod worker.
"""

import os
import sys
import time
import traceback
import logging
import json
import gc

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test-inference")

def run_test_inference():
    """Run a simple test inference"""
    try:
        import torch
        from inference_util import AnimateDiff, check_data_format
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA is available")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB")
        else:
            logger.warning("CUDA is not available, inference will be slow")
        
        # Initialize AnimateDiff
        logger.info("Initializing AnimateDiff...")
        start_time = time.time()
        animatediff = AnimateDiff()
        init_time = time.time() - start_time
        logger.info(f"AnimateDiff initialized in {init_time:.2f} seconds")
        
        # Load test input
        test_input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_input_simple.json")
        logger.info(f"Loading test input from {test_input_path}")
        
        if not os.path.exists(test_input_path):
            logger.error(f"Test input file not found: {test_input_path}")
            return False
        
        with open(test_input_path, "r") as f:
            test_input = json.load(f)["input"]
        
        logger.info(f"Test input: {test_input}")
        
        # Validate input
        logger.info("Validating input...")
        validated_input = check_data_format(test_input)
        logger.info(f"Validated input: {validated_input}")
        
        # Run inference
        logger.info("Running inference...")
        inference_start = time.time()
        
        try:
            save_path = animatediff.inference(
                prompt=validated_input["prompt"],
                steps=validated_input["steps"],
                width=validated_input["width"],
                height=validated_input["height"],
                n_prompt=validated_input["n_prompt"],
                guidance_scale=validated_input["guidance_scale"],
                seed=validated_input["seed"],
                base_model=validated_input["base_model"],
                base_loras=validated_input["base_loras"],
                motion_lora=validated_input["motion_lora"],
            )
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            logger.info(f"Output saved to: {save_path}")
            
            # Check if file exists
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / 1024  # Size in KB
                logger.info(f"Output file size: {file_size:.2f} KB")
            else:
                logger.warning(f"Output file not found: {save_path}")
            
            # Clean up
            try:
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memory cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up memory: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"Error in test inference: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("=== AnimateDiff Test Inference ===")
    
    success = run_test_inference()
    
    if success:
        logger.info("Test inference completed successfully")
        return 0
    else:
        logger.error("Test inference failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())