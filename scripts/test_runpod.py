#!/usr/bin/env python3
"""
Script to test the RunPod endpoint for AnimateDiff.
This script sends a test request to the RunPod endpoint and checks the response.
"""

import os
import sys
import json
import time
import argparse
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test-runpod")

def test_endpoint(endpoint_id, api_key, prompt="A cat walking", steps=5, width=512, height=512):
    """Test the RunPod endpoint with a simple request"""
    logger.info(f"Testing RunPod endpoint {endpoint_id} with prompt: '{prompt}'")
    
    # Prepare the request
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "n_prompt": "bad quality, blurry",
            "guidance_scale": 7.5,
            "seed": 42
        }
    }
    
    # Send the request
    logger.info("Sending request to RunPod endpoint...")
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        logger.info(f"Request submitted successfully. Job ID: {result.get('id')}")
        
        # Check if the request was accepted
        if "id" not in result:
            logger.error(f"Request failed. Response: {result}")
            return False
        
        # Poll for the result
        job_id = result["id"]
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        
        logger.info(f"Polling for result (job ID: {job_id})...")
        max_attempts = 60  # 5 minutes (5s interval)
        for attempt in range(max_attempts):
            time.sleep(5)
            
            try:
                status_response = requests.get(status_url, headers=headers)
                status_response.raise_for_status()
                status = status_response.json()
                
                if status.get("status") == "COMPLETED":
                    logger.info("Job completed successfully!")
                    logger.info(f"Result: {json.dumps(status.get('output', {}), indent=2)}")
                    return True
                elif status.get("status") == "FAILED":
                    logger.error(f"Job failed. Error: {status.get('error')}")
                    return False
                else:
                    logger.info(f"Job status: {status.get('status')} (attempt {attempt + 1}/{max_attempts})")
            except Exception as e:
                logger.error(f"Error checking job status: {e}")
        
        logger.error(f"Job timed out after {max_attempts} attempts")
        return False
    
    except Exception as e:
        logger.error(f"Error sending request: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the RunPod endpoint for AnimateDiff")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--prompt", default="A cat walking", help="Prompt for the test request")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument("--width", type=int, default=512, help="Width of the output video")
    parser.add_argument("--height", type=int, default=512, help="Height of the output video")
    args = parser.parse_args()
    
    logger.info("=== AnimateDiff RunPod Endpoint Tester ===")
    
    # Test the endpoint
    success = test_endpoint(
        args.endpoint_id,
        args.api_key,
        args.prompt,
        args.steps,
        args.width,
        args.height
    )
    
    if success:
        logger.info("=== Test completed successfully! ===")
        return 0
    else:
        logger.error("=== Test failed! ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())