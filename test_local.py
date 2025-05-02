#!/usr/bin/env python3
"""
Script para probar localmente el servidor de AnimateDiff sin necesidad de RunPod.
"""

import json
import sys
import os
import time

# Add the current directory to the path so we can import the server module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the server module
import server

# Create a test job
test_job = {
    "id": "test-job-123",
    "input": {
        "prompt": "A beautiful sunset over the ocean",
        "steps": 20,
        "width": 512,
        "height": 512,
        "n_prompt": "low quality, bad anatomy, worst quality",
        "guidance_scale": 7.5,
        "seed": 42,
        "base_model": "stable-diffusion-v1-5",
        "base_loras": {},  # Debe ser un diccionario, no una lista
        "motion_lora": None
    }
}

# Run the job
print("Running test job...")
start_time = time.time()
result = server.text2video(test_job)
end_time = time.time()

# Print the result
print(f"Job completed in {end_time - start_time:.2f} seconds")
print("Result:")
print(json.dumps(result, indent=2))