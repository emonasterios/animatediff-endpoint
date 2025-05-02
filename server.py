import os
import sys
import traceback

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import runpod
import base64
import signal

print("==== Server starting ====")
print(f"Python version: {sys.version}")
print(f"CUDA available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error importing torch: {e}")
    traceback.print_exc()

try:
    from inference_util import AnimateDiff, check_data_format
    print("Successfully imported AnimateDiff and check_data_format")
    
    # Initialize AnimateDiff
    print("Initializing AnimateDiff...")
    animatediff = AnimateDiff()
    print("AnimateDiff initialized successfully")
    
except Exception as e:
    print(f"Error initializing AnimateDiff: {e}")
    traceback.print_exc()
    raise

timeout_s = 60 * 5


def encode_data(data_path):
    with open(data_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def handle_timeout(signum, frame):
    # raise an error when timeout, so that the serverless function will be terminated to avoid extra cost
    raise TimeoutError("Request Timeout! Please check the log for more details.")


def text2video(job):
    # set timeout to 5 minutes, should be enough for most cases
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(timeout_s)
        
        print(f"Received job: {job.get('id', 'unknown')}")
        
        # Validate input
        if "input" not in job:
            return {"error": "No input provided in job"}
            
        job_input = job["input"]
        print(f"Raw job input: {job_input}")
        
        try:
            job_input = check_data_format(job_input)
            print(f"Validated job input: {job_input}")
        except ValueError as e:
            print(f"Input validation error: {e}")
            return {"error": f"Input validation failed: {e}"}
        
        print(f"Processing prompt: '{job_input['prompt']}'")
        
        # Run inference
        try:
            save_path = animatediff.inference(
                prompt         = job_input["prompt"],
                steps          = job_input["steps"],
                width          = job_input["width"],
                height         = job_input["height"],
                n_prompt       = job_input["n_prompt"],
                guidance_scale = job_input["guidance_scale"],
                seed           = job_input["seed"],
                base_model     = job_input["base_model"],
                base_loras     = job_input["base_loras"],
                motion_lora    = job_input["motion_lora"],
            )
            print(f"Inference completed successfully. Video saved to: {save_path}")
        except Exception as e:
            print(f"Inference error: {e}")
            traceback.print_exc()
            return {"error": f"Inference failed: {e}"}
        
        # Encode video data
        try:
            video_data = encode_data(save_path)
            print(f"Video encoded successfully: {os.path.basename(save_path)}")
            return {"filename": os.path.basename(save_path), "data": video_data}
        except Exception as e:
            print(f"Error encoding video data: {e}")
            traceback.print_exc()
            return {"error": f"Failed to encode video: {e}"}
            
    except TimeoutError as e:
        print(f"Timeout error: {e}")
        return {"error": f"Request timed out after {timeout_s} seconds: {e}"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return {"error": f"Something went wrong: {e}"}
    finally:
        signal.alarm(0)
        print("Request processing completed")


runpod.serverless.start({"handler": text2video})
