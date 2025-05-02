import os
import sys
import traceback
import logging
import time
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("animatediff-server")

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"

logger.info("==== Server starting ====")
logger.info(f"Python version: {sys.version}")
logger.info(f"CUDA available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Import libraries with detailed error handling
try:
    import runpod
    logger.info("Successfully imported runpod")
except Exception as e:
    logger.error(f"Error importing runpod: {e}")
    traceback.print_exc()
    raise

try:
    import base64
    import signal
    logger.info("Successfully imported base64 and signal")
except Exception as e:
    logger.error(f"Error importing base64 or signal: {e}")
    traceback.print_exc()
    raise

try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
except Exception as e:
    logger.error(f"Error importing torch: {e}")
    traceback.print_exc()
    raise

try:
    from inference_util import AnimateDiff, check_data_format
    logger.info("Successfully imported AnimateDiff and check_data_format")
    
    # Initialize AnimateDiff
    logger.info("Initializing AnimateDiff...")
    start_time = time.time()
    animatediff = AnimateDiff()
    logger.info(f"AnimateDiff initialized successfully in {time.time() - start_time:.2f} seconds")
    
    # Check GPU memory after initialization
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated after init: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA memory reserved after init: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
except Exception as e:
    logger.error(f"Error initializing AnimateDiff: {e}")
    traceback.print_exc()
    raise

timeout_s = 60 * 5


def encode_data(data_path):
    try:
        with open(data_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        logger.info(f"Successfully encoded file: {data_path} (size: {len(data)} bytes)")
        return encoded
    except Exception as e:
        logger.error(f"Error encoding data from {data_path}: {e}")
        raise


def handle_timeout(signum, frame):
    # raise an error when timeout, so that the serverless function will be terminated to avoid extra cost
    logger.error("Request timeout triggered!")
    raise TimeoutError("Request Timeout! Please check the log for more details.")


def text2video(job):
    # set timeout to 5 minutes, should be enough for most cases
    job_id = job.get('id', 'unknown')
    start_time = time.time()
    
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(timeout_s)
        
        logger.info(f"Received job: {job_id}")
        logger.info(f"Current memory usage - CUDA allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Validate input
        if "input" not in job:
            logger.error(f"Job {job_id}: No input provided")
            return {"error": "No input provided in job"}
            
        job_input = job["input"]
        logger.info(f"Job {job_id}: Raw job input: {job_input}")
        
        try:
            job_input = check_data_format(job_input)
            logger.info(f"Job {job_id}: Validated job input: {job_input}")
        except ValueError as e:
            logger.error(f"Job {job_id}: Input validation error: {e}")
            return {"error": f"Input validation failed: {e}"}
        
        logger.info(f"Job {job_id}: Processing prompt: '{job_input['prompt']}'")
        
        # Run inference
        try:
            # Log system info before inference
            if torch.cuda.is_available():
                logger.info(f"Job {job_id}: Pre-inference CUDA memory - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            inference_start = time.time()
            logger.info(f"Job {job_id}: Starting inference...")
            
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
            
            inference_time = time.time() - inference_start
            logger.info(f"Job {job_id}: Inference completed in {inference_time:.2f} seconds. Video saved to: {save_path}")
            
            # Log system info after inference
            if torch.cuda.is_available():
                logger.info(f"Job {job_id}: Post-inference CUDA memory - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                
            # Try to free up some memory
            torch.cuda.empty_cache()
            gc.collect()
            
            if torch.cuda.is_available():
                logger.info(f"Job {job_id}: After cleanup CUDA memory - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                
        except Exception as e:
            logger.error(f"Job {job_id}: Inference error: {e}")
            traceback.print_exc()
            return {"error": f"Inference failed: {e}"}
        
        # Encode video data
        try:
            logger.info(f"Job {job_id}: Encoding video data from {save_path}")
            video_data = encode_data(save_path)
            logger.info(f"Job {job_id}: Video encoded successfully: {os.path.basename(save_path)}")
            
            # Try to remove the temporary file to free up disk space
            try:
                os.remove(save_path)
                logger.info(f"Job {job_id}: Removed temporary file: {save_path}")
            except Exception as e:
                logger.warning(f"Job {job_id}: Failed to remove temporary file {save_path}: {e}")
                
            return {"filename": os.path.basename(save_path), "data": video_data}
        except Exception as e:
            logger.error(f"Job {job_id}: Error encoding video data: {e}")
            traceback.print_exc()
            return {"error": f"Failed to encode video: {e}"}
            
    except TimeoutError as e:
        logger.error(f"Job {job_id}: Timeout error: {e}")
        return {"error": f"Request timed out after {timeout_s} seconds: {e}"}
    except Exception as e:
        logger.error(f"Job {job_id}: Unexpected error: {e}")
        traceback.print_exc()
        return {"error": f"Something went wrong: {e}"}
    finally:
        signal.alarm(0)
        total_time = time.time() - start_time
        logger.info(f"Job {job_id}: Request processing completed in {total_time:.2f} seconds")


# Add a wrapper to catch any errors during runpod initialization
def start_server():
    try:
        logger.info("Starting RunPod serverless handler")
        runpod.serverless.start({"handler": text2video})
    except Exception as e:
        logger.error(f"Error starting RunPod serverless handler: {e}")
        traceback.print_exc()
        # Re-raise the exception to ensure the process exits with an error code
        # This will help identify issues in the RunPod logs
        raise

# Start the server
if __name__ == "__main__":
    try:
        start_server()
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        traceback.print_exc()
        sys.exit(1)  # Exit with error code 1 to indicate failure
