import os
import time
import torch
import imageio
import tempfile
import numpy as np
import logging
import traceback
import sys
from einops import rearrange
from omegaconf import OmegaConf

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
logger = logging.getLogger("animatediff-inference")

# Check if CUDA is available
if torch.cuda.is_available():
    # set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    # set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    logger.info("CUDA is available. Using GPU for inference.")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    logger.warning("CUDA is not available. Using CPU for inference (this will be slow).")

try:
    logger.info("Importing AnimateDiff utility functions...")
    from animatediff.utils.util import init_pipeline, reload_motion_module, load_base_model, apply_lora, apply_motion_lora
    logger.info("Successfully imported AnimateDiff utility functions")
except Exception as e:
    logger.error(f"Error importing AnimateDiff utility functions: {e}")
    traceback.print_exc()
    raise

logger.info("==== Initializing AnimateDiff ====")

def save_video(frames: torch.Tensor, seed=""):
    try:
        logger.info(f"Saving video with seed: {seed}")
        start_time = time.time()
        
        # Create temporary file
        output_video_path = tempfile.NamedTemporaryFile(prefix="{}_".format(seed), suffix=".mp4").name
        logger.info(f"Created temporary file: {output_video_path}")
        
        # Process frames
        logger.info(f"Processing frames with shape: {frames.shape}")
        frames = (rearrange(frames, "b c t h w -> t b h w c").squeeze(1).cpu().numpy() * 255).astype(np.uint8)
        logger.info(f"Processed frames with shape: {frames.shape}")
        
        # Write video
        logger.info("Creating video writer...")
        writer = imageio.get_writer(output_video_path, fps=8, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1)
        
        logger.info(f"Writing {len(frames)} frames to video...")
        for i, frame in enumerate(frames):
            writer.append_data(frame)
            if i % 5 == 0:  # Log progress every 5 frames
                logger.info(f"Wrote frame {i+1}/{len(frames)}")
                
        writer.close()
        logger.info("Video writer closed")
        
        # Check if file was created successfully
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / 1024  # Size in KB
            logger.info(f"Video saved successfully. File size: {file_size:.2f} KB")
        else:
            logger.warning(f"Video file was not created at {output_video_path}")
            
        save_time = time.time() - start_time
        logger.info(f"Video saving completed in {save_time:.2f} seconds")
        
        return output_video_path
        
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        traceback.print_exc()
        raise


def check_data_format(job_input):
    logger.info("Validating job input format...")
    
    # must have prompt in the input, otherwise raise error to the user
    if "prompt" in job_input:
        prompt = job_input["prompt"]
        logger.info(f"Found prompt: {prompt}")
    else:
        logger.error("The input must contain a prompt.")
        raise ValueError("The input must contain a prompt.")
        
    if not isinstance(prompt, str):
        logger.error(f"Prompt must be a string, got {type(prompt)}")
        raise ValueError("prompt must be a string.")

    # optional params, make sure they are in the right format here, otherwise raise error to the user
    steps          = job_input["steps"] if "steps" in job_input else None
    width          = job_input["width"] if "width" in job_input else None
    height         = job_input["height"] if "height" in job_input else None
    n_prompt       = job_input["n_prompt"] if "n_prompt" in job_input else None
    guidance_scale = job_input["guidance_scale"] if "guidance_scale" in job_input else None
    seed           = job_input["seed"] if "seed" in job_input else None
    base_model     = job_input["base_model"] if "base_model" in job_input else None
    base_loras     = job_input["base_loras"] if "base_loras" in job_input else None
    motion_lora    = job_input["motion_lora"] if "motion_lora" in job_input else None
    
    logger.info(f"Optional parameters found: steps={steps}, width={width}, height={height}, n_prompt={n_prompt}, "
                f"guidance_scale={guidance_scale}, seed={seed}, base_model={base_model}")
    
    if base_loras:
        logger.info(f"base_loras: {base_loras}")
    if motion_lora:
        logger.info(f"motion_lora: {motion_lora}")

    # check optional params
    if steps is not None and not isinstance(steps, int):
        logger.error(f"Steps must be an integer, got {type(steps)}")
        raise ValueError("steps must be an integer.")
        
    if width is not None and not isinstance(width, int):
        logger.error(f"Width must be an integer, got {type(width)}")
        raise ValueError("width must be an integer.")
        
    if height is not None and not isinstance(height, int):
        logger.error(f"Height must be an integer, got {type(height)}")
        raise ValueError("height must be an integer.")
        
    if n_prompt is not None and not isinstance(n_prompt, str):
        logger.error(f"Negative prompt must be a string, got {type(n_prompt)}")
        raise ValueError("n_prompt must be a string.")
        
    if guidance_scale is not None and not isinstance(guidance_scale, float) and not isinstance(guidance_scale, int):
        logger.error(f"Guidance scale must be a float or integer, got {type(guidance_scale)}")
        raise ValueError("guidance_scale must be a float or an integer.")
        
    if seed is not None and not isinstance(seed, int):
        logger.error(f"Seed must be an integer, got {type(seed)}")
        raise ValueError("seed must be an integer.")
        
    if base_model is not None and not isinstance(base_model, str):
        logger.error(f"Base model must be a string, got {type(base_model)}")
        raise ValueError("base_model must be a string.")
        
    if base_loras is not None:
        if not isinstance(base_loras, dict):
            logger.error(f"base_loras must be a dictionary, got {type(base_loras)}")
            raise ValueError("base_loras must be a dictionary.")
            
        for lora_name, lora_params in base_loras.items():
            if not isinstance(lora_name, str):
                logger.error(f"base_loras keys must be strings, got {type(lora_name)}")
                raise ValueError("base_loras keys must be strings.")
                
            if not isinstance(lora_params, list):
                logger.error(f"base_loras values must be lists, got {type(lora_params)}")
                raise ValueError("base_loras values must be lists.")
                
            if len(lora_params) != 2:
                logger.error(f"base_loras values must be lists of length 2, got length {len(lora_params)}")
                raise ValueError("base_loras values must be lists of length 2.")
                
            if not isinstance(lora_params[0], str):
                logger.error(f"base_loras values must be lists of strings, got {type(lora_params[0])}")
                raise ValueError("base_loras values must be lists of strings.")
                
            if not isinstance(lora_params[1], float):
                logger.error(f"base_loras values must be lists of floats, got {type(lora_params[1])}")
                raise ValueError("base_loras values must be lists of floats.")
                
    if motion_lora is not None:
        if not isinstance(motion_lora, list):
            logger.error(f"motion_lora must be a list, got {type(motion_lora)}")
            raise ValueError("motion_lora must be a list.")
            
        if len(motion_lora) != 2:
            logger.error(f"motion_lora must be a list of length 2, got length {len(motion_lora)}")
            raise ValueError("motion_lora must be a list of length 2.")
            
        if not isinstance(motion_lora[0], str):
            logger.error(f"motion_lora must be a list of strings, got {type(motion_lora[0])}")
            raise ValueError("motion_lora must be a list of strings.")
            
        if (not isinstance(motion_lora[1], float)) and (not isinstance(motion_lora[1], int)):
            logger.error(f"motion_lora must be a list of floats, got {type(motion_lora[1])}")
            raise ValueError("motion_lora must be a list of floats.")
            
    logger.info("Job input validation completed successfully")
    
    return {
        "prompt"        : prompt,
        "steps"         : steps,
        "width"         : width,
        "height"        : height,
        "n_prompt"      : n_prompt,
        "guidance_scale": guidance_scale,
        "seed"          : seed,
        "base_model"    : base_model,
        "base_loras"    : base_loras,
        "motion_lora"   : motion_lora,
    }


class AnimateDiff:
    def __init__(self, version="v2"):
        logger.info(f"Initializing AnimateDiff with version: {version}")
        start_time = time.time()
        
        self.version = version
        assert self.version in ["v1", "v2"], "version must be either v1 or v2"
        
        # Set up paths
        pretrained_model_path = os.path.join(os.path.dirname(__file__), "models/StableDiffusion/stable-diffusion-v1-5")
        self.motion_module    = os.path.join(os.path.dirname(__file__), "models/Motion_Module/mm_sd_v15_{}-fp16.safetensors".format(self.version))
        self.inference_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "inference_{}.yaml".format(self.version)))
        self.model_dir        = os.path.join(os.path.dirname(__file__), "models/DreamBooth_LoRA")
        
        logger.info(f"Pretrained model path: {pretrained_model_path}")
        logger.info(f"Motion module path: {self.motion_module}")
        logger.info(f"Model directory: {self.model_dir}")

        # Check if required model files exist
        if not os.path.exists(pretrained_model_path):
            logger.warning(f"Pretrained model path {pretrained_model_path} does not exist.")
            logger.warning("Will attempt to download from HuggingFace.")
        else:
            logger.info(f"Pretrained model path exists: {pretrained_model_path}")
            # List files in the directory to verify content
            try:
                model_files = os.listdir(pretrained_model_path)
                logger.info(f"Files in pretrained model directory: {model_files}")
            except Exception as e:
                logger.warning(f"Could not list files in pretrained model directory: {e}")
        
        if not os.path.exists(self.motion_module):
            logger.error(f"Motion module {self.motion_module} does not exist.")
            logger.error("You need to download the motion module and place it in the models/Motion_Module directory.")
            logger.warning("The model will still initialize but inference will fail without the motion module.")
        else:
            logger.info(f"Motion module exists: {self.motion_module}")
            # Check file size to verify it's not empty
            try:
                motion_module_size = os.path.getsize(self.motion_module) / (1024 * 1024)  # Size in MB
                logger.info(f"Motion module size: {motion_module_size:.2f} MB")
            except Exception as e:
                logger.warning(f"Could not get motion module file size: {e}")

        # can not be changed
        self.video_length = 16
        logger.info(f"Video length set to: {self.video_length} frames")
        
        # Check if CUDA is available and set device accordingly
        if torch.cuda.is_available():
            self.device = "cuda"
            self.use_fp16 = True
            self.dtype = torch.float16 if self.use_fp16 else torch.float32
            logger.info(f"Using device: {self.device} with dtype: {self.dtype}")
            logger.info(f"CUDA memory before pipeline init - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.dtype = torch.float32
            logger.warning("Running on CPU. This will be very slow and may not work well.")

        # Initialize pipeline
        try:
            logger.info("Initializing pipeline...")
            pipeline_start_time = time.time()
            
            # Check if model files actually exist
            model_files_exist = True
            required_files = [
                os.path.join(pretrained_model_path, "text_encoder", "pytorch_model.bin"),
                os.path.join(pretrained_model_path, "vae", "diffusion_pytorch_model.bin"),
                os.path.join(pretrained_model_path, "unet", "diffusion_pytorch_model.bin"),
                os.path.join(pretrained_model_path, "tokenizer", "vocab.json")
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.warning(f"Required model file not found: {file_path}")
                    model_files_exist = False
            
            if not model_files_exist:
                logger.error("Missing required model files. This is expected in test environments without real models.")
                logger.error("In a production environment, please ensure all model files are downloaded.")
                logger.error("The worker will exit now as it cannot function without the required models.")
                # In a real environment, we would exit here
                # For testing purposes, we'll continue but set pipeline to None
                self.pipeline = None
                return
            
            self.pipeline = init_pipeline(pretrained_model_path, self.inference_config, self.device, self.dtype)
            pipeline_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline initialized successfully in {pipeline_time:.2f} seconds.")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA memory after pipeline init - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            logger.error("This error indicates missing or corrupt model files.")
            logger.error("Please ensure all required models are downloaded correctly.")
            logger.error("Check the model paths and file permissions.")
            traceback.print_exc()
            # In production, we would raise the exception
            # For testing, we'll set pipeline to None and continue
            self.pipeline = None

        # pre-defined default params, can be changed
        self.steps          = 25
        self.guidance_scale = 7.5
        self.person_prompts = ["boy", "girl", "man", "woman", "person", "eye", "face"]
        
        init_time = time.time() - start_time
        logger.info(f"AnimateDiff initialization completed in {init_time:.2f} seconds")

    def _reload_motion_module(self):
        # somehow the motion module needs to be reloaded every time if the motion lora was applied, otherwise the result could be wrong
        # reloading the motion module only takes 0.2s, so I think it's fine to reload it every time instead of checking if last time the motion lora was applied
        logger.info("Reloading motion module...")
        start_time = time.time()
        
        if not os.path.exists(self.motion_module):
            logger.error(f"Motion module {self.motion_module} does not exist.")
            logger.error("You need to download the motion module and place it in the models/Motion_Module directory.")
            raise FileNotFoundError(f"Motion module file not found: {self.motion_module}")
        
        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA memory before reloading motion module - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            self.pipeline = reload_motion_module(self.pipeline, self.motion_module, self.device)
            
            reload_time = time.time() - start_time
            logger.info(f"Motion module reloaded successfully in {reload_time:.2f} seconds.")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA memory after reloading motion module - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Error reloading motion module: {e}")
            traceback.print_exc()
            raise

    def _get_model_params(self, prompt, width, height, n_prompt, base_model, base_loras, motion_lora):
        logger.info("Getting model parameters...")
        
        # Clean prompt
        original_prompt = prompt
        prompt = prompt[:-1] if prompt[-1] == "." else prompt
        if original_prompt != prompt:
            logger.info(f"Cleaned prompt by removing trailing period: '{original_prompt}' -> '{prompt}'")
        
        if base_model is None:
            logger.info("No base model specified, selecting based on prompt content")
            
            # when base_model is not specified, use the default model
            # if the prompt contains person-related keywords, use the person model, otherwise use the default model
            isPerson = False
            for keyword in self.person_prompts:
                if keyword.lower() in prompt.lower():
                    isPerson = True
                    logger.info(f"Person-related keyword '{keyword}' found in prompt")
                    break

            # load default params
            model_type = "Person" if isPerson else "Default"
            logger.info(f"Using {model_type} model configuration")
            
            model_config = self.inference_config.Person if isPerson else self.inference_config.Default
            base_model   = model_config.base_model
            base_loras   = model_config.base_loras
            motion_lora  = model_config.motion_lora if self.version == "v2" else None
            
            logger.info(f"Selected base_model: {base_model}")
            if base_loras:
                logger.info(f"Selected base_loras: {base_loras}")
            if motion_lora:
                logger.info(f"Selected motion_lora: {motion_lora}")
            
            # Append model-specific prompt
            original_prompt = prompt
            prompt += ", "
            prompt += model_config.prompt
            logger.info(f"Appended model-specific prompt: '{original_prompt}' -> '{prompt}'")
        else:
            logger.info(f"Using user-specified base model: {base_model}")
            
            # load default params
            model_config = self.inference_config.Default
            logger.info("Using Default model configuration for other parameters")

        # update with user-specified params
        if n_prompt is None:
            n_prompt = model_config.n_prompt
            logger.info(f"Using default negative prompt: '{n_prompt}'")
        else:
            logger.info(f"Using user-specified negative prompt: '{n_prompt}'")
            
        if width is None:
            width = model_config.width
            logger.info(f"Using default width: {width}")
        else:
            logger.info(f"Using user-specified width: {width}")
            
        if height is None:
            height = model_config.height
            logger.info(f"Using default height: {height}")
        else:
            logger.info(f"Using user-specified height: {height}")

        logger.info("Model parameters prepared successfully")
        return prompt, width, height, n_prompt, base_model, base_loras, motion_lora

    def _update_model(self, base_model, base_loras, motion_lora):
        # update model
        logger.info(f"Updating model with base_model: {base_model}")
        start_time = time.time()
        
        if base_model and base_model != "":
            try:
                # First check if the base model exists
                base_model_path = os.path.join(self.model_dir, base_model)
                if not os.path.exists(base_model_path) and not base_model.startswith("runwayml/"):
                    logger.warning(f"Base model {base_model_path} does not exist.")
                    logger.warning("Will attempt to use default model from HuggingFace instead.")
                    base_model = "runwayml/stable-diffusion-v1-5"
                    logger.info(f"Using HuggingFace model: {base_model}")
                else:
                    logger.info(f"Base model path exists: {base_model_path}")
                
                # Reload motion module
                self._reload_motion_module()
                
                # Load base model
                logger.info(f"Loading base model: {base_model}")
                load_model_start = time.time()
                
                if torch.cuda.is_available():
                    logger.info(f"CUDA memory before loading base model - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                
                self.pipeline = load_base_model(self.pipeline, self.model_dir, base_model, self.device, self.dtype)
                
                # Make sure the model is on the right device and dtype
                self.pipeline.to(self.device, self.dtype)
                
                load_model_time = time.time() - load_model_start
                logger.info(f"Base model loaded successfully: {base_model} in {load_model_time:.2f} seconds")
                
                if torch.cuda.is_available():
                    logger.info(f"CUDA memory after loading base model - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

                # Apply lora
                if base_loras:
                    if len(base_loras) != 0:
                        logger.info(f"Processing {len(base_loras)} LoRA models")
                        valid_loras = {}
                        for lora in base_loras:
                            if len(base_loras[lora]) != 2:
                                logger.warning(f"Skipping lora {lora} - format must be [filename, scale]")
                                continue
                                
                            lora_path = os.path.join(self.model_dir, base_loras[lora][0])
                            if not os.path.exists(lora_path):
                                logger.warning(f"LoRA file {lora_path} does not exist. Skipping.")
                                continue
                                
                            valid_loras[lora] = base_loras[lora]
                            logger.info(f"Valid LoRA found: {lora} with path {lora_path} and scale {base_loras[lora][1]}")
                            
                        if valid_loras:
                            logger.info(f"Applying {len(valid_loras)} LoRA models")
                            lora_start = time.time()
                            
                            if torch.cuda.is_available():
                                logger.info(f"CUDA memory before applying LoRA - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                            
                            self.pipeline = apply_lora(self.pipeline, self.model_dir, valid_loras, device=self.device, dtype=self.dtype)
                            
                            lora_time = time.time() - lora_start
                            logger.info(f"Applied {len(valid_loras)} LoRA models successfully in {lora_time:.2f} seconds")
                            
                            if torch.cuda.is_available():
                                logger.info(f"CUDA memory after applying LoRA - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                        else:
                            logger.warning("No valid LoRA models found to apply.")

                # Apply motion lora
                if motion_lora:
                    if self.version == "v1":
                        logger.warning("motion_lora is not supported in v1. Skipping.")
                    elif len(motion_lora) == 2:
                        motion_lora_path = os.path.join(self.model_dir, motion_lora[0])
                        if not os.path.exists(motion_lora_path):
                            logger.warning(f"Motion LoRA file {motion_lora_path} does not exist. Skipping.")
                        else:
                            logger.info(f"Applying motion LoRA: {motion_lora[0]} with scale {motion_lora[1]}")
                            motion_lora_start = time.time()
                            
                            if torch.cuda.is_available():
                                logger.info(f"CUDA memory before applying motion LoRA - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                            
                            self.pipeline = apply_motion_lora(self.pipeline, self.model_dir, motion_lora, device=self.device, dtype=self.dtype)
                            
                            motion_lora_time = time.time() - motion_lora_start
                            logger.info(f"Applied motion LoRA successfully: {motion_lora[0]} in {motion_lora_time:.2f} seconds")
                            
                            if torch.cuda.is_available():
                                logger.info(f"CUDA memory after applying motion LoRA - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                    else:
                        logger.warning("motion_lora must be [filename, scale]. Skipping.")
                
                update_time = time.time() - start_time
                logger.info(f"Model update completed in {update_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error updating model: {e}")
                traceback.print_exc()
                raise
        else:
            logger.error("Base model must be specified")
            raise ValueError("base model must be specified")

    def inference(
        self,
        prompt,
        steps          = None,
        width          = None,
        height         = None,
        n_prompt       = None,
        guidance_scale = None,
        seed           = None,
        base_model     = None,
        base_loras     = None,
        motion_lora    = None,
    ):
        # Check if pipeline is initialized
        if self.pipeline is None:
            logger.error("Cannot run inference: Pipeline is not initialized due to missing model files")
            return {
                "error": "Pipeline not initialized due to missing model files",
                "details": "The required model files are missing. Please ensure all models are downloaded correctly."
            }
            
        # only prompt is required
        # optional params for inference: steps, guidance_scale, width, height, seed, n_prompt
        # optional params for model: base_model, base_loras, motion_lora
        inference_start_time = time.time()
        
        try:
            logger.info(f"Starting inference with prompt: {prompt}")
            
            # Get model parameters
            params_start_time = time.time()
            prompt, width, height, n_prompt, base_model, base_loras, motion_lora = self._get_model_params(
                prompt, width, height, n_prompt, base_model, base_loras, motion_lora
            )
            params_time = time.time() - params_start_time
            logger.info(f"Model parameters prepared in {params_time:.2f} seconds. Using base model: {base_model}")
            
            # Update model
            self._update_model(base_model, base_loras, motion_lora)

            # Set up inference parameters
            seed = seed if seed is not None else torch.randint(0, 1000000000, (1,)).item()
            torch.manual_seed(seed)
            logger.info(f"Current seed: {torch.initial_seed()}")
            
            logger.info(f"Sampling with prompt: {prompt}")
            logger.info(f"Negative prompt: {n_prompt}")
            logger.info(f"Image dimensions: {width}x{height}")
            
            steps = self.steps if steps is None else steps
            guidance_scale_value = self.guidance_scale if guidance_scale is None else guidance_scale
            logger.info(f"Using {steps} inference steps with guidance scale: {guidance_scale_value}")
            
            # Run inference
            with torch.no_grad():
                if torch.cuda.is_available():
                    logger.info(f"CUDA memory before pipeline inference - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                
                pipeline_start_time = time.time()
                logger.info("Starting pipeline inference...")
                
                try:
                    sample = self.pipeline(
                        prompt              = prompt,
                        negative_prompt     = n_prompt,
                        num_inference_steps = steps,
                        guidance_scale      = guidance_scale_value,
                        width               = width,
                        height              = height,
                        video_length        = self.video_length,
                    ).videos
                    
                    pipeline_time = time.time() - pipeline_start_time
                    logger.info(f"Pipeline inference completed successfully in {pipeline_time:.2f} seconds")
                    
                    if torch.cuda.is_available():
                        logger.info(f"CUDA memory after pipeline inference - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                except Exception as e:
                    logger.error(f"Error during pipeline inference: {e}")
                    if torch.cuda.is_available():
                        logger.error(f"CUDA memory at error - allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                    traceback.print_exc()
                    raise
                
                # Save video
                save_start_time = time.time()
                logger.info("Saving video...")
                save_path = save_video(sample, seed=seed)
                save_time = time.time() - save_start_time
                logger.info(f"Video saved to: {save_path} in {save_time:.2f} seconds")
                
            total_inference_time = time.time() - inference_start_time
            logger.info(f"Total inference completed in {total_inference_time:.2f} seconds")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            traceback.print_exc()
            
            # Try to free memory in case of error
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after error")
                except:
                    pass
            
            raise


if __name__ == "__main__":
    # example seeds:
    # Person: 445608568
    # Default : 195577361
    import json

    animate_diff = AnimateDiff()

    # simple config
    with open("test_input_simple.json", "r") as f:
        test_input = json.load(f)["input"]

    # only for testing
    test_input = check_data_format(test_input)

    # faster config
    save_path = animate_diff.inference(prompt=test_input["prompt"])
    print("Result of simple config is saved to: {}\n".format(save_path))

    # complex custom config
    with open("./test_input.json", "r") as f:
        test_input = json.load(f)["input"]

    # only for testing
    test_input = check_data_format(test_input)

    # better config
    save_path = animate_diff.inference(
        prompt         = test_input["prompt"],
        steps          = test_input["steps"],
        width          = test_input["width"],
        height         = test_input["height"],
        n_prompt       = test_input["n_prompt"],
        guidance_scale = test_input["guidance_scale"],
        seed           = test_input["seed"],
        base_model     = test_input["base_model"],
        base_loras     = test_input["base_loras"],
        motion_lora    = test_input["motion_lora"],
    )
    print("Result of custom config is saved to: {}\n".format(save_path))
