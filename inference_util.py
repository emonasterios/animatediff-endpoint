import os
import time
import torch
import imageio
import tempfile
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

# Check if CUDA is available
if torch.cuda.is_available():
    # set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    # set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    print("CUDA is available. Using GPU for inference.")
else:
    print("CUDA is not available. Using CPU for inference (this will be slow).")

from animatediff.utils.util import init_pipeline, reload_motion_module, load_base_model, apply_lora, apply_motion_lora

print("==== Initializing AnimateDiff ====")

def save_video(frames: torch.Tensor, seed=""):
    # save seed to the fil e name, for reproducibility
    output_video_path = tempfile.NamedTemporaryFile(prefix="{}_".format(seed), suffix=".mp4").name
    frames = (rearrange(frames, "b c t h w -> t b h w c").squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    writer = imageio.get_writer(output_video_path, fps=8, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return output_video_path


def check_data_format(job_input):
    # must have prompt in the input, otherwise raise error to the user
    if "prompt" in job_input:
        prompt = job_input["prompt"]
    else:
        raise ValueError("The input must contain a prompt.")
    if not isinstance(prompt, str):
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

    # check optional params
    if steps is not None and not isinstance(steps, int):
        raise ValueError("steps must be an integer.")
    if width is not None and not isinstance(width, int):
        raise ValueError("width must be an integer.")
    if height is not None and not isinstance(height, int):
        raise ValueError("height must be an integer.")
    if n_prompt is not None and not isinstance(n_prompt, str):
        raise ValueError("n_prompt must be a string.")
    if guidance_scale is not None and not isinstance(guidance_scale, float) and not isinstance(guidance_scale, int):
        raise ValueError("guidance_scale must be a float or an integer.")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError("base_model must be a string.")
    if base_loras is not None:
        if not isinstance(base_loras, dict):
            raise ValueError("base_loras must be a dictionary.")
        for lora_name, lora_params in base_loras.items():
            if not isinstance(lora_name, str):
                raise ValueError("base_loras keys must be strings.")
            if not isinstance(lora_params, list):
                raise ValueError("base_loras values must be lists.")
            if len(lora_params) != 2:
                raise ValueError("base_loras values must be lists of length 2.")
            if not isinstance(lora_params[0], str):
                raise ValueError("base_loras values must be lists of strings.")
            if not isinstance(lora_params[1], float):
                raise ValueError("base_loras values must be lists of floats.")
    if motion_lora is not None:
        if not isinstance(motion_lora, list):
            raise ValueError("motion_lora must be a list.")
        if len(motion_lora) != 2:
            raise ValueError("motion_lora must be a list of length 2.")
        if not isinstance(motion_lora[0], str):
            raise ValueError("motion_lora must be a list of strings.")
        if (not isinstance(motion_lora[1], float)) and (not isinstance(motion_lora[1], int)):
            raise ValueError("motion_lora must be a list of floats.")
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
        self.version = version
        assert self.version in ["v1", "v2"], "version must be either v1 or v2"
        pretrained_model_path = os.path.join(os.path.dirname(__file__), "models/StableDiffusion/stable-diffusion-v1-5")
        self.motion_module    = os.path.join(os.path.dirname(__file__), "models/Motion_Module/mm_sd_v15_{}-fp16.safetensors".format(self.version))
        self.inference_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "inference_{}.yaml".format(self.version)))
        self.model_dir        = os.path.join(os.path.dirname(__file__), "models/DreamBooth_LoRA")

        # Check if required model files exist
        if not os.path.exists(pretrained_model_path):
            print(f"WARNING: Pretrained model path {pretrained_model_path} does not exist.")
            print("Will attempt to download from HuggingFace.")
        
        if not os.path.exists(self.motion_module):
            print(f"WARNING: Motion module {self.motion_module} does not exist.")
            print("You need to download the motion module and place it in the models/Motion_Module directory.")
            print("The model will still initialize but inference will fail without the motion module.")

        # can not be changed
        self.video_length = 16
        
        # Check if CUDA is available and set device accordingly
        if torch.cuda.is_available():
            self.device = "cuda"
            self.use_fp16 = True
            self.dtype = torch.float16 if self.use_fp16 else torch.float32
            print(f"Using device: {self.device} with dtype: {self.dtype}")
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.dtype = torch.float32
            print("WARNING: Running on CPU. This will be very slow and may not work well.")

        try:
            self.pipeline = init_pipeline(pretrained_model_path, self.inference_config, self.device, self.dtype)
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise

        # pre-defined default params, can be changed
        self.steps          = 25
        self.guidance_scale = 7.5
        self.person_prompts = ["boy", "girl", "man", "woman", "person", "eye", "face"]

    def _reload_motion_module(self):
        # somehow the motion module needs to be reloaded every time if the motion lora was applied, otherwise the result could be wrong
        # reloading the motion module only takes 0.2s, so I think it's fine to reload it every time instead of checking if last time the motion lora was applied
        if not os.path.exists(self.motion_module):
            print(f"ERROR: Motion module {self.motion_module} does not exist.")
            print("You need to download the motion module and place it in the models/Motion_Module directory.")
            raise FileNotFoundError(f"Motion module file not found: {self.motion_module}")
        
        try:
            self.pipeline = reload_motion_module(self.pipeline, self.motion_module, self.device)
            print("Motion module reloaded successfully.")
        except Exception as e:
            print(f"Error reloading motion module: {e}")
            raise

    def _get_model_params(self, prompt, width, height, n_prompt, base_model, base_loras, motion_lora):
        prompt = prompt[:-1] if prompt[-1] == "." else prompt
        if base_model is None:
            # when base_model is not specified, use the default model
            # if the prompt contains person-related keywords, use the person model, otherwise use the default model
            isPerson = False
            for keyword in self.person_prompts:
                if keyword in prompt:
                    isPerson = True
                    break

            # load default params
            model_config = self.inference_config.Person if isPerson else self.inference_config.Default
            base_model   = model_config.base_model
            base_loras   = model_config.base_loras
            motion_lora  = model_config.motion_lora if self.version == "v2" else None
            prompt += ", "
            prompt += model_config.prompt
        else:
            # load default params
            model_config = self.inference_config.Default

        # update with user-specified params
        n_prompt = model_config.n_prompt if n_prompt is None else n_prompt
        width    = model_config.width if width is None else width
        height   = model_config.height if height is None else height

        return prompt, width, height, n_prompt, base_model, base_loras, motion_lora

    def _update_model(self, base_model, base_loras, motion_lora):
        # update model
        if base_model and base_model != "":
            try:
                # First check if the base model exists
                base_model_path = os.path.join(self.model_dir, base_model)
                if not os.path.exists(base_model_path) and not base_model.startswith("runwayml/"):
                    print(f"WARNING: Base model {base_model_path} does not exist.")
                    print("Will attempt to use default model from HuggingFace instead.")
                    base_model = "runwayml/stable-diffusion-v1-5"
                
                # Reload motion module
                self._reload_motion_module()
                
                # Load base model
                self.pipeline = load_base_model(self.pipeline, self.model_dir, base_model, self.device, self.dtype)
                
                # Make sure the model is on the right device and dtype
                self.pipeline.to(self.device, self.dtype)
                print(f"Base model loaded successfully: {base_model}")

                # Apply lora
                if base_loras:
                    if len(base_loras) != 0:
                        valid_loras = {}
                        for lora in base_loras:
                            if len(base_loras[lora]) != 2:
                                print(f"WARNING: Skipping lora {lora} - format must be [filename, scale]")
                                continue
                                
                            lora_path = os.path.join(self.model_dir, base_loras[lora][0])
                            if not os.path.exists(lora_path):
                                print(f"WARNING: LoRA file {lora_path} does not exist. Skipping.")
                                continue
                                
                            valid_loras[lora] = base_loras[lora]
                            
                        if valid_loras:
                            self.pipeline = apply_lora(self.pipeline, self.model_dir, valid_loras, device=self.device, dtype=self.dtype)
                            print(f"Applied {len(valid_loras)} LoRA models successfully.")
                        else:
                            print("No valid LoRA models found to apply.")

                # Apply motion lora
                if motion_lora:
                    if self.version == "v1":
                        print("WARNING: motion_lora is not supported in v1. Skipping.")
                    elif len(motion_lora) == 2:
                        motion_lora_path = os.path.join(self.model_dir, motion_lora[0])
                        if not os.path.exists(motion_lora_path):
                            print(f"WARNING: Motion LoRA file {motion_lora_path} does not exist. Skipping.")
                        else:
                            self.pipeline = apply_motion_lora(self.pipeline, self.model_dir, motion_lora, device=self.device, dtype=self.dtype)
                            print(f"Applied motion LoRA successfully: {motion_lora[0]}")
                    else:
                        print("WARNING: motion_lora must be [filename, scale]. Skipping.")
            except Exception as e:
                print(f"Error updating model: {e}")
                raise
        else:
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
        # only prompt is required
        # optional params for inference: steps, guidance_scale, width, height, seed, n_prompt
        # optional params for model: base_model, base_loras, motion_lora
        try:
            print(f"Starting inference with prompt: {prompt}")
            
            prompt, width, height, n_prompt, base_model, base_loras, motion_lora = self._get_model_params(
                prompt, width, height, n_prompt, base_model, base_loras, motion_lora
            )
            
            print(f"Model parameters prepared. Using base model: {base_model}")
            self._update_model(base_model, base_loras, motion_lora)

            # inference
            seed = seed if seed is not None else torch.randint(0, 1000000000, (1,)).item()
            torch.manual_seed(seed)

            print(f"Current seed: {torch.initial_seed()}")
            print(f"Sampling with prompt: {prompt}")
            print(f"Negative prompt: {n_prompt}")
            print(f"Image dimensions: {width}x{height}")
            
            steps = self.steps if steps is None else steps
            print(f"Using {steps} inference steps")
            
            with torch.no_grad():
                print("Starting pipeline inference...")
                sample = self.pipeline(
                    prompt              = prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = steps,
                    guidance_scale      = self.guidance_scale if guidance_scale is None else guidance_scale,
                    width               = width,
                    height              = height,
                    video_length        = self.video_length,
                ).videos
                
                print("Pipeline inference completed successfully")
                save_path = save_video(sample, seed=seed)
                print(f"Video saved to: {save_path}")
                
            return save_path
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
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
