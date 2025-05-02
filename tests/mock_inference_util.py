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