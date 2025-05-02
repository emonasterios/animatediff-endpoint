# Include base image
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Define working directory
WORKDIR /workspace/

# Set timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 vim htop procps net-tools curl
RUN apt-get autoremove -y && apt-get clean -y

# Add pretrained model
ADD animatediff ./animatediff
ADD models ./models

# Add necessary files
ADD inference_v1.yaml ./
ADD inference_v2.yaml ./
ADD inference_util.py ./
ADD server.py ./

# Add test files
ADD test_input.json ./
ADD test_input_simple.json ./

# Add diagnostic scripts
ADD scripts ./scripts
RUN chmod +x ./scripts/*.py

# pip install
ADD requirements.txt ./
RUN pip install -r requirements.txt
# Install xformers for GPU acceleration
RUN pip install xformers==0.0.20
# Install additional diagnostic tools
RUN pip install psutil huggingface_hub gdown

# Create minimal model structure for testing without downloading
RUN echo "Setting up model structure..." && \
    mkdir -p /workspace/models/StableDiffusion/stable-diffusion-v1-5/vae && \
    mkdir -p /workspace/models/StableDiffusion/stable-diffusion-v1-5/unet && \
    mkdir -p /workspace/models/StableDiffusion/stable-diffusion-v1-5/text_encoder && \
    mkdir -p /workspace/models/StableDiffusion/stable-diffusion-v1-5/tokenizer && \
    mkdir -p /workspace/models/StableDiffusion/stable-diffusion-v1-5/scheduler && \
    echo '{"_class_name":"StableDiffusionPipeline","_diffusers_version":"0.6.0","scheduler":["diffusers","PNDMScheduler"],"text_encoder":["transformers","CLIPTextModel"],"tokenizer":["transformers","CLIPTokenizer"],"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}' > /workspace/models/StableDiffusion/stable-diffusion-v1-5/model_index.json && \
    mkdir -p /workspace/models/Motion_Module && \
    touch /workspace/models/Motion_Module/mm_sd_v15_v1-fp16.safetensors && \
    touch /workspace/models/Motion_Module/mm_sd_v15_v2-fp16.safetensors && \
    echo "Model structure setup completed"

# Run server
CMD [ "python", "-u", "./server.py" ]
