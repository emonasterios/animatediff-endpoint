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

# Download actual models at build time
RUN python3 - <<EOF
from huggingface_hub import snapshot_download

# 1) Stable Diffusion v1.5
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="models/StableDiffusion/stable-diffusion-v1-5",
    local_dir_use_symlinks=False
)

# 2) Motion module
snapshot_download(
    repo_id="openai/AnimateDiff-Motion-Module",
    local_dir="models/Motion_Module",
    local_dir_use_symlinks=False
)
EOF

# Debug: list downloaded model files
RUN echo "Listing StableDiffusion model files:" \
 && ls -R models/StableDiffusion/stable-diffusion-v1-5 \
 && echo "Listing Motion Module model files:" \
 && ls -R models/Motion_Module


# Run server
CMD [ "python", "-u", "./server.py" ]
