# AnimateDiff_Serverless_Runpod

## 1. Introduction
This is a serverless application that uses [AnimateDiff](https://animatediff.github.io/) to run a **Text-to-Video** task on [RunPod](https://www.runpod.io/).

See also [SDXL_Serverless_Runpod](https://github.com/sky24h/SDXL_Serverless_Runpod) for **Text-to-Imgae** task.

Serverless means that you are only charged for the time you use the application, and you don't need to pay for the idle time, which is very suitable for this kind of application that is not used frequently but needs to respond quickly.

Theoretically, this application can be called by any other application. Here we provide two examples:
1. A simple Python script
2. A Telegram bot

See [Usage](#Usage) below for more details.

### Example Result:
Input Prompt:
(random seed: 445608568)
```
1girl, focus on face, offshoulder, light smile, shiny skin, best quality, masterpiece, photorealistic
```

Result:
(Original | PanLeft, 28 steps, 768x512, around 60 seconds on RTX 3090, 0.015$😱 on RunPod)


https://github.com/sky24h/AnimateDiff_Serverless_Runpod/assets/26270672/b15fb186-b9a3-4077-b212-4b0c22e02dd1




Input Prompt:
(random seed: 195577361)
```
photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3
```

Result:
(Original | ZoomOut, 28 steps, 768x512, around 60 seconds on RTX 3090, 0.015$😱 on RunPod)


https://github.com/sky24h/AnimateDiff_Serverless_Runpod/assets/26270672/563f020f-cd65-433f-8ac9-ee4af4c9d1f9




#### Time Measurement Explanation:
The time is measured from the moment the input prompt is sent to the moment the result image is received, including the time for all the following steps:
- Receive the request from the client
- Serverless container startup
- Model loading
- Inference
- Sending the result image back to the client.

## 2. Dependencies
- Python >= 3.9
- Docker
- Local GPU is necessary for testing but not necessary for deployment. (Recommended: RTX 3090)
- Required model files (see below)

If you don't have a GPU, you can modify and test the code on [Google Colab](https://colab.research.google.com/) and then build and deploy the application on RunPod.

Example Notebook: [link](https://colab.research.google.com/drive/1Gd6uuiItbIFjVPFNyJQhEEEL9khdAyY7?usp=sharing)

### Required Model Files

Before running the application, you need to download the following model files:

1. **Motion Module**: Download the motion module files and place them in the `models/Motion_Module` directory:
   - For v1: `mm_sd_v15_v1-fp16.safetensors`
   - For v2: `mm_sd_v15_v2-fp16.safetensors`

2. **Base Models**: The application uses Stable Diffusion v1.5 by default, which will be downloaded automatically from HuggingFace if not present.

3. **LoRA Models**: If you want to use LoRA models, place them in the `models/DreamBooth_LoRA` directory.

#### Automatic Download

You can use the provided script to download the required model files automatically:

```bash
# Download both v1 and v2 motion modules
python scripts/download_models.py

# Download only v1 motion module
python scripts/download_models.py --version v1

# Download only v2 motion module
python scripts/download_models.py --version v2

# Download motion modules and example LoRA
python scripts/download_models.py --example-lora
```

Alternatively, you can download the motion module files manually from the [AnimateDiff repository](https://github.com/guoyww/AnimateDiff#features).

<a id="Usage"></a>
## 3. Usage
#### 1. Test on Local Machine
```bash
# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py

# Edit (or not) config to customize your inference, e.g., change base model, lora model, motion lora model, etc.
# If inference_v2.yaml doesn't exist, copy inference_v2(example).yaml to inference_v2.yaml
cp inference_v2.yaml inference_v2.yaml

# Run inference test
python inference_util.py

# Run server.py local test
python server.py
```

**Troubleshooting Downloads**:
- If you encounter errors like "gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link of the file.", try reinstalling gdown:
  ```bash
  pip install --upgrade --no-cache-dir gdown
  ```
  Then run the download script again.

- If the download script fails, you can manually download the files from the [AnimateDiff repository](https://github.com/guoyww/AnimateDiff#features) and place them in the appropriate directories.


#### 2. Deploy on RunPod
1. First, make sure you have installed Docker and have accounts on both DockerHub and RunPod.

2. Then, decide a name for your Docker image, e.g., "your_username/anidiff:v1" and set your image name in "./scripts/build.sh".

3. **Important**: Before building the Docker image, make sure you have downloaded the required model files as described in the "Required Model Files" section above. The Docker build process will include these files in the image.

4. Run the following commands to build and push your Docker image to DockerHub:

```bash
bash scripts/build.sh
```

5. Finally, deploy your application on RunPod to create [Template](https://docs.runpod.io/docs/template-creation) and [Endpoint](https://docs.runpod.io/docs/autoscaling).

### Troubleshooting RunPod Deployment

If your RunPod worker shows as "unhealthy" or exits with code 1, check the following:

1. **Missing Motion Module Files**: Make sure you've included the motion module files in your Docker image. These files should be in the `models/Motion_Module` directory.

2. **CUDA Compatibility**: Ensure that the CUDA version in your Docker image is compatible with the GPU on RunPod.

3. **Memory Issues**: If the worker is running out of memory, try using a GPU with more VRAM or reduce the model size/batch size.

4. **Logs**: Check the worker logs in the RunPod dashboard for specific error messages.

Feel free to contact me if you encounter any problems after checking these common issues.

#### 3. Call the Application
##### Call from a Python script
```
# Make sure to set API key and endpoint ID before running the script.
python test_client.py
```

##### Showcase: Call from a Telegram bot
![Example Result](./assets/telegram_bot_example.jpg)

## 4. Testing
The project includes unit tests to ensure the functionality of the code. The tests are located in the `tests` directory.

To run the tests, navigate to the `tests` directory and run:

```bash
python -m unittest test_check_data_format.py
```

See the [tests/README.md](./tests/README.md) file for more information about the tests.

## 5. TODO
- [x] Support for specific base model for different objectives. (Person and Scene)
- [x] Support for LoRA models. (Edit yaml file and place your model in "./models/DreamBooth_LoRA")
- [x] Support for Motion LoRA models. (Also editable in yaml file, see [here](https://github.com/guoyww/AnimateDiff#features) for details and downloads.)
- [x] Add unit tests for input validation
- [ ] More detailed instructions
- [ ] One-click deploy (If anyone is interested...)

## 6. Acknowledgement
Thanks to [AnimateDiff](https://animatediff.github.io/) and [RunPod](https://www.runpod.io/).
