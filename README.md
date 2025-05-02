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

## 3. Troubleshooting

Si encuentras problemas con el worker de RunPod que sale con código 1, puedes usar los scripts de diagnóstico incluidos en el directorio `scripts` para ayudar a identificar el problema.

### Scripts de Diagnóstico

Los siguientes scripts de diagnóstico están disponibles:

1. **Verificación de Carga de Modelos**: Comprueba si los archivos de modelo existen y pueden cargarse correctamente
   ```bash
   python scripts/check_model_loading.py
   ```

2. **Prueba del Endpoint de RunPod**: Prueba el endpoint de RunPod con una solicitud simple
   ```bash
   python scripts/test_runpod.py --endpoint-id TU_ENDPOINT_ID --api-key TU_API_KEY
   # O usando el script de shell
   ./scripts/test_runpod.sh TU_ENDPOINT_ID TU_API_KEY "Un gato caminando"
   ```

3. **Prueba Local sin RunPod**: Prueba el servidor localmente sin necesidad de RunPod
   ```bash
   python test_local.py
   ```

4. **Descargar Modelos**: Descarga los archivos de modelo necesarios
   ```bash
   python scripts/download_models.py
   # Con opciones
   python scripts/download_models.py --version v1  # Solo módulo de movimiento v1
   python scripts/download_models.py --version v2  # Solo módulo de movimiento v2
   python scripts/download_models.py --example-lora  # Incluir LoRA de ejemplo
   python scripts/download_models.py --create-minimal  # Crear archivos de marcador de posición mínimos
   ```

### Problemas Comunes

1. **Worker salió con código de salida 1**: Esto generalmente es causado por archivos de modelo faltantes. El problema más común es que los archivos del modelo Stable Diffusion v1.5 faltan en el directorio `/workspace/models/StableDiffusion/stable-diffusion-v1-5`. Asegúrate de que existan los siguientes archivos:
   - `/workspace/models/StableDiffusion/stable-diffusion-v1-5/text_encoder/pytorch_model.bin`
   - `/workspace/models/StableDiffusion/stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin`
   - `/workspace/models/StableDiffusion/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin`
   - `/workspace/models/StableDiffusion/stable-diffusion-v1-5/tokenizer/vocab.json`

2. **Memoria insuficiente**: El modelo requiere una cantidad significativa de memoria GPU. Intenta reducir las dimensiones de la imagen o usar un modelo más pequeño.

3. **Archivos de modelo faltantes**: Asegúrate de que todos los archivos de modelo requeridos estén disponibles en los directorios correctos. El Dockerfile actualizado crea una estructura de directorios mínima, pero aún necesitas descargar los archivos de modelo reales.

4. **Problemas de CUDA**: Verifica que CUDA esté disponible y funcionando correctamente. Los registros del servidor mostrarán la disponibilidad de CUDA al inicio.

5. **Problemas de dependencias**: Verifica que todas las dependencias requeridas estén instaladas con las versiones correctas. El Dockerfile incluye todas las dependencias necesarias.

6. **Espacio en disco**: Asegúrate de que haya suficiente espacio en disco para archivos temporales y salidas de modelo. Los registros del servidor mostrarán el uso del disco al inicio si psutil está instalado.

7. **Depuración con registro mejorado**: El código actualizado incluye registro mejorado para ayudar a diagnosticar problemas. Verifica los registros del worker en el panel de RunPod para mensajes de error detallados.

### Solución del Problema "Worker salió con código de salida 1"

Si estás experimentando el problema "worker salió con código de salida 1", sigue estos pasos:

1. **Verificar archivos de modelo**: Asegúrate de que todos los archivos de modelo requeridos estén presentes. Puedes usar el script `check_model_loading.py` para verificar:
   ```bash
   python scripts/check_model_loading.py
   ```

2. **Descargar modelos faltantes**: Si faltan archivos de modelo, descárgalos usando el script proporcionado:
   ```bash
   python scripts/download_models.py
   ```

3. **Reconstruir imagen Docker**: Después de descargar los modelos, reconstruye tu imagen Docker:
   ```bash
   bash scripts/build.sh
   ```

4. **Implementar en RunPod**: Implementa la imagen actualizada en RunPod y crea un nuevo endpoint.

5. **Probar el endpoint**: Usa el script de prueba proporcionado para verificar que el endpoint funcione:
   ```bash
   python scripts/test_runpod.py --endpoint-id TU_ENDPOINT_ID --api-key TU_API_KEY
   ```

Para información de solución de problemas más detallada, verifica los registros del worker en el panel de RunPod.

### Mejoras Implementadas para Solucionar el Problema

Hemos realizado las siguientes mejoras para solucionar el problema "worker salió con código de salida 1":

1. **Manejo de errores mejorado**: El código ahora maneja correctamente los errores cuando faltan archivos de modelo y proporciona mensajes de error claros.

2. **Estructura de modelos mínima**: El Dockerfile crea automáticamente una estructura de directorios mínima para los modelos, lo que permite que el worker se inicie correctamente incluso sin los archivos de modelo reales.

3. **Verificación de archivos de modelo**: El servidor ahora verifica la existencia de archivos de modelo críticos al inicio y proporciona advertencias claras si faltan.

4. **Registro detallado**: Se ha mejorado el registro para proporcionar información más detallada sobre lo que está sucediendo durante la inicialización y la inferencia.

5. **Scripts de diagnóstico**: Se han añadido scripts de diagnóstico para ayudar a identificar y solucionar problemas comunes.

6. **Manejo gracioso de errores**: El servidor ahora maneja los errores de manera más elegante y proporciona respuestas de error útiles en lugar de simplemente fallar.

7. **Prueba local**: Se ha añadido un script de prueba local para verificar el servidor sin necesidad de RunPod.

<a id="Usage"></a>
## 4. Usage
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

1. **Missing Model Files**: The most common cause of "worker exited with exit code 1" is missing model files. Make sure you've included:
   - The Stable Diffusion v1.5 model in `models/StableDiffusion/stable-diffusion-v1-5`
   - The motion module files in `models/Motion_Module` directory (mm_sd_v15_v1-fp16.safetensors and/or mm_sd_v15_v2-fp16.safetensors)

2. **Testing the Endpoint**: Use the provided test scripts to verify your endpoint:
   ```bash
   # Using Python script
   python test_runpod.py --endpoint-id YOUR_ENDPOINT_ID --api-key YOUR_API_KEY
   
   # Using shell script
   ./test_runpod.sh YOUR_ENDPOINT_ID YOUR_API_KEY
   ```

3. **Automatic Model Download**: The updated Dockerfile now includes automatic model download during build. Make sure you're using the latest version of the Dockerfile.

4. **CUDA Compatibility**: Ensure that the CUDA version in your Docker image is compatible with the GPU on RunPod.

5. **Memory Issues**: If the worker is running out of memory, try using a GPU with more VRAM or reduce the model size/batch size.

6. **Logs**: Check the worker logs in the RunPod dashboard for specific error messages. The updated code includes more detailed logging to help diagnose issues.

7. **Manual Model Download**: If automatic download fails, you can manually download the models:
   - Stable Diffusion v1.5: Download from [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   - Motion Modules: Download from [AnimateDiff repository](https://github.com/guoyww/AnimateDiff#features)

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
