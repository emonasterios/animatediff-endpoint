# AnimateDiff Diagnostic Scripts

This directory contains scripts to help diagnose issues with the AnimateDiff RunPod worker.

## Available Scripts

### 1. Environment Check

```bash
./check_environment.py
```

This script checks the system environment and dependencies for AnimateDiff. It provides information about:
- Python version
- System information (platform, processor, memory, disk space)
- CUDA availability and version
- Required dependencies
- Model files
- RunPod environment variables

### 2. Test Inference

```bash
./test_inference.py
```

This script runs a simple test inference with AnimateDiff. It helps diagnose issues with the inference process by:
- Initializing AnimateDiff
- Loading a test input
- Running inference
- Checking the output

### 3. RunPod Worker Check

```bash
./check_runpod_worker.py
```

This script checks the RunPod worker status and logs. It provides information about:
- RunPod environment variables
- Docker status
- RunPod worker logs
- System resources (CPU, memory, disk, GPU)

## Troubleshooting Tips

If the worker is exiting with code 1, try the following steps:

1. Run the environment check script to verify that all dependencies and model files are available:
   ```bash
   ./scripts/check_environment.py
   ```

2. Check the RunPod worker logs for error messages:
   ```bash
   ./scripts/check_runpod_worker.py
   ```

3. Try running a test inference to see if the model works correctly:
   ```bash
   ./scripts/test_inference.py
   ```

4. Check the system resources to ensure there's enough memory and disk space:
   ```bash
   htop
   df -h
   nvidia-smi
   ```

5. Check the server logs for detailed error messages:
   ```bash
   docker logs <container_id>
   ```

## Common Issues

1. **Out of memory**: The model requires a significant amount of GPU memory. Try reducing the image dimensions or using a smaller model.

2. **Missing model files**: Ensure that all required model files are available in the correct directories.

3. **CUDA issues**: Verify that CUDA is available and working correctly.

4. **Dependency issues**: Check that all required dependencies are installed with the correct versions.

5. **Disk space**: Ensure there's enough disk space for temporary files and model outputs.

If you continue to experience issues, please provide the output of these diagnostic scripts when seeking help.