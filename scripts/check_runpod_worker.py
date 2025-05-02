#!/usr/bin/env python3
"""
Script to check the RunPod worker status and logs.
This can help diagnose issues with the RunPod worker.
"""

import os
import sys
import time
import traceback
import logging
import json
import subprocess
import signal
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("runpod-worker-check")

def run_command(command, timeout=30):
    """Run a command and return the output"""
    try:
        logger.info(f"Running command: {command}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        
        # Set up a timer to kill the process if it takes too long
        timer = threading.Timer(timeout, process.kill)
        try:
            timer.start()
            stdout, stderr = process.communicate()
            return_code = process.returncode
        finally:
            timer.cancel()
        
        logger.info(f"Command completed with return code: {return_code}")
        
        if stdout:
            logger.info(f"Command stdout: {stdout}")
        if stderr:
            logger.warning(f"Command stderr: {stderr}")
            
        return return_code, stdout, stderr
    except Exception as e:
        logger.error(f"Error running command: {e}")
        traceback.print_exc()
        return -1, "", str(e)

def check_runpod_cli():
    """Check if RunPod CLI is installed and configured"""
    logger.info("Checking RunPod CLI...")
    
    # Check if runpod command exists
    return_code, stdout, stderr = run_command("which runpod")
    if return_code != 0:
        logger.warning("RunPod CLI not found in PATH")
        return False
    
    # Check runpod version
    return_code, stdout, stderr = run_command("runpod version")
    if return_code != 0:
        logger.warning("RunPod CLI version check failed")
        return False
    
    logger.info("RunPod CLI is installed")
    return True

def check_docker_status():
    """Check Docker status"""
    logger.info("Checking Docker status...")
    
    # Check if docker command exists
    return_code, stdout, stderr = run_command("which docker")
    if return_code != 0:
        logger.warning("Docker not found in PATH")
        return False
    
    # Check docker version
    return_code, stdout, stderr = run_command("docker --version")
    if return_code != 0:
        logger.warning("Docker version check failed")
        return False
    
    # Check docker ps
    return_code, stdout, stderr = run_command("docker ps")
    if return_code != 0:
        logger.warning("Docker ps failed")
        return False
    
    logger.info("Docker is running")
    return True

def check_runpod_worker_logs():
    """Check RunPod worker logs"""
    logger.info("Checking RunPod worker logs...")
    
    # Check docker logs
    return_code, stdout, stderr = run_command("docker ps -a | grep runpod")
    if return_code != 0 and return_code != 1:  # grep returns 1 if no matches
        logger.warning("Failed to check for RunPod containers")
        return False
    
    if "runpod" not in stdout:
        logger.warning("No RunPod containers found")
        return False
    
    # Get container IDs
    containers = []
    for line in stdout.strip().split("\n"):
        if line:
            parts = line.split()
            if parts:
                containers.append(parts[0])
    
    if not containers:
        logger.warning("No RunPod container IDs found")
        return False
    
    logger.info(f"Found RunPod containers: {containers}")
    
    # Check logs for each container
    for container_id in containers:
        logger.info(f"Checking logs for container {container_id}...")
        return_code, stdout, stderr = run_command(f"docker logs {container_id} --tail 100")
        if return_code != 0:
            logger.warning(f"Failed to get logs for container {container_id}")
            continue
        
        # Look for error messages in the logs
        error_keywords = ["error", "exception", "failed", "fatal", "killed", "exited with code"]
        errors_found = []
        
        for line in stdout.strip().split("\n"):
            for keyword in error_keywords:
                if keyword.lower() in line.lower():
                    errors_found.append(line)
                    break
        
        if errors_found:
            logger.warning(f"Found {len(errors_found)} potential error messages in container {container_id} logs:")
            for error in errors_found:
                logger.warning(f"  {error}")
    
    return True

def check_system_resources():
    """Check system resources"""
    logger.info("Checking system resources...")
    
    # Check CPU usage
    return_code, stdout, stderr = run_command("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'")
    if return_code == 0:
        try:
            cpu_usage = float(stdout.strip())
            logger.info(f"CPU usage: {cpu_usage}%")
        except ValueError:
            logger.warning(f"Could not parse CPU usage: {stdout}")
    
    # Check memory usage
    return_code, stdout, stderr = run_command("free -m | awk 'NR==2{printf \"%.2f\", $3*100/$2}'")
    if return_code == 0:
        try:
            memory_usage = float(stdout.strip())
            logger.info(f"Memory usage: {memory_usage}%")
        except ValueError:
            logger.warning(f"Could not parse memory usage: {stdout}")
    
    # Check disk usage
    return_code, stdout, stderr = run_command("df -h / | awk 'NR==2{print $5}'")
    if return_code == 0:
        logger.info(f"Disk usage: {stdout.strip()}")
    
    # Check GPU usage if nvidia-smi is available
    return_code, stdout, stderr = run_command("which nvidia-smi")
    if return_code == 0:
        return_code, stdout, stderr = run_command("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv")
        if return_code == 0:
            logger.info(f"GPU stats:\n{stdout}")
    else:
        logger.warning("nvidia-smi not found, skipping GPU check")
    
    return True

def main():
    """Main function"""
    logger.info("=== RunPod Worker Check ===")
    
    try:
        # Check environment variables
        logger.info("Checking environment variables...")
        runpod_vars = [var for var in os.environ if var.startswith("RUNPOD_")]
        if runpod_vars:
            logger.info(f"Found {len(runpod_vars)} RunPod environment variables")
            for var in runpod_vars:
                # Don't log sensitive values
                if "TOKEN" in var or "SECRET" in var or "KEY" in var:
                    logger.info(f"  {var}: [REDACTED]")
                else:
                    logger.info(f"  {var}: {os.environ.get(var)}")
        else:
            logger.warning("No RunPod environment variables found")
        
        # Run checks
        check_docker_status()
        check_runpod_cli()
        check_runpod_worker_logs()
        check_system_resources()
        
        logger.info("RunPod worker check completed")
        return 0
    except Exception as e:
        logger.error(f"Error during RunPod worker check: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())