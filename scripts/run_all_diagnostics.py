#!/usr/bin/env python3
"""
Script to run all diagnostic checks for AnimateDiff.
This script runs all the diagnostic scripts in sequence and saves the output to a file.
"""

import os
import sys
import subprocess
import datetime
import traceback

def run_script(script_path, output_file):
    """Run a script and write the output to a file"""
    try:
        print(f"Running {script_path}...")
        with open(output_file, "a") as f:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Running {script_path} at {datetime.datetime.now()}\n")
            f.write(f"{'='*80}\n\n")
        
        process = subprocess.Popen(
            [script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        with open(output_file, "a") as f:
            for line in process.stdout:
                f.write(line)
                print(line, end="")
        
        process.wait()
        
        with open(output_file, "a") as f:
            f.write(f"\nScript completed with return code: {process.returncode}\n")
        
        print(f"{script_path} completed with return code: {process.returncode}")
        return process.returncode
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        traceback.print_exc()
        with open(output_file, "a") as f:
            f.write(f"\nError running {script_path}: {e}\n")
            traceback.print_exc(file=f)
        return -1

def main():
    """Main function"""
    print("=== Running All AnimateDiff Diagnostics ===")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"diagnostics_{timestamp}.log")
    
    print(f"Output will be saved to: {output_file}")
    
    with open(output_file, "w") as f:
        f.write(f"AnimateDiff Diagnostics - {datetime.datetime.now()}\n")
        f.write(f"{'='*80}\n\n")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run scripts
    scripts = [
        os.path.join(script_dir, "check_environment.py"),
        os.path.join(script_dir, "check_runpod_worker.py"),
        os.path.join(script_dir, "test_inference.py")
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_script(script, output_file)
        else:
            print(f"Script not found: {script}")
            with open(output_file, "a") as f:
                f.write(f"\nScript not found: {script}\n")
    
    # Add system information
    with open(output_file, "a") as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Additional System Information at {datetime.datetime.now()}\n")
        f.write(f"{'='*80}\n\n")
    
    # Run additional commands
    commands = [
        "uname -a",
        "df -h",
        "free -h",
        "nvidia-smi",
        "docker ps -a",
        "docker images",
        "pip list"
    ]
    
    for command in commands:
        try:
            print(f"Running command: {command}")
            with open(output_file, "a") as f:
                f.write(f"\n$ {command}\n")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                universal_newlines=True
            )
            
            with open(output_file, "a") as f:
                for line in process.stdout:
                    f.write(line)
            
            process.wait()
            
            with open(output_file, "a") as f:
                f.write(f"\nCommand completed with return code: {process.returncode}\n")
            
            print(f"Command completed with return code: {process.returncode}")
        except Exception as e:
            print(f"Error running command {command}: {e}")
            with open(output_file, "a") as f:
                f.write(f"\nError running command {command}: {e}\n")
    
    print(f"\nAll diagnostics completed. Results saved to: {output_file}")
    print(f"Please provide this file when seeking help with issues.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())