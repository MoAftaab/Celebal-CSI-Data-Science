#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Complete Pipeline
------------------------------------------
This script runs the complete network anomaly detection pipeline.
"""

import os
import argparse
import logging
import subprocess
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_pipeline')


def run_script(script_path, description):
    """
    Run a Python script.
    
    Parameters:
    -----------
    script_path : str
        Path to the script to run
    description : str
        Description of the script for logging
    
    Returns:
    --------
    int
        Return code from the script
    """
    logger.info(f"Running {description}...")
    start_time = time.time()
    
    # Run the script
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    
    # Check result
    if result.returncode == 0:
        elapsed_time = time.time() - start_time
        logger.info(f"{description} completed successfully in {elapsed_time:.2f} seconds")
        
        # Print output
        if result.stdout:
            logger.info(f"Output from {os.path.basename(script_path)}:\n{result.stdout}")
    else:
        logger.error(f"{description} failed with code {result.returncode}")
        
        # Print error
        if result.stderr:
            logger.error(f"Error from {os.path.basename(script_path)}:\n{result.stderr}")
    
    return result.returncode


def run_pipeline(steps=None, start_streamlit=True):
    """
    Run the complete pipeline or specific steps.
    
    Parameters:
    -----------
    steps : list or None
        List of steps to run (if None, all steps are run)
    start_streamlit : bool
        Whether to start the Streamlit app after running the pipeline
    
    Returns:
    --------
    bool
        True if all steps completed successfully, False otherwise
    """
    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define steps with script paths and descriptions
    pipeline_steps = {
        'preprocess': {
            'script': os.path.join(base_dir, 'data_preprocessing.py'),
            'description': 'Data preprocessing'
        },
        'train': {
            'script': os.path.join(base_dir, 'model_training.py'),
            'description': 'Model training'
        },
        'evaluate': {
            'script': os.path.join(base_dir, 'model_evaluation.py'),
            'description': 'Model evaluation'
        }
    }
    
    # If no steps specified, run all steps
    if steps is None:
        steps = list(pipeline_steps.keys())
    
    # Check if specified steps are valid
    for step in steps:
        if step not in pipeline_steps:
            logger.error(f"Invalid step: {step}")
            return False
    
    # Run each step
    all_successful = True
    
    for step in steps:
        step_info = pipeline_steps[step]
        return_code = run_script(step_info['script'], step_info['description'])
        
        if return_code != 0:
            all_successful = False
            logger.error(f"Pipeline step '{step}' failed. Stopping pipeline.")
            return False
    
    # Start Streamlit app if requested
    if start_streamlit and all_successful:
        logger.info("Starting Streamlit app...")
        streamlit_script = os.path.join(base_dir, 'app.py')
        
        # Check if the script exists
        if not os.path.exists(streamlit_script):
            logger.error(f"Streamlit script not found: {streamlit_script}")
            return False
        
        # Start Streamlit in a new process (non-blocking)
        streamlit_process = subprocess.Popen(['streamlit', 'run', streamlit_script])
        
        logger.info(f"Streamlit app started with PID {streamlit_process.pid}")
        logger.info("Press Ctrl+C to stop the app")
        
        try:
            # Wait for the process to complete (or for the user to interrupt)
            streamlit_process.wait()
        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C)
            logger.info("Stopping Streamlit app...")
            streamlit_process.terminate()
    
    return all_successful


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the network anomaly detection pipeline.')
    
    parser.add_argument('--steps', nargs='+', choices=['preprocess', 'train', 'evaluate', 'all'],
                      help='Pipeline steps to run (default: all)')
    
    parser.add_argument('--no-streamlit', action='store_true',
                      help='Do not start the Streamlit app after running the pipeline')
    
    args = parser.parse_args()
    
    # Process steps argument
    if args.steps is None or 'all' in args.steps:
        pipeline_steps = None  # Run all steps
    else:
        pipeline_steps = args.steps
    
    # Run the pipeline
    success = run_pipeline(steps=pipeline_steps, start_streamlit=not args.no_streamlit)
    
    # Exit with appropriate code
    exit(0 if success else 1) 