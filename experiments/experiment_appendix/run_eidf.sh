#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create a log file with a timestamp
LOG_FILE="ablation_study_output_$(date +%Y%m%d_%H%M%S).log"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to log and display output
log_and_display() {
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

# Redirect all output to the log file and display it
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Script started at $(date)"

# Install pip if not already installed
if ! command_exists pip; then
    echo "pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    log_and_display python3 get-pip.py
    rm get-pip.py
else
    echo "pip is already installed."
fi

# Install required packages
echo "Installing required packages..."
pip install torch torchvision torchaudio scikit-learn bayesian-optimization wandb tabulate
# Run the Python module
echo "Running the Python module..."
cd /mnt/ceph_rbd/PoolAgent/src/
log_and_display python3 -m experiment_appendix.training_full_ablation

echo "Script execution completed at $(date)"
echo "Log file: $LOG_FILE"