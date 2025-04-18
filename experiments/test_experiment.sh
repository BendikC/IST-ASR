#!/bin/bash

#SBATCH --job-name=asr-training
#SBATCH --partition=gpu-v100
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --output=experiments/logs/%j_output_test.log  # Added output logging
#SBATCH --error=experiments/logs/%j_error_test.log    # Added error logging

# Load modules:
module load 2023r1
module load cuda/11.6
module load openmpi
module load miniconda3
module load ffmpeg

# Activate the conda environment
conda activate IST-ASR

# Run your script
python experiments/train_model.py experiments/hparams/1.yaml --test_mode

# Deactivate the environment when done
conda deactivate