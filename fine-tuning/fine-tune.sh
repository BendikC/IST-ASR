#!/bin/bash

#SBATCH --job-name=asr-fine-tuning
#SBATCH --partition=gpu-v100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --output=fine-tuning/speakers_output_%j.log
#SBATCH --error=fine-tuning/speakers_error_%j.log


# Load modules:
module load 2023r1
module load cuda/11.6
module load openmpi
module load miniconda3
module load ffmpeg
module load git-lfs

# Activate the conda environment
conda activate IST-ASR

# Run your script
python fine-tuning/fine-tune.py fine-tuning/fine-tune.yaml

# Deactivate the environment when done
conda deactivate