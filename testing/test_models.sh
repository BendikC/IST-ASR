#!/bin/bash

#SBATCH --job-name=asr-model-testing
#SBATCH --partition=gpu-v100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --output=testing/o_libri_%j.log
#SBATCH --error=testing/e_libri_%j.log

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
python testing/test_models.py

# Deactivate the environment when done
conda deactivate