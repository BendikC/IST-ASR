#!/bin/bash

#SBATCH --job-name=example-gpu
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2023r1
module load cuda/11.6
module load openmpi
module load miniconda3

# Activate the conda environment
conda activate IST-ASR

# Run your script
python train_tokenizer.py hparams/tokenizer.yaml

# Deactivate the environment when done
conda deactivate