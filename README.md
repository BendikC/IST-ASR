# IST-ASR

Automatic Speech Recognition (ASR) project for the Inclusive Speech Technology course. 

## Project Overview

This project implements an end-to-end Automatic Speech Recognition system, featuring model training, fine-tuning, and evaluation components. The system is designed to transcribe speech audio into text with a focus on minimizing Word Error Rate (WER).

## Project Structure

- **[experiments/](experiments/)**: Scripts for running and analyzing ASR experiments. Used for training the baseline model
  - `run_experiment.sh`: Main script for launching experiments
  - `test_experiment.sh`: Script for testing experiment results
  - `train_model.py`: Python script for model training
  - `batch_scripts/`: Contains multiple experiment batch scripts
  - `hparams/`: Hyperparameter configuration files
  - `logs/`: Experiment log files

- **[fine-tuning/](fine-tuning/)**: Components for fine-tuning pre-trained models
  - `fine-tune.py`: Fine-tuning implementation
  - `fine-tune.sh`: Shell script wrapper for fine-tuning
  - `fine-tune.yaml`: Configuration for fine-tuning parameters

- **[random_split/](random_split/)**: Dataset splits for training and evaluation
  - `split_stats.txt`: Statistics about the dataset splits
  - `train.csv`: Training dataset
  - `val.csv`: Validation dataset
  - `test.csv`: Test dataset

- **[results/](results/)**: Evaluation results and metrics
  - `aggregated-wer-results.txt`: Compiled Word Error Rate results
  - Results organized by model variant (baseline/, fine-tuning/, model_testing/)

- **[testing/](testing/)**: Model evaluation scripts
  - `test_models.py`: Python script for model evaluation
  - `test_models.sh`: Shell script wrapper for testing

- **[tokenizer/](tokenizer/)**: Tools for tokenization of text data
  - `tokenizer.yaml`: Configuration for the tokenizer
  - `train_tokenizer.py`: Script to train custom tokenizers
  - `train_tokenizer.sh`: Shell wrapper for tokenizer training