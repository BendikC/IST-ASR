#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with pre-split data.

To run this recipe, do the following:
> python train_tokenizer.py hyperparams/your_config.yaml
> python train_tokenizer.py hyperparams/your_config.yaml --test_mode  # For testing with few samples
"""

import sys
import os
import shutil
import argparse
import pandas as pd

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

def setup_csvs(src_csvs, output_folder, test_mode=False, max_samples=10):
    """Copy existing CSV files to the output directory."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy training CSV (this is used for tokenizer training)
    if "train" in src_csvs and os.path.exists(src_csvs["train"]):
        dest_path = os.path.join(output_folder, "train.csv")
        shutil.copy(src_csvs["train"], dest_path)
        
        # If test mode, limit the number of samples
        if test_mode:
            df = pd.read_csv(dest_path)
            limited_df = df.head(max_samples)
            limited_df.to_csv(dest_path, index=False)
            print(f"Limited training data to {len(limited_df)} samples for testing.")
    
    # Copy other CSV files if provided
    if "dev" in src_csvs and os.path.exists(src_csvs["dev"]):
        shutil.copy(src_csvs["dev"], os.path.join(output_folder, "dev.csv"))
        
    if "test" in src_csvs and os.path.exists(src_csvs["test"]):
        shutil.copy(src_csvs["test"], os.path.join(output_folder, "test.csv"))

if __name__ == "__main__":
    # Add test_mode argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with few samples")
    parser.add_argument("--test_samples", type=int, default=10, help="Number of samples to use in test mode")
    args, cmd_args = parser.parse_known_args()
    
    # Let SpeechBrain handle the remaining arguments
    sys.argv = [sys.argv[0]] + cmd_args
    
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Force CPU if specified in hparams
    if "device" in hparams and hparams["device"] == "cpu":
        run_opts["device"] = "cpu"

    # Create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Copy CSV files to output directory
    run_on_main(
        setup_csvs,
        kwargs={
            "src_csvs": hparams["csv_files"],
            "output_folder": hparams["output_folder"],
            "test_mode": args.test_mode,
            "max_samples": args.test_samples,
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()