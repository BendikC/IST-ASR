#!/usr/bin/env/python3
import os
import sys
import argparse
from pathlib import Path

import torch
import csv
import pandas as pd
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

def create_mini_dataset(csv_path, max_samples=50, random_seed=1234):
    """Creates a smaller version of the dataset with random samples for testing purposes
    Args:
        csv_path (str): Path to the original CSV file.
        max_samples (int): Maximum number of samples to include in the mini dataset.
        random_seed (int): Random seed for reproducibility.
    Returns:
        str: Path to the mini dataset CSV file.
    """

    # first we check if the file exists
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return csv_path
        
    # get the mini_csv path by placing in the same directory as the original
    output_dir = os.path.dirname(csv_path)
    mini_csv_path = os.path.join(output_dir, f"mini_{os.path.basename(csv_path)}")
    
    # read the original csv file and sample randomly
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        mini_df = df.sample(max_samples, random_state=random_seed)
    else:
        mini_df = df
    
    # save the dataset
    mini_df.to_csv(mini_csv_path, index=False)
    logger.info(f"Created mini dataset with {len(mini_df)} samples at {mini_csv_path}")
    
    return mini_csv_path


class ASR(sb.Brain):
    """Main class for the ASR model, defines the training and evaluation logic."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # this is added to track metrics per speaker
        self.speaker_metrics = {}

    def compute_forward(self, batch, stage):
        """
        Forward pass of the model, computing features and applying
        augmentations if specified.
        Args:
            batch: The batch of data to process.
            stage: The current stage (train, valid, test).
        Returns:
            p_ctc: The CTC predictions (if applicable).
            p_ce: The CE predictions (if applicable).
            logits_transducer: The transducer logits.
            wav_lens: The lengths of the waveforms.
        """

        # getting input data
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos

        # if specified, we augment with waveform augmentations
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
                tokens_with_bos = self.hparams.wav_augment.replicate_labels(
                    tokens_with_bos
                )

        feats = self.hparams.compute_features(wavs)

        # feature augmentation is added if specified
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_with_bos = self.hparams.fea_augment.replicate_labels(
                tokens_with_bos
            )

        current_epoch = self.hparams.epoch_counter.current

        # streaming setup if enabled
        if hasattr(self.hparams, "streaming") and self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
        else:
            dynchunktrain_config = None

        # normalizing features
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # encoding features with CNN and encoder
        src = self.modules.CNN(feats)
        x = self.modules.enc(
            src,
            wav_lens,
            pad_idx=self.hparams.pad_index,
            dynchunktrain_config=dynchunktrain_config,
        )
        x = self.modules.proj_enc(x)

        # decoding with RNN
        e_in = self.modules.emb(tokens_with_bos)
        e_in = torch.nn.functional.dropout(
            e_in,
            self.hparams.dec_emb_dropout,
            training=(stage == sb.Stage.TRAIN),
        )
        h, _ = self.modules.dec(e_in)
        h = torch.nn.functional.dropout(
            h, self.hparams.dec_dropout, training=(stage == sb.Stage.TRAIN)
        )
        h = self.modules.proj_dec(h)

        # joint network for learning alignment
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # output layer for transducer log-probabilities
        logits_transducer = self.modules.transducer_lin(joint)

        # compute outputs
        if stage == sb.Stage.TRAIN:
            p_ctc = None
            p_ce = None

            if (
                self.hparams.ctc_weight > 0.0
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                # output layer for ctc log-probabilities
                out_ctc = self.modules.proj_ctc(x)
                p_ctc = self.hparams.log_softmax(out_ctc)

            if self.hparams.ce_weight > 0.0:
                # output layer for ce log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)

            return p_ctc, p_ce, logits_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            # during validation we use greedy beamsearch
            best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
            return logits_transducer, wav_lens, best_hyps
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.Beamsearcher(x)
            return logits_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets.
        
        Args:
            predictions: The model predictions.
            batch: The batch of data.
            stage: The current stage (train, valid, test).
        Returns:
            loss: The computed loss.
        """
        # getting input data
        ids = batch.id
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos

        # train returns 4 elements vs 3 for val and test
        if len(predictions) == 4:
            p_ctc, p_ce, logits_transducer, wav_lens = predictions
        else:
            logits_transducer, wav_lens, predicted_tokens = predictions

        if stage == sb.Stage.TRAIN:
            # extending labels for feature augmentation
            if hasattr(self.hparams, "fea_augment"):
                (
                    tokens,
                    token_lens,
                    tokens_eos,
                    token_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, token_lens, tokens_eos, token_eos_lens
                )

        # loss computation
        if stage == sb.Stage.TRAIN:
            CTC_loss = 0.0
            CE_loss = 0.0
            if p_ctc is not None:
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
            if p_ce is not None:
                CE_loss = self.hparams.ce_cost(
                    p_ce, tokens_eos, length=token_eos_lens
                )
            loss_transducer = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )
            loss = (
                self.hparams.ctc_weight * CTC_loss
                + self.hparams.ce_weight * CE_loss
                + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                * loss_transducer
            )
        else:
            loss = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )

        if stage != sb.Stage.TRAIN:
            # decoding token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            
            # extract speakers from IDs if available
            has_speakers = False
            speaker_ids = []
            
            # Check if IDs look like speaker IDs (e.g., spk_1234)
            if all(isinstance(id, str) and ('_' in id) for id in ids):
                # Extract speaker portion from the ID (assuming format like "spk1_sample1")
                speaker_ids = [id.split('_')[0] for id in ids]
                has_speakers = True
            
            # Track per-speaker metrics if we have speaker information
            if has_speakers:
                for i, speaker_id in enumerate(speaker_ids):
                    spk = str(speaker_id)  # Convert to string to ensure consistency
                    
                    # Initialize speaker stats if needed
                    if spk not in self.speaker_metrics:
                        self.speaker_metrics[spk] = sb.utils.metric_stats.ErrorRateStats()
                    
                    # Add this sample's metrics to the speaker's stats
                    self.speaker_metrics[spk].append(
                        [ids[i]],  # sample id
                        [predicted_words[i]],  # prediction
                        [target_words[i]]  # reference
                    )

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch, initializing metrics."""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.speaker_metrics = {}

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch, reporting metrics and saving checkpoints."""
        # Compute/store important stats
        logger.info(f"Stage {stage} ended with loss: {stage_loss}")
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            logger.info("Computing CER and WER metrics...")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            logger.info(f"CER: {stage_stats['CER']}, WER: {stage_stats['WER']}")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Report speaker metrics if available
            if self.speaker_metrics:
                self._report_speaker_metrics("Validation", save_to_file=True)

            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            # Report speaker metrics if available
            if self.speaker_metrics:
                self._report_speaker_metrics("Test", save_to_file=True)

            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # WER is set to -0.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"WER": -0.1, "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=1,
            )

    def _report_speaker_metrics(self, stage_name, save_to_file=False):
        """Report and optionally save speaker-specific metrics."""
        logger.info(f"\n{stage_name} WER by Speaker:")
        
        # Calculate WER for each speaker
        speaker_wers = {}
        for spk, stats in self.speaker_metrics.items():
            # Get summary stats to check if there's data
            summary = stats.summarize()
            # Check if it has scored tokens, which indicates data was processed
            if "num_scored_tokens" in summary and summary["num_scored_tokens"] > 0:
                speaker_wers[spk] = summary["WER"]
        
        # Sort speakers by WER
        sorted_speakers = sorted(speaker_wers.items(), key=lambda x: x[1])
        
        # Print speaker metrics
        logger.info(f"Found metrics for {len(sorted_speakers)} speakers")
        
        # Print top 5 best performing speakers
        logger.info("\nBest performing speakers:")
        for spk, wer in sorted_speakers[:5]:
            logger.info(f"  Speaker {spk}: WER = {wer:.2f}")
        
        # Print 5 worst performing speakers
        logger.info("\nWorst performing speakers:")
        for spk, wer in sorted_speakers[-5:]:
            logger.info(f"  Speaker {spk}: WER = {wer:.2f}")
        
        # Save detailed speaker metrics to file if requested
        if save_to_file:
            output_dir = os.path.join(self.hparams.output_folder, 'speaker_analysis')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as CSV
            csv_path = os.path.join(output_dir, f"{stage_name.lower()}_speaker_wer.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Speaker', 'WER', 'Num_Tokens', 'Insertions', 'Deletions', 'Substitutions'])
                for spk, stats in self.speaker_metrics.items():
                    summary = stats.summarize()
                    if "num_scored_tokens" in summary and summary["num_scored_tokens"] > 0:
                        writer.writerow([
                            spk, 
                            summary["WER"], 
                            summary["num_scored_tokens"],
                            summary["insertions"],
                            summary["deletions"],
                            summary["substitutions"]
                        ])
            
            logger.info(f"Speaker WER metrics saved to {csv_path}")


    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()

        # Skip checkpoint averaging in test mode
        if hasattr(self.hparams, "test_mode") and self.hparams.test_mode:
            logger.info("Test mode: Skipping checkpoint averaging")
            self.hparams.model.eval()
            return

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # Defining tokenizer and loading it
    # To avoid mismatch, we have to use the same tokenizer used for LM training
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # Add test_mode argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with few samples")
    args, cmd_args = parser.parse_known_args()
    
    # Let SpeechBrain handle the remaining arguments
    sys.argv = [sys.argv[0]] + cmd_args
    
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if we need to prepare LibriSpeech data or use custom CSVs
    if hparams.get("skip_prep", False):
        logger.info("Skipping data preparation, using provided CSV files")
    else:
        # 1. Dataset prep (parsing Librispeech)
        from librispeech_prepare import prepare_librispeech  # noqa

        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "tr_splits": hparams["train_splits"],
                "dev_splits": hparams["dev_splits"],
                "te_splits": hparams["test_splits"],
                "save_folder": hparams["output_folder"],
                "merge_lst": hparams["train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams["skip_prep"],
            },
        )
    
    # Apply test mode if enabled (before dataset creation)
    if args.test_mode:
        logger.info("Test mode enabled! Creating mini datasets with 20 samples each")
        # Create mini datasets for train, valid, and test
        if isinstance(hparams["train_csv"], str):
            hparams["train_csv"] = create_mini_dataset(hparams["train_csv"], max_samples=20)
        
        if isinstance(hparams["valid_csv"], str):
            hparams["valid_csv"] = create_mini_dataset(hparams["valid_csv"], max_samples=20)
        
        # Handle test CSVs (could be a list)
        if isinstance(hparams["test_csv"], list):
            mini_test_csvs = []
            for test_csv in hparams["test_csv"]:
                mini_test_csvs.append(create_mini_dataset(test_csv, max_samples=20))
            hparams["test_csv"] = mini_test_csvs
        elif isinstance(hparams["test_csv"], str):
            hparams["test_csv"] = create_mini_dataset(hparams["test_csv"], max_samples=20)
            
        # Reduce number of epochs for test mode
        if "number_of_epochs" in hparams:
            original_epochs = hparams["number_of_epochs"]
            hparams["number_of_epochs"] = min(original_epochs, 2)
            logger.info(f"Reduced epochs from {original_epochs} to {hparams['number_of_epochs']} for test mode")

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # Load tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Pass test_mode to hparams
    if args.test_mode:
        asr_brain.hparams.test_mode = True
        asr_brain.hparams.Beamsearcher.beam_size = 4
        logger.info("Reduced beam size to 4 for test mode")
    else:
        asr_brain.hparams.test_mode = False
        asr_brain.hparams.Beamsearcher.beam_size = 10

    # We dynamically add the tokenizer to our brain class.
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

        # Modify the train_dataloader_opts reduction in the main section:
    if args.test_mode:
        if "batch_size" in train_dataloader_opts:
            train_dataloader_opts["batch_size"] = min(train_dataloader_opts["batch_size"], 2)
        if "num_workers" in train_dataloader_opts:
            train_dataloader_opts["num_workers"] = 0  # Use 0 workers for test mode
        if "batch_size" in valid_dataloader_opts:
            valid_dataloader_opts["batch_size"] = min(valid_dataloader_opts["batch_size"], 2)
        logger.info("Reduced batch sizes and disabled workers for test mode")

    if train_bsampler is not None and hparams["dynamic_batching"]:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None and hparams["dynamic_batching"]:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}
    
    # Reduce batch size for test mode
    if args.test_mode:
        if "batch_size" in train_dataloader_opts:
            train_dataloader_opts["batch_size"] = min(train_dataloader_opts["batch_size"], 2)
        if "batch_size" in valid_dataloader_opts:
            valid_dataloader_opts["batch_size"] = min(valid_dataloader_opts["batch_size"], 2)
        logger.info("Reduced batch sizes for test mode")

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)
    
    # Reduce test batch size for test mode
    test_loader_opts = hparams["test_dataloader_opts"]
    if args.test_mode and "batch_size" in test_loader_opts:
        test_loader_opts["batch_size"] = min(test_loader_opts["batch_size"], 2)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=test_loader_opts,
            min_key="WER",
        )