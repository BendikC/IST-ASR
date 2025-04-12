#!/usr/bin/python3
import os
import sys
import torch
import logging
import pandas as pd

from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.ASR import WhisperASR

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_whisper_model(model_path):
    """
    Load a Whisper model from a local path with fallback options for offline clusters
    
    Args:
        model_path (str): Path to the local model directory
    
    Returns:
        Dict with loaded model and processor
    """ 
    processor = WhisperProcessor.from_pretrained(
        model_path,
        use_safetensors=False,
        local_files_only=True
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        use_safetensors=False,
        local_files_only=True,
        torch_dtype=torch.float16  # Use float16 to save memory
    )
    print("Successfully loaded model with safetensors disabled", flush=True)
    return {
        "processor": processor,
        "model": model
    }

class SimpleWhisperASR(torch.nn.Module):
    """Simple Whisper ASR class that inherits from nn.Module"""
    
    def __init__(self, whisper_model, whisper_processor, device="cpu"):
        super().__init__()  # Initialize the parent class
        
        self.model = whisper_model
        self.processor = whisper_processor
        self.device = device
        
        # Move model to the right device
        self.model = self.model.to(device)
    
    def forward(self, x):
        # Required method for nn.Module - can redirect to transcribe or similar
        return self.transcribe(x)
    
    def transcribe(self, audio, language='en', task='transcribe'):
        """Transcribe audio using Whisper model"""
        # Process audio based on format
        if isinstance(audio, torch.Tensor):
            # If multiple files are given (batch), convert each separately
            if len(audio.shape) == 2:
                processed_audio = []
                for i in range(audio.shape[0]):
                    waveform = audio[i].cpu().numpy()
                    processed_audio.append(waveform)
                audio = processed_audio
            else:
                # Single file
                audio = audio.cpu().numpy()
        
        # Get model dtype once
        if not hasattr(self, '_model_dtype'):
            self._model_dtype = next(self.model.parameters()).dtype
            print(f"Model is using dtype: {self._model_dtype}")
        
        # Use processor and model directly with explicit sampling rate
        # Most audio models use 16kHz sampling rate
        input_features = self.processor(
            audio, 
            sampling_rate=16000,  # Explicitly set sampling rate
            return_tensors="pt"
        ).input_features
        
        # Create attention mask for batched inputs (all ones since we don't have padding)
        attention_mask = None
        if isinstance(input_features, torch.Tensor) and len(input_features.shape) > 2:
            # For batched inputs, create an attention mask
            attention_mask = torch.ones(
                input_features.shape[0], 
                input_features.shape[1], 
                device=input_features.device
            )
        
        # IMPORTANT: Match the dtype of input_features with the model's dtype
        input_features = input_features.to(device=self.device, dtype=self._model_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        
        # Generate token ids from the input audio
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )
        
        # Generate with attention mask
        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            attention_mask=attention_mask,
            max_length=448,
        )
        
        # Convert token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )
        
        # Return as list for batch processing
        return transcription, None
    
    def transcribe_batch(self, wavs, wav_lens=None):
        """Transcribe a batch of audio files"""
        # Process the audio batch 
        transcriptions, _ = self.transcribe(wavs)
        
        # Return as list for SpeechBrain
        return transcriptions

class ASRBrain(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add speaker tracking
        self.speaker_metrics = {}

    def compute_forward(self, batch, stage):
        """Forward computations from the batch to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Perform ASR
        if hasattr(self.modules, 'encoder_model') and hasattr(self.modules.encoder_model, 'transcribe'):
            # For Whisper models
            predicted_tokens, predicted_probs = self.modules.encoder_model.transcribe(
                wavs, 
                language='en',
                task='transcribe'
            )
            return predicted_tokens, predicted_probs
        elif hasattr(self.modules, 'asr_model') and hasattr(self.modules.asr_model, 'transcribe'):
            # For SpeechBrain 1.0+ inference models
            hypotheses = self.modules.asr_model.transcribe_batch(
                wavs, 
                wav_lens
            )
            # Convert to compatible format
            return hypotheses, None
        else:
            # Fallback for custom models
            raise ValueError("No compatible ASR model found in modules")

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss for the given stage."""
        # Get reference texts
        words = batch.wrd

        # Extract speaker IDs if available
        speaker_ids = batch.id
        has_speakers = False

        # Check if IDs look like speaker IDs (e.g., spk_1234)
        if hasattr(batch, 'spk_id'):
            speaker_ids = batch.spk_id
            has_speakers = True
        elif all(isinstance(id, str) and ('_' in id) for id in speaker_ids):
            # Try to extract speaker IDs from the sample IDs
            # This assumes IDs like "spk1_sample1" or similar format
            has_speakers = True
        
        # Handle predictions
        predicted_text = []
        if isinstance(predictions, tuple):
            predicted_tokens, _ = predictions
            if isinstance(predicted_tokens, list) and isinstance(predicted_tokens[0], str):
                predicted_text = predicted_tokens
            else:
                predicted_text = ["" for _ in range(len(words))]
        elif isinstance(predictions, list):
            predicted_text = predictions
        
        # Create ID list for WER computation
        ids = list(range(len(words)))

        standardized_predictions = [self.standardize_text(text) for text in predicted_text]
        standardized_references = [self.standardize_text(text) for text in words]
        
        # Always compute WER for logging
        error_rate = sb.utils.metric_stats.ErrorRateStats()
        error_rate.append(ids, standardized_predictions, standardized_references)
        wer_stats = error_rate.summarize()
        
        # Store WER for logging during validation/test
        if stage != sb.Stage.TRAIN:
            self.wer_metric = wer_stats["WER"]
            
            # Track per-speaker metrics if we have speaker information
            if has_speakers:
                for i, speaker_id in enumerate(speaker_ids):
                    spk = str(speaker_id)  # Convert to string to ensure consistency
                    
                    # Initialize speaker stats if needed
                    if spk not in self.speaker_metrics:
                        self.speaker_metrics[spk] = sb.utils.metric_stats.ErrorRateStats()
                    
                    # Add this sample's metrics to the speaker's stats
                    self.speaker_metrics[spk].append(
                        [i],  # sample id
                        [standardized_predictions[i]],  # prediction
                        [standardized_references[i]]  # reference
                    )
            
            # Print some examples for debugging
            if stage == sb.Stage.VALID and self.step % 100 == 0:
                for i in range(min(3, len(ids))):
                    print(f"Example {i}:", flush=True)
                    print(f"  Speaker: '{speaker_ids[i]}'", flush=True)
                    print(f"  Raw Reference: '{words[i]}'", flush=True)
                    print(f"  Raw Prediction: '{predicted_text[i]}'", flush=True)
                    print(f"  Std Reference: '{standardized_references[i]}'", flush=True)
                    print(f"  Std Prediction: '{standardized_predictions[i]}'", flush=True)
                    print(f"  WER: {wer_stats['WER']:.2f}", flush=True)
        
        # Always return a tensor with grad requirement based on stage
        return torch.tensor(
            wer_stats["WER"], 
            device=self.device, 
            requires_grad=(stage == sb.Stage.TRAIN)
        )
    
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        if stage != sb.Stage.TRAIN:
            # Initialize metric for validation/test
            self.wer_metric = 0
            self.speaker_metrics = {}  # Reset speaker metrics for each evaluation

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            print(f"Validation loss: {stage_loss}", flush=True)
            print(f"Validation WER: {self.wer_metric}", flush=True)

            # Report speaker metrics if available
            if self.speaker_metrics:
                self._report_speaker_metrics("Validation")
            
            # For validation, you'd typically save checkpoints if better than previous
            if hasattr(self, "checkpointer") and self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta={"WER": self.wer_metric, "epoch": epoch},
                    min_keys=["WER"],
                )
        elif stage == sb.Stage.TEST:
            print(f"Test loss: {stage_loss}", flush=True)
            print(f"Test WER: {self.wer_metric}", flush=True)
        
                        # Report speaker metrics and save to file
            if self.speaker_metrics:
                self._report_speaker_metrics("Test", save_to_file=True)
    
    def _report_speaker_metrics(self, stage_name, save_to_file=False):
        """Report and optionally save speaker-specific metrics."""
        print(f"\n{stage_name} WER by Speaker:", flush=True)
        
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
        print(f"Found metrics for {len(sorted_speakers)} speakers", flush=True)
        
        # Print top 5 best performing speakers
        print("\nBest performing speakers:", flush=True)
        for spk, wer in sorted_speakers[:5]:
            print(f"  Speaker {spk}: WER = {wer:.2f}", flush=True)
        
        # Print 5 worst performing speakers
        print("\nWorst performing speakers:", flush=True)
        for spk, wer in sorted_speakers[-5:]:
            print(f"  Speaker {spk}: WER = {wer:.2f}", flush=True)
        
        # Save detailed speaker metrics to file if requested
        if save_to_file:
            output_dir = './fine-tuning/outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as CSV
            import csv
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
            
            print(f"Speaker WER metrics saved to {csv_path}", flush=True)
            
            # Create visualization if matplotlib is available
            try:
                self._visualize_speaker_metrics(stage_name, speaker_wers)
            except ImportError:
                print("Matplotlib not available, skipping visualization", flush=True)

    def _visualize_speaker_metrics(self, stage_name, speaker_wers):
        """Create visualizations for speaker metrics."""
        import matplotlib.pyplot as plt
        
        output_dir = './fine-tuning/outputs'
        
        # Create WER distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(list(speaker_wers.values()), bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Number of Speakers')
        plt.title(f'{stage_name} WER Distribution Across Speakers')
        plt.axvline(x=self.wer_metric, color='red', linestyle='--', 
                   label=f'Overall WER: {self.wer_metric:.2f}')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'{stage_name.lower()}_wer_distribution.png'))
        plt.close()
        
        # Create top/bottom performers bar chart
        num_speakers = min(10, len(speaker_wers) // 2)  # Show top/bottom 10 speakers or half if fewer
        plt.figure(figsize=(12, 8))
        
        # Get top and bottom speakers
        sorted_speakers = sorted(speaker_wers.items(), key=lambda x: x[1])
        top_speakers = sorted_speakers[:num_speakers]
        bottom_speakers = sorted_speakers[-num_speakers:]
        
        # Combine for plotting
        all_speakers = top_speakers + bottom_speakers
        spk_ids = [spk for spk, _ in all_speakers]
        wers = [wer for _, wer in all_speakers]
        
        # Create colors (green for best, red for worst)
        colors = ['green'] * num_speakers + ['red'] * num_speakers
        
        # Plot
        bars = plt.bar(range(len(spk_ids)), wers, color=colors)
        plt.xlabel('Speaker ID')
        plt.ylabel('Word Error Rate (WER)')
        plt.title(f'Best and Worst Performing Speakers - {stage_name}')
        plt.xticks(range(len(spk_ids)), spk_ids, rotation=90)
        plt.axhline(y=self.wer_metric, color='blue', linestyle='--',
                   label=f'Overall WER: {self.wer_metric:.2f}')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'{stage_name.lower()}_speaker_performance.png'))
        plt.close()
        
        print(f"Speaker visualizations saved to {output_dir}", flush=True)

    def standardize_text(self, text):
        """
        Standardize text for consistent comparison:
        - Remove punctuation
        - Convert to uppercase
        - Remove extra whitespace
        
        Args:
            text (str): Text to standardize
            
        Returns:
            str: Standardized text
        """
        import re
        # Remove all punctuation except apostrophes (to handle contractions)
        text = re.sub(r'[^\w\s\']', '', text)
        # Replace apostrophes with space
        text = text.replace("'", " ")
        # Convert to uppercase
        text = text.upper()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

def create_dataset(csv_file, tokenizer=None, sample_size=10):
    """
    Create a SpeechBrain dataset from CSV with speaker information
    
    Args:
        csv_file (str): Path to CSV file
        tokenizer (object, optional): Tokenizer for text processing
        sample_size (int, optional): Number of samples to use (0 for all)
        
    Returns:
        DynamicItemDataset: SpeechBrain dataset
    """
    print(f"Loading dataset from {csv_file}...", flush=True)
     
    # Apply sampling if requested by creating a temporary CSV
    temp_csv = None
    if sample_size > 0:
        df = pd.read_csv(csv_file)
        if len(df) > sample_size:
            # Sample the dataframe
            sampled_df = df.sample(sample_size, random_state=1234)
            
            # Create a temporary CSV file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_csv = temp_file.name
            sampled_df.to_csv(temp_csv, index=False)
            print(f"Sampled {sample_size} entries from {csv_file}", flush=True)
            
            # Use the temporary CSV instead
            csv_file = temp_csv
    
    try:
        # Create dataset from CSV directly
        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file,
            replacements={"data_root": os.path.dirname(csv_file)},
        )
        
        # Check if spk_id exists in the dataset
        df = pd.read_csv(csv_file)
        has_speaker_info = 'spk_id' in df.columns
        if not has_speaker_info:
            print("WARNING: No 'spk_id' column found in CSV. Speaker analysis will be limited.", flush=True)
        
        # Define audio pipeline
        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            sig = sb.dataio.dataio.read_audio(wav)
            return sig
        
        # Define text pipeline
        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides("wrd", "tokens", "tokens_lens")
        def text_pipeline(wrd):
            yield wrd
            
            # Only tokenize if a tokenizer is provided
            if tokenizer is not None:
                tokens_list = tokenizer.encode_as_ids(wrd)
                tokens = torch.LongTensor(tokens_list)
                yield tokens
                yield len(tokens)
            else:
                # Return empty values if no tokenizer
                yield torch.LongTensor([])
                yield 0
        
        # Define speaker ID pipeline if available
        @sb.utils.data_pipeline.takes("spk_id")
        @sb.utils.data_pipeline.provides("spk_id")
        def speaker_pipeline(spk_id):
            yield spk_id
        
        # Add dynamic items
        sb.dataio.dataset.add_dynamic_item([dataset], audio_pipeline)
        sb.dataio.dataset.add_dynamic_item([dataset], text_pipeline)
        
        # Add speaker pipeline if column exists
        if has_speaker_info:
            sb.dataio.dataset.add_dynamic_item([dataset], speaker_pipeline)
        
        # Set output keys - include tokens if tokenizer is provided
        output_keys = ["id", "sig", "wrd"]
        if tokenizer is not None:
            output_keys.extend(["tokens", "tokens_lens"])
        if has_speaker_info:
            output_keys.append("spk_id")
        
        sb.dataio.dataset.set_output_keys([dataset], output_keys)
        
        print(f"Successfully created dataset with {len(dataset)} items", flush=True)
        return dataset
        
    finally:
        # Clean up the temporary CSV file if created
        if temp_csv and os.path.exists(temp_csv):
            try:
                os.unlink(temp_csv)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_csv}: {e}", flush=True)

def main(device="cuda"):
    # Argument parsing
    hparams_file = sys.argv[1] if len(sys.argv) > 1 else "fine-tune.yaml"
    
    # Load hyperparameters
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    output_folder = hparams.get('output_folder', './fine-tuning/outputs')
    os.makedirs(output_folder, exist_ok=True)
    hparams['output_folder'] = output_folder
    
    # Initialize tokenizer (if using your own)
    tokenizer = hparams.get('tokenizer', None)

    models_base_dir = '/scratch/bchristensen/models'
    whisper_medium_path = os.path.join(models_base_dir, 'whisper-medium')
    
    # First, load the local model
    print(f"Loading local Whisper model from {whisper_medium_path}", flush=True)
    whisper_model_data = load_whisper_model(whisper_medium_path)
    
    asr_model = SimpleWhisperASR(
        whisper_model=whisper_model_data['model'],
        whisper_processor=whisper_model_data['processor'],
        device=device
    )

    # Add to modules
    hparams['modules'] = {'encoder_model': asr_model}
    print("Configured local Whisper model for ASR", flush=True)
    
    # Set up datasets
    train_data = create_dataset(hparams['train_csv'], tokenizer, sample_size=0)
    valid_data = create_dataset(hparams['valid_csv'], tokenizer, sample_size=0)
    test_data = create_dataset(hparams['test_csv'], tokenizer, sample_size=0)
    print("Datasets created", flush=True)

    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=hparams.get('batch_size', 8), 
        collate_fn=PaddedBatch,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_data, 
        batch_size=hparams.get('batch_size', 8), 
        collate_fn=PaddedBatch
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=hparams.get('batch_size', 8), 
        collate_fn=PaddedBatch
    )
    print("Data loaders created", flush=True)

    # Initialize the ASR Brain
    asr_brain = ASRBrain(
        modules=hparams.get('modules', {}),
        opt_class=hparams.get('optimizer', torch.optim.Adam),
        hparams=hparams,
        run_opts={'device': device},
        checkpointer=hparams.get('checkpointer', None)
    )
    print("ASR Brain initialized", flush=True)

    # Fine-tune the model
    print("Starting fine-tuning...", flush=True)
    asr_brain.fit(
        epoch_counter=hparams.get('epoch_counter', None),
        train_set=train_loader,
        valid_set=valid_loader,
        progressbar=True,
        train_loader_kwargs={'collate_fn': PaddedBatch},
        valid_loader_kwargs={'collate_fn': PaddedBatch},
    )

    # Evaluate on test set
    print("Evaluating on test set with speaker analysis...", flush=True)
    asr_brain.evaluate(
        test_loader,
        test_loader_kwargs={'collate_fn': PaddedBatch},
    )

if __name__ == "__main__":
    main()