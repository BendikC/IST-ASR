import os
import sys
import pandas as pd
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
from tqdm import tqdm

class ASRModelTester:
    def __init__(self, test_csv_path, output_dir='./grouped_test_results'):
        """
        Initialize the ASR Model Tester
        
        Args:
            test_csv_path (str): Path to the CSV file containing test data
            output_dir (str): Directory to save test results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Load test data
        self.test_df = pd.read_csv(test_csv_path)
        print(f"Loaded test data with {len(self.test_df)} entries", flush=True)
    
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
    
    def load_whisper_model(self, model_path):
        """Load Whisper model with performance optimizations"""
        print(f"Loading Whisper model from {model_path}", flush=True)
        
        try:
            processor = WhisperProcessor.from_pretrained(
                model_path,
                use_safetensors=False,
                local_files_only=True
            )
            
            # Force model to stay on GPU with optimal settings for inference
            model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                use_safetensors=False,
                local_files_only=True,
                torch_dtype=torch.float16,  # Use float16 to save memory and improve speed
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            # Explicitly move to GPU and set to eval mode
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()  # Set to evaluation mode
            
            # Apply inference optimizations
            model = torch.compile(model) if hasattr(torch, 'compile') else model
            
            print(f"Model loaded on: {next(model.parameters()).device}, dtype: {next(model.parameters()).dtype}")
            print("Successfully loaded model with performance optimizations", flush=True)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}", flush=True)
            raise
        
        return {
            'model': model,
            'processor': processor
        }
    
    def load_speechbrain_model(self, model_path):
        """
        Load a SpeechBrain model from a local path
        
        Args:
            model_path (str): Path to the local model directory
        
        Returns:
            Loaded ASR model
        """
        print(f"Loading SpeechBrain model from {model_path}", flush=True)
        # Add the model directory to Python path to ensure proper importing
        sys.path.insert(0, model_path)
        
        # Use the default ASR inference method
        asr_model = EncoderDecoderASR.from_hparams(
            source=model_path,
            savedir=model_path
        )
        return asr_model
    
    def transcribe_audio(self, model_info, audio_path):
        """
        Transcribe audio using a loaded ASR model with optimizations for speed
        
        Args:
            model_info (dict/object): Loaded ASR model
            audio_path (str): Path to the audio file
        
        Returns:
            Transcription of the audio
        """
        # Handle Whisper models
        if isinstance(model_info, dict) and 'processor' in model_info:
            try:
                # Get model's device once outside the function
                if not hasattr(self, '_model_device'):
                    self._model_device = next(model_info['model'].parameters()).device
                    self._model_dtype = next(model_info['model'].parameters()).dtype
                    print(f"Model is on device: {self._model_device}, dtype: {self._model_dtype}")
                
                # Load audio - use faster loading with just_waveform=True
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Apply optimizations for inference
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
                    # Process audio with processor - ensure it matches model's device and dtype
                    input_features = model_info['processor'](
                        waveform.squeeze().cpu().numpy(), 
                        sampling_rate=sample_rate, 
                        return_tensors="pt"
                    ).input_features.to(dtype=self._model_dtype)

                    input_features = input_features.to(device=self._model_device, dtype=self._model_dtype)
                    
                    # Generate transcription with optimized settings
                    generated_ids = model_info['model'].generate(
                        input_features,
                        language="en",
                        task="transcribe",
                        # Speed optimizations:
                        max_new_tokens=128,  # Limit output length appropriately
                        return_timestamps=False,  # Disable timestamp generation
                        condition_on_prev_tokens=True  # Speed up generation
                    )
                    
                    transcription = model_info['processor'].batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0]
                
                return transcription
            except Exception as e:
                print(f"Error transcribing with Whisper for {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                return ""
        
        # Handle SpeechBrain models
        else:
            try:
                # Use with inference_mode for faster processing
                with torch.inference_mode():
                    transcription = model_info.transcribe_file(audio_path)
                return transcription
            except Exception as e:
                print(f"Error transcribing with SpeechBrain for {audio_path}: {e}")
                return ""
    
    def visualize_speaker_performance(self, model_name, results):
        """
        Create visualizations of speaker performance
        
        Args:
            model_name (str): Name of the model being tested
            results (dict): Results dictionary with speaker metrics
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create speaker WER chart
            plt.figure(figsize=(12, 8))
            
            # Get speaker data
            speakers = list(results['speaker_ranking'].keys())
            wers = list(results['speaker_ranking'].values())
            
            # Sort by WER for better visualization
            speakers_sorted = [x for _, x in sorted(zip(wers, speakers))]
            wers_sorted = sorted(wers)
            
            # Plot
            plt.bar(range(len(speakers_sorted)), wers_sorted, color='skyblue')
            plt.xticks(range(len(speakers_sorted)), speakers_sorted, rotation=90)
            plt.xlabel('Speaker ID')
            plt.ylabel('Word Error Rate (WER)')
            plt.title(f'Speaker Performance for {model_name}')
            
            # Add overall WER as horizontal line
            plt.axhline(y=results['overall']['WER'], color='red', linestyle='--', 
                    label=f'Overall WER: {results["overall"]["WER"]:.2f}')
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, f'{model_name}_speaker_performance.png'))
            plt.close()
            
            # Create histogram of WER distribution
            plt.figure(figsize=(10, 6))
            plt.hist(wers, bins=20, color='lightgreen', edgecolor='black')
            plt.xlabel('Word Error Rate (WER)')
            plt.ylabel('Number of Speakers')
            plt.title(f'WER Distribution Across Speakers - {model_name}')
            plt.axvline(x=results['overall']['WER'], color='red', linestyle='--',
                    label=f'Overall WER: {results["overall"]["WER"]:.2f}')
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, f'{model_name}_wer_distribution.png'))
            plt.close()
            
            print(f"Speaker performance visualizations saved to {self.output_dir}", flush=True)
            
        except ImportError:
            print("Matplotlib not available for visualization. Skipping charts.", flush=True)
        except Exception as e:
            print(f"Error creating visualizations: {e}", flush=True)
    
    def evaluate_model(self, model_info, ref_column='wrd', max_samples=500):
        """
        Evaluate the ASR model on the test dataset with per-speaker metrics
        
        Args:
            model_info (dict/object): Loaded ASR model
            ref_column (str): Column name containing reference transcriptions
            max_samples (int): Maximum number of samples to evaluate (0 for all)
            
        Returns:
            Dictionary of performance metrics overall and by speaker
        """
        # Overall metrics
        overall_wer = ErrorRateStats()
        
        # Per-speaker metrics
        speaker_metrics = {}
        
        # Raw results for detailed analysis
        all_results = []
        
        # Limit the number of samples to evaluate
        test_df_limited = self.test_df.head(max_samples) if max_samples > 0 else self.test_df
        print(f"Evaluating on {len(test_df_limited)} out of {len(self.test_df)} samples")
        
        for _, row in tqdm(test_df_limited.iterrows(), total=len(test_df_limited), desc="Evaluating"):
            # Get speaker ID
            speaker_id = row['spk_id']
            
            # Create per-speaker metric tracker if it doesn't exist
            if speaker_id not in speaker_metrics:
                speaker_metrics[speaker_id] = ErrorRateStats()
            
            # Transcribe audio
            raw_transcription = self.transcribe_audio(model_info, row['wav'])
            raw_ref = row[ref_column]

            # Standardize the transcription and reference text
            transcription = self.standardize_text(raw_transcription)
            reference = self.standardize_text(raw_ref)
            
            # Append to overall metrics
            overall_wer.append(
                ids=[len(all_results)],
                predict=[transcription],
                target=[reference]
            )
            
            # Append to speaker-specific metrics
            speaker_metrics[speaker_id].append(
                ids=[len(all_results)],
                predict=[transcription],
                target=[reference]
            )
            
            # Store detailed results for later analysis
            all_results.append({
                'id': row.get('id', f"sample_{len(all_results)}"),
                'speaker_id': speaker_id,
                'reference': reference,
                'raw_reference': raw_ref,
                'transcription': transcription,
                'raw_transcription': raw_transcription,
                'audio_path': row['wav']
            })
        
        # Compute overall metrics
        overall_results = overall_wer.summarize()
        
        # Compute per-speaker metrics
        speaker_results = {}
        for speaker_id, metrics in speaker_metrics.items():
            summary  = metrics.summarize()
            speaker_results[speaker_id] = summary
        
        # Sort speakers by performance (WER)
        sorted_speakers = sorted(
            speaker_results.items(),
            key=lambda x: x[1]['WER']
        )
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(self.output_dir, f'{model_info.__class__.__name__}_detailed_results.csv'), index=False)
        
        # Combine all results
        final_results = {
            'overall': overall_results,
            'per_speaker': speaker_results,
            'speaker_ranking': {spk: results['WER'] for spk, results in sorted_speakers},
            'num_speakers': len(speaker_results),
            'detailed_results_path': os.path.join(self.output_dir, f'{model_info.__class__.__name__}_detailed_results.csv')
        }
        
        return final_results
    
    def run_tests(self, models):
        """
        Run tests on multiple models with per-speaker analysis
        
        Args:
            models (dict): Dictionary of models to test
        """
        results = {}
        
        for model_name, model_info in models.items():
            print(f"\nTesting model: {model_name}", flush=True)
            try:
                # Clear GPU cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory before evaluation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # Evaluate model
                model_results = self.evaluate_model(model_info, max_samples=0)
                results[model_name] = model_results
                
                # Clear GPU cache after evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # Save results to files
                # Overall results
                with open(os.path.join(self.output_dir, f'{model_name}_overall_results.txt'), 'w') as f:
                    f.write(f"OVERALL METRICS:\n")
                    for metric, value in model_results['overall'].items():
                        f.write(f"{metric}: {value}\n")
                    
                    f.write(f"\nSPEAKER ANALYSIS:\n")
                    f.write(f"Total speakers: {model_results['num_speakers']}\n\n")
                    
                    # Write speaker ranking (best to worst)
                    f.write("SPEAKER RANKING BY WER (BEST TO WORST):\n")
                    for speaker, wer in model_results['speaker_ranking'].items():
                        f.write(f"Speaker {speaker}: {wer:.2f}\n")
                
                # Per-speaker detailed results
                with open(os.path.join(self.output_dir, f'{model_name}_speaker_results.txt'), 'w') as f:
                    f.write("DETAILED SPEAKER METRICS:\n\n")
                    for speaker, speaker_metrics in model_results['per_speaker'].items():
                        f.write(f"Speaker {speaker}:\n")
                        for metric, value in speaker_metrics.items():
                            f.write(f"  {metric}: {value}\n")
                        f.write("\n")
                
                # Print summary results
                print(f"\nResults for {model_name}:", flush=True)
                print(f"Overall WER: {model_results['overall']['WER']:.2f}", flush=True)
                print(f"Number of speakers: {model_results['num_speakers']}", flush=True)
                
                # Print top 3 and bottom 3 speakers by WER
                sorted_speakers = list(model_results['speaker_ranking'].items())
                
                print("\nBest performing speakers (lowest WER):", flush=True)
                for speaker, wer in sorted_speakers[:3]:
                    print(f"  Speaker {speaker}: {wer:.2f}", flush=True)
                
                print("\nWorst performing speakers (highest WER):", flush=True)
                for speaker, wer in sorted_speakers[-3:]:
                    print(f"  Speaker {speaker}: {wer:.2f}", flush=True)
                
                # Save visualization of speaker performance
                self.visualize_speaker_performance(model_name, model_results)
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
            
        return results

def main():
    print("Starting ASR Model Tester...", flush=True)
    # Path to your test CSV file
    test_csv_path = '/home/bchristensen/random_split/test.csv'
    
    # Base directory for models
    models_base_dir = '/scratch/bchristensen/models'
    
    # Initialize tester
    tester = ASRModelTester(test_csv_path)
    
    # Load models to test
    models = {
        'whisper_medium': tester.load_whisper_model(
            os.path.join(models_base_dir, 'whisper-medium')
        ),
        'whisper_large_v3_turbo': tester.load_whisper_model(
            os.path.join(models_base_dir, 'whisper-large-v3')
        ),
        # 'speechbrain_librispeech': tester.load_speechbrain_model(
        #     os.path.join(models_base_dir, 'asr-crdnn-rnnlm-librispeech')
        # )
    }
    
    # Run tests
    results = tester.run_tests(models)

if __name__ == "__main__":
    main()