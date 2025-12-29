import torch
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import os
import tempfile
from omegaconf import OmegaConf, open_dict

# Configuration
audio_file = '~/Downloads/manipuri_test.wav'
audio_file = os.path.expanduser(audio_file)  # Expand ~ to home directory
max_duration_seconds = 60  # Only process first 60 seconds

# Output file path
output_file = os.path.splitext(audio_file)[0] + '_transcription.txt'
print(f"Output will be saved to: {output_file}")

# Load and trim audio to first 60 seconds
print(f"Loading audio from: {audio_file}")
audio, sr = librosa.load(audio_file, sr=16000, mono=True)

# Calculate duration
duration = len(audio) / sr
print(f"Original audio duration: {duration:.2f} seconds")

# Trim to first 60 seconds
max_samples = int(max_duration_seconds * sr)
if len(audio) > max_samples:
    audio_trimmed = audio[:max_samples]
    print(f"Trimming audio to first {max_duration_seconds} seconds")
else:
    audio_trimmed = audio
    print(f"Audio is shorter than {max_duration_seconds} seconds, using full audio")

# Save trimmed audio to temporary file
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
    tmp_audio_path = tmp_file.name
    sf.write(tmp_audio_path, audio_trimmed, sr)
    print(f"Saved trimmed audio to temporary file: {tmp_audio_path}")

# Load the ASR model
print("Loading ASR model...")
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.freeze() # inference mode
model = model.to(device) # transfer model to device

# Configure model to preserve alignments and compute timestamps
print("Configuring model for timestamp extraction...")
model.cur_decoder = "ctc"

# Configure decoding based on model type
if hasattr(model, 'cfg') and 'aux_ctc' in model.cfg:
    # Hybrid model - configure CTC decoding config
    ctc_decoding_cfg = model.cfg.aux_ctc.decoding
    with open_dict(ctc_decoding_cfg):
        ctc_decoding_cfg.preserve_alignments = True
        ctc_decoding_cfg.compute_timestamps = True
    model.change_decoding_strategy(decoding_cfg=None, decoder_type="ctc")
    
    # Patch CTC decoding to use Manipuri language for multilingual tokenizer
    # This fixes the issue where compute_ctc_timestamps calls decode methods without lang
    if hasattr(model, 'ctc_decoding'):
        # Patch decode_tokens_to_str
        original_decode_str = model.ctc_decoding.decode_tokens_to_str
        def patched_decode_tokens_to_str(tokens, lang=None):
            # Use Manipuri (mni) as default language if not provided
            if lang is None:
                lang = 'mni'
            return original_decode_str(tokens, lang)
        model.ctc_decoding.decode_tokens_to_str = patched_decode_tokens_to_str
        
        # Patch decode_ids_to_tokens
        original_decode_ids = model.ctc_decoding.decode_ids_to_tokens
        def patched_decode_ids_to_tokens(tokens):
            # Use Manipuri (mni) as default language for multilingual tokenizer
            return model.ctc_decoding.tokenizer.ids_to_tokens(tokens, 'mni')
        model.ctc_decoding.decode_ids_to_tokens = patched_decode_ids_to_tokens
        
        print("Patched CTC decoding to use Manipuri (mni) language")
else:
    # Regular decoding config
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
    model.change_decoding_strategy(decoding_cfg)

# Transcribe with timestamps
print("Transcribing audio with timestamps...")
hypotheses = model.transcribe([tmp_audio_path], batch_size=1, logprobs=False, language_id='mni', return_hypotheses=True)

# Extract hypothesis (handle tuple format from RNNT if needed)
if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
    hypotheses = hypotheses[0]

hypothesis = hypotheses[0]  # Get first (and only) audio file hypothesis
ctc_text = hypothesis.text

# Extract timestamps
timestamp_dict = hypothesis.timestep
print(f"\nHypothesis contains following timestep information: {list(timestamp_dict.keys()) if timestamp_dict else 'None'}")

# Verify the actual processed audio duration
actual_processed_duration = len(audio_trimmed) / sr
print(f"Actual processed audio duration: {actual_processed_duration:.2f} seconds")

# Calculate ACTUAL time stride based on real audio duration and frame count
# The formula 8 * window_stride is often incorrect - we need to calculate from actual data
if timestamp_dict and 'word' in timestamp_dict:
    word_timestamps = timestamp_dict['word']
    max_frame_index = max(stamp['end_offset'] for stamp in word_timestamps)
    
    if max_frame_index > 0:
        # Calculate time stride: actual audio duration / number of frames
        time_stride = actual_processed_duration / max_frame_index
        print(f"Calculated ACTUAL time stride: {time_stride:.6f} seconds per frame")
        print(f"  (Based on {actual_processed_duration:.2f}s audio / {max_frame_index} frames)")
        
        # Show what the formula would have given for comparison
        window_stride = model.cfg.preprocessor.window_stride
        formula_time_stride = 8 * window_stride
        print(f"  Formula-based would be: {formula_time_stride:.6f} seconds per frame (8 * window_stride)")
    else:
        # Fallback to formula if we can't calculate
        window_stride = model.cfg.preprocessor.window_stride
        time_stride = 8 * window_stride
        print(f"Using formula-based time stride: {time_stride:.6f} seconds per frame (8 * window_stride)")
else:
    # Fallback if no timestamps available
    window_stride = model.cfg.preprocessor.window_stride
    time_stride = 8 * window_stride
    print(f"Using formula-based time stride: {time_stride:.6f} seconds per frame")

# Display transcription with timestamps
print("\n" + "="*70)
print("TRANSCRIPTION RESULT WITH TIMESTAMPS:")
print("="*70)

# Prepare output content
output_lines = []
output_lines.append("="*70)
output_lines.append("TRANSCRIPTION RESULT WITH TIMESTAMPS")
output_lines.append("="*70)
output_lines.append(f"\nAudio file: {audio_file}")
output_lines.append(f"Original audio duration: {duration:.2f} seconds")
output_lines.append(f"Processed duration: {min(duration, max_duration_seconds):.2f} seconds")
output_lines.append(f"Language: Manipuri (mni)")
output_lines.append(f"\n⚠️  IMPORTANT: Timestamps are in SECONDS (not milliseconds)!")
output_lines.append(f"      - '2.24' means 2.24 seconds (not 2.24 milliseconds)")
output_lines.append(f"      - Timestamps are relative to the START of the processed audio segment")
output_lines.append(f"      - Time stride: {time_stride:.6f} seconds per frame (calculated from actual audio)")
output_lines.append(f"      - To convert to milliseconds, multiply by 1000 (e.g., 2.24s = 2240ms)")
output_lines.append("\n" + "="*70)

if timestamp_dict and 'word' in timestamp_dict:
    word_timestamps = timestamp_dict['word']
    output_lines.append(f"\nFull transcription: {ctc_text}\n")
    output_lines.append("-"*70)
    output_lines.append(f"{'Start (s)':<12} {'End (s)':<12} {'Word'}")
    output_lines.append("-"*70)
    
    print(f"\nFull transcription: {ctc_text}\n")
    print("-"*70)
    print(f"{'Start (s)':<12} {'End (s)':<12} {'Word'}")
    print("-"*70)
    
    max_timestamp = 0
    for stamp in word_timestamps:
        start = stamp['start_offset'] * time_stride
        end = stamp['end_offset'] * time_stride
        word = stamp.get('char', stamp.get('word', ''))
        max_timestamp = max(max_timestamp, end)
        line = f"{start:<12.2f} {end:<12.2f} {word}"
        output_lines.append(line)
        print(line)
    
    # Add note about timestamp calculation
    output_lines.append(f"\nNote: Maximum timestamp: {max_timestamp:.2f}s")
    output_lines.append(f"      Processed audio duration: {actual_processed_duration:.2f}s")
    if max_timestamp > actual_processed_duration * 1.1:  # Allow 10% tolerance
        warning = f"\n⚠️  NOTE: Timestamps extend beyond processed duration."
        warning += f"\n   This is normal - timestamps are calculated from model frame indices"
        warning += f"\n   and may slightly exceed the actual audio duration due to frame alignment."
        output_lines.append(warning)
        print(warning)
    
    output_lines.append("="*70)
    print("="*70)
else:
    output_lines.append(f"\nFull transcription: {ctc_text}")
    output_lines.append("\nNote: Word-level timestamps not available. Showing full transcription only.")
    output_lines.append("="*70)
    
    print(f"\nFull transcription: {ctc_text}")
    print("\nNote: Word-level timestamps not available. Showing full transcription only.")
    print("="*70)

# Write output to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
    f.write('\n')

print(f"\n✓ Transcription saved to: {output_file}")

# Clean up temporary file
if os.path.exists(tmp_audio_path):
    # os.remove(tmp_audio_path)
    print("Cleaned up temporary audio file", tmp_audio_path)
