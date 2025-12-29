import torch
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import os
import tempfile
import gc
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Transcribe Manipuri audio')
parser.add_argument('--mode', choices=['full', 'chunked'], default='chunked',
                    help='Transcription mode: "full" for single-pass transcription, "chunked" for VAD-based chunking (default: chunked)')
parser.add_argument('--audio', type=str, default='~/Downloads/manipuri_test.wav',
                    help='Path to audio file')
parser.add_argument('--chunk-duration', type=int, default=30,
                    help='Target chunk duration for chunked mode in seconds (default: 30)')
args = parser.parse_args()

# Configuration
audio_file = os.path.expanduser(args.audio)
target_chunk_duration = args.chunk_duration
min_speech_gap = 0.1  # Minimum gap between speech segments (seconds)
use_chunked_mode = (args.mode == 'chunked')

# Output file paths - different files for each mode
base_name = os.path.splitext(audio_file)[0]
if use_chunked_mode:
    output_file = base_name + '_transcription_chunked.txt'
else:
    output_file = base_name + '_transcription_full.txt'

print(f"Mode: {'Chunked (VAD-based)' if use_chunked_mode else 'Full audio'}")
print(f"Output will be saved to: {output_file}")

# Load audio
print(f"\nLoading audio from: {audio_file}")
audio_np, sr = librosa.load(audio_file, sr=16000, mono=True)

# Calculate duration
duration = len(audio_np) / sr
print(f"Audio duration: {duration:.2f} seconds")


def transcribe_full(audio_np, sr, model):
    """Transcribe entire audio in one pass."""
    print("\nTranscribing full audio...")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_audio_path = tmp_file.name
        sf.write(tmp_audio_path, audio_np, sr)
    
    try:
        with torch.no_grad():
            transcriptions = model.transcribe([tmp_audio_path], batch_size=1, language_id='mni')
        
        if isinstance(transcriptions, tuple):
            transcriptions = transcriptions[0]
        
        transcription = transcriptions[0] if isinstance(transcriptions, list) else transcriptions
        return transcription
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def transcribe_chunked(audio_np, sr, model):
    """Transcribe audio in chunks using VAD for splitting."""
    from silero_vad import load_silero_vad, get_speech_timestamps
    
    print("\nLoading Silero VAD model...")
    vad_model = load_silero_vad()
    print("VAD model loaded successfully")
    
    # Convert to torch tensor for VAD
    audio_tensor = torch.from_numpy(audio_np)
    
    # Get speech timestamps from VAD
    print("\nAnalyzing audio with VAD to find natural split points...")
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        vad_model, 
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
        speech_pad_ms=30
    )
    
    if not speech_timestamps:
        print("No speech detected by VAD! Falling back to full transcription.")
        del vad_model
        gc.collect()
        return transcribe_full(audio_np, sr, model)
    
    print(f"VAD detected {len(speech_timestamps)} speech segments")
    
    # Find gaps between speech segments
    gaps = []
    for i in range(len(speech_timestamps) - 1):
        gap_start = speech_timestamps[i]['end']
        gap_end = speech_timestamps[i + 1]['start']
        gap_duration = (gap_end - gap_start) / sr
        if gap_duration >= min_speech_gap:
            midpoint = (gap_start + gap_end) // 2
            gaps.append((midpoint, gap_duration))
    
    print(f"Found {len(gaps)} gaps >= {min_speech_gap}s suitable for splitting")
    
    # Create chunks
    chunks = []
    chunk_start = 0
    target_samples = int(target_chunk_duration * sr)
    audio_len = len(audio_tensor)
    
    while chunk_start < audio_len:
        target_end = chunk_start + target_samples
        min_end = chunk_start + int(target_samples * 0.5)
        max_end = chunk_start + int(target_samples * 1.5)
        
        candidates = [(g, abs(g - target_end), d) for g, d in gaps 
                      if g > chunk_start and min_end <= g <= max_end]
        
        if candidates:
            candidates.sort(key=lambda x: (x[1], -x[2]))
            chunk_end = candidates[0][0]
        else:
            if target_end >= audio_len:
                chunk_end = audio_len
            else:
                future_gaps = [(g, d) for g, d in gaps if g > target_end]
                chunk_end = future_gaps[0][0] if future_gaps else audio_len
        
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
        if chunk_start >= audio_len:
            break
    
    print(f"\nCreated {len(chunks)} chunks for transcription:")
    for i, (start, end) in enumerate(chunks):
        print(f"  Chunk {i+1}: {start/sr:.1f}s - {end/sr:.1f}s (duration: {(end-start)/sr:.1f}s)")
    
    # Clean up VAD model
    del vad_model
    gc.collect()
    
    # Transcribe each chunk
    all_transcriptions = []
    temp_files = []
    
    try:
        for i, (start_sample, end_sample) in enumerate(chunks):
            chunk_audio = audio_np[start_sample:end_sample]
            
            chunk_start_time = start_sample / sr
            chunk_end_time = end_sample / sr
            chunk_duration = (end_sample - start_sample) / sr
            print(f"\nTranscribing chunk {i+1}/{len(chunks)}: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s ({chunk_duration:.1f}s)")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_audio_path = tmp_file.name
                temp_files.append(tmp_audio_path)
                sf.write(tmp_audio_path, chunk_audio, sr)
            
            with torch.no_grad():
                transcriptions = model.transcribe([tmp_audio_path], batch_size=1, language_id='mni')
            
            if isinstance(transcriptions, tuple):
                transcriptions = transcriptions[0]
            
            transcription = transcriptions[0] if isinstance(transcriptions, list) else transcriptions
            all_transcriptions.append(transcription)
            
            preview = transcription[:80] + "..." if len(transcription) > 80 else transcription
            print(f"  → {preview}")
            
            del chunk_audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    finally:
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        print("\nCleaned up temporary audio files")
    
    return ' '.join(all_transcriptions)


# Load the ASR model
print("\nLoading ASR model...")
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    model = model.half()
    print("Using half precision (float16) to save memory")

model.freeze()
model = model.to(device)

# Transcribe based on mode
if use_chunked_mode:
    full_transcription = transcribe_chunked(audio_np, sr, model)
else:
    full_transcription = transcribe_full(audio_np, sr, model)

# Display result
print(f"\n{'='*70}")
print("FULL TRANSCRIPTION:")
print('='*70)
print(full_transcription)
print('='*70)

# Save to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Audio file: {audio_file}\n")
    f.write(f"Duration: {duration:.2f} seconds\n")
    f.write(f"Mode: {'Chunked (VAD-based)' if use_chunked_mode else 'Full audio'}\n")
    f.write(f"Language: Manipuri (mni)\n\n")
    f.write(f"Transcription:\n{full_transcription}\n")

print(f"\n✓ Transcription saved to: {output_file}")

# Final cleanup
del audio_np
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
