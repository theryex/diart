import whisperx
import torchaudio

# Define the audio file path
audio_file = "tests/data/audio/sample.wav"

# Define compute type based on PyTorch version (CPU or CUDA)
# Since we installed the CPU version of PyTorch, we'll use "int8" for CPU
device = "cpu"
compute_type = "int8" # or "float16" for GPU, "int8" for CPU

# 1. Load the Whisper model
# Using "tiny" model for faster processing in this test
# Other options: "base", "small", "medium", "large-v1", "large-v2", "large-v3"
try:
    print(f"Loading Whisper model 'tiny' for device '{device}' with compute_type '{compute_type}'...")
    # Note: WhisperX might download the model on first use if not cached
    model = whisperx.load_model("tiny", device, compute_type=compute_type)
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit(1)

# 2. Load audio
try:
    print(f"Loading audio file: {audio_file}...")
    audio, sample_rate = torchaudio.load(audio_file)
    # Whisper expects mono audio, ensure it's mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample if necessary (Whisper expects 16kHz) - torchaudio handles this internally if needed by model
    print(f"Audio loaded. Sample rate: {sample_rate}, Shape: {audio.shape}")
except Exception as e:
    print(f"Error loading audio file: {e}")
    exit(1)

# 3. Transcribe the audio
try:
    print("Transcribing audio...")
    # The input to model.transcribe should be the path to the audio file or the loaded audio tensor
    # For pre-loaded audio, it expects a numpy array or torch tensor.
    # WhisperX's transcribe function internally handles the conversion if you pass the waveform.
    # Let's pass the waveform directly.
    result = model.transcribe(audio.squeeze().numpy()) # Squeeze to make it 1D if it's [1, N]
    print("Transcription completed.")
except Exception as e:
    print(f"Error during transcription: {e}")
    exit(1)

# 4. Print the transcript
print("\nTranscription Result:")
if result and "segments" in result:
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
    # Also print the full text if available
    if "text" in result:
        print(f"\nFull Text: {result['text']}")
else:
    print("No segments found in transcription result or result is empty.")
    print(f"Raw result: {result}")
