import sys
from pathlib import Path
import numpy as np
import os
import shutil
import tempfile # Should be imported in app.py, but good to have for this script too if needed
from unittest.mock import patch, MagicMock

# Add src and root to Python path to allow importing app and its dependencies
# Assuming smoke_test_runner.py is in the root of the repository.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imports from app.py or other modules
# These will be resolved based on PYTHONPATH and the structure of app.py
# We expect app.py to define process_audio and DEFAULT_SAMPLE_RATE
# and for _format_timestamp_srt to be defined or imported there.
# The other classes are imported from their respective libraries.
from app import process_audio, DEFAULT_SAMPLE_RATE
# _format_timestamp_srt is defined within app.py, so not directly imported here unless refactored.
# For the smoke test, the SRT content check will rely on process_audio using its internal helper.

from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
import gradio as gr # For gr.Progress, or use a mock

# Mock gr.Progress for non-Gradio context
class MockProgress:
    def __init__(self, track_tqdm=False):
        self.last_progress = None
        self.last_desc = None
        if track_tqdm: # Just to match signature
            pass
    def __call__(self, progress_value, desc=None): # Renamed progress to progress_value
        self.last_progress = progress_value
        self.last_desc = desc
        print(f"MockProgress: {self.last_progress}, Desc: {self.last_desc}") # Use instance vars
    def __enter__(self): # For potential 'with' usage if Gradio does that
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass


def run_smoke_test_logic():
    print("--- Running Smoke Test via smoke_test_runner.py ---")
    sys.stdout.flush()

    sample_audio_path_str = "tests/data/audio/sample.wav"
    sample_audio = Path(sample_audio_path_str)

    if not sample_audio.exists():
        print(f"SMOKE TEST: {sample_audio} not found. Creating dummy audio.")
        sample_audio.parent.mkdir(parents=True, exist_ok=True)
        sr = DEFAULT_SAMPLE_RATE
        duration_s = 2
        dummy_audio_data = np.random.uniform(-0.1, 0.1, sr * duration_s).astype(np.float32)
        try:
            import soundfile as sf
            sf.write(sample_audio_path_str, dummy_audio_data, sr)
            print(f"SMOKE TEST: Dummy {sample_audio} created.")
        except Exception as e:
            print(f"SMOKE TEST: Could not create dummy audio {sample_audio}: {e}. Test might fail.")
            return # Cannot proceed without audio

    model_size = 'tiny'
    language_code = None
    min_speakers = None
    max_speakers = None

    mock_progress_instance = MockProgress()

    mock_annotation = Annotation(uri="smoke_test_anno")
    # Storing text as the third element of the tuple for a track is not standard for pyannote.
    # pyannote.Annotation usually stores the label (speaker ID) as the value.
    # The process_audio function formats its output based on iterating tracks and getting a label.
    # Let's assume the WhisperXDiarization pipeline correctly populates the annotation
    # such that `itertracks(yield_label=True)` yields (segment, speaker_id, text_content).
    mock_annotation[Segment(0.123, 0.845), "SPEAKER_00"] = "Hello"
    mock_annotation[Segment(1.001, 1.599), "SPEAKER_01"] = "world"

    dummy_swf_data = np.array([[0.0]], dtype=np.float32)
    dummy_duration = 0.01
    dummy_sliding_window = SlidingWindow(start=0.0, duration=dummy_duration, step=dummy_duration)
    dummy_swf = SlidingWindowFeature(dummy_swf_data, dummy_sliding_window)
    mock_pipeline_output = [(mock_annotation, dummy_swf)]

    # Patching targets in 'app' module now, as process_audio uses them from its own module scope
    # This assumes app.py can be imported and its members (like WhisperXDiarization, whisperx) can be patched.
    with patch('app.WhisperXDiarization') as MockedPipeline, \
         patch('app.whisperx.load_audio') as mock_load_audio:

        dummy_loaded_audio = np.random.uniform(-0.1, 0.1, DEFAULT_SAMPLE_RATE * 2).astype(np.float32)
        mock_load_audio.return_value = dummy_loaded_audio

        mock_pipeline_instance = MockedPipeline.return_value
        mock_pipeline_instance.__call__.return_value = mock_pipeline_output

        print(f"SMOKE TEST: Calling process_audio with: {sample_audio_path_str}, model: {model_size}...")
        results = process_audio(
            sample_audio_path_str, model_size, language_code,
            min_speakers, max_speakers, progress=mock_progress_instance
        )

    transcript, segments_df, txt_path, srt_path = results

    print("\n--- Smoke Test Results ---")
    sys.stdout.flush() # Ensure this gets printed before potential later errors
    print(f"Transcript type: {type(transcript)}")
    print(f"Transcript (first 100 chars):\n{transcript[:100]}...")

    print(f"\nSegments DataFrame type: {type(segments_df)}")
    if isinstance(segments_df, list):
        print(f"Number of segments: {len(segments_df)}")
        if segments_df:
            print(f"First segment: {segments_df[0]}")
    else:
        print(f"Segments DataFrame is not a list: {segments_df}")

    temp_files_to_clean = []
    temp_dirs_to_clean = set() # Use a set to avoid trying to delete the same dir multiple times

    for label, file_path_obj in [("TXT", txt_path), ("SRT", srt_path)]:
        print(f"\n{label} File Path: {file_path_obj}")
        if file_path_obj:
            file_path = Path(file_path_obj)
            if file_path.exists():
                print(f"{label} file exists.")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"{label} file size: {len(content)} bytes")
                    print(f"{label} file content (first 150 chars):\n{content[:150]}...")
                temp_files_to_clean.append(file_path)
                temp_dirs_to_clean.add(file_path.parent) # Store parent dir for cleanup
            else:
                print(f"{label} file does NOT exist.")
        else:
            print(f"{label} file path is None.")

    # Cleanup
    for file_path in temp_files_to_clean:
        try:
            os.remove(file_path)
            print(f"SMOKE TEST: Cleaned up temp file {file_path}")
        except OSError as e:
            print(f"SMOKE TEST: Error cleaning up temp file {file_path}: {e}")

    for temp_dir in temp_dirs_to_clean:
        try:
            if not any(temp_dir.iterdir()): # Check if directory is empty
                os.rmdir(temp_dir)
                print(f"SMOKE TEST: Cleaned up empty temp directory {temp_dir}")
            else:
                print(f"SMOKE TEST: Temp directory {temp_dir} not empty, not removing.")
        except OSError as e:
            print(f"SMOKE TEST: Error cleaning up temp directory {temp_dir}: {e}")

    print("--- Smoke Test Finished ---")
    sys.stdout.flush()

if __name__ == "__main__":
    run_smoke_test_logic()
