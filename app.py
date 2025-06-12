import gradio as gr
import torch
import numpy as np
from pathlib import Path
import whisperx # For whisperx.load_audio()
from pyannote.core import SlidingWindowFeature, SlidingWindow, Annotation
from src.diart.blocks.whisperx_diarization import WhisperXDiarizationConfig, WhisperXDiarization
import tempfile
import os # For os.path.join
import sys # For smoke test argv parsing
import shutil # For smoke test cleanup
from unittest.mock import patch, MagicMock # For smoke test mocking
from pyannote.core import Segment # For smoke test mocking

# Define WhisperX model sizes
WHISPERX_MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']
DEFAULT_SAMPLE_RATE = 16000 # WhisperX expects 16kHz

# Helper function to format SRT timestamps
def _format_timestamp_srt(seconds: float) -> str:
    assert seconds >= 0, "SRT timestamp must be non-negative"
    millis = round(seconds * 1000)
    secs = millis // 1000
    millis %= 1000
    mins = secs // 60
    secs %= 60
    hours = mins // 60
    mins %= 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def process_audio(audio_file_path, model_size, language_code, min_speakers, max_speakers, progress=gr.Progress(track_tqdm=True)):
    error_transcript = "An error occurred. Please check the console logs."
    empty_segments_df = []
    no_file = None

    if audio_file_path is None:
        return "Error: No audio file provided. Please upload or record audio.", empty_segments_df, no_file, no_file

    # Silenced print statements for cleaner Gradio use, uncomment for debugging
    # print("\n--- Inputs received ---")
    # print(f"Audio path: {audio_file_path}")
    # print(f"Model size: {model_size}")
    # print(f"Language code: {language_code}")
    # print(f"Min speakers: {min_speakers}")
    # print(f"Max speakers: {max_speakers}")
    # print("-----------------------\n")

    # progress(0, desc="Initializing...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lang_code = language_code if language_code and language_code.strip() else None

        # Process min_speakers from Gradio input
        min_spk = None
        if min_speakers is not None: # Gradio Number might pass float e.g. 0.0 for empty if not handled by value=None
            try:
                val_min = float(min_speakers) # Handle potential float like "0.0"
                if val_min > 0:
                    min_spk = int(val_min)
                # else, if val_min is 0 or negative, it remains None (or could raise error)
            except ValueError: # If input is not convertible to float (e.g. empty string if Textbox was used)
                pass # min_spk remains None

        # Process max_speakers from Gradio input
        max_spk = None
        if max_speakers is not None:
            try:
                val_max = float(max_speakers)
                if val_max > 0:
                    max_spk = int(val_max)
            except ValueError:
                pass # max_spk remains None

        # print(f"Using device: {device}") # Silenced
        # print(f"Configuring WhisperXDiarization with: model='{model_size}', lang='{lang_code}', "
        #       f"min_speakers={min_spk}, max_speakers={max_spk}") # Silenced

        config = WhisperXDiarizationConfig(
            model_name=model_size,
            language_code=lang_code,
            device=device,
            min_speakers=min_spk,
            max_speakers=max_spk,
            _sample_rate=DEFAULT_SAMPLE_RATE
        )

        # progress(0.1, desc="Initializing Diarization Pipeline...")
        pipeline = WhisperXDiarization(config) # Model loading happens here

        # progress(0.2, desc="Loading audio...")
        try:
            audio_data = whisperx.load_audio(audio_file_path)
        except Exception as e_audio:
            print(f"Error loading audio file {audio_file_path}: {e_audio}") # Keep critical errors
            return f"Error: Could not load audio file. Details: {str(e_audio)}", empty_segments_df, no_file, no_file

        duration_s = audio_data.shape[0] / DEFAULT_SAMPLE_RATE

        if audio_data.ndim == 1:
            audio_data_swf = np.expand_dims(audio_data, axis=1)
        else:
            audio_data_swf = audio_data if audio_data.shape[1] == 1 else np.expand_dims(audio_data[:,0], axis=1)

        window = SlidingWindow(start=0.0, duration=duration_s, step=duration_s if duration_s > 0 else 1.0)
        sliding_window_feature = SlidingWindowFeature(audio_data_swf, window)

        # progress(0.3, desc="Transcribing and diarizing...")
        results = pipeline([sliding_window_feature])

        if not results or not results[0]:
            return "Error: Transcription failed or returned no results.", empty_segments_df, no_file, no_file

        full_annotation = results[0][0]

        if not isinstance(full_annotation, Annotation) or not list(full_annotation.itersegments()):
             full_transcript_text = "No speech detected or transcribed."
             segments_for_df = []
             return full_transcript_text, segments_for_df, no_file, no_file

        transcript_parts = []
        sorted_segments = sorted(list(full_annotation.itertracks(yield_label=True)), key=lambda x: x[0].start)

        for segment, speaker, text in sorted_segments:
            transcript_parts.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {speaker}: {text}")

        full_transcript_text = "\n".join(transcript_parts)
        if not full_transcript_text.strip():
            full_transcript_text = "No speech detected or transcribed."

        segments_for_df = []
        for segment, speaker, text in sorted_segments:
            segments_for_df.append({
                'Speaker': speaker,
                'Start (s)': f"{segment.start:.2f}",
                'End (s)': f"{segment.end:.2f}",
                'Text': text
            })

        txt_file_path = None
        srt_file_path = None

        # progress(0.8, desc="Saving output files...")
        try:
            temp_dir = tempfile.mkdtemp()
            txt_file_path = os.path.join(temp_dir, "output.txt")
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(full_transcript_text)

            srt_file_path = os.path.join(temp_dir, "output.srt")
            with open(srt_file_path, "w", encoding="utf-8") as f:
                for i, (segment, speaker, text) in enumerate(sorted_segments):
                    start_time_srt = _format_timestamp_srt(segment.start)
                    end_time_srt = _format_timestamp_srt(segment.end)
                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{speaker}: {text}\n\n")

        except Exception as e_file:
            print(f"Error writing output files: {e_file}") # Keep
            txt_file_path = None
            srt_file_path = None
            full_transcript_text += "\n\nError: Could not save TXT/SRT files."

        # progress(1.0, desc="Completed.")
        return full_transcript_text, segments_for_df, txt_file_path, srt_file_path

    except Exception as e:
        print(f"An error occurred during pipeline processing: {e}") # Keep
        import traceback
        traceback.print_exc()
        return f"Error during processing: {str(e)}", empty_segments_df, no_file, no_file

inputs_list = [
    gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload Audio File or Record from Microphone"),
    gr.Dropdown(choices=WHISPERX_MODEL_SIZES, value='base', label="WhisperX Model Size"),
    gr.Textbox(label="Language Code (e.g., 'en', 'es', leave blank for auto-detect)", value=""),
    gr.Number(label="Min Speakers (optional)", value=None, step=1), # minimum=1 and allow_none=True removed
    gr.Number(label="Max Speakers (optional)", value=None, step=1)  # minimum=1 and allow_none=True removed
]

iface = gr.Interface(
    fn=process_audio,
    inputs=inputs_list,
    outputs=[
        gr.Textbox(label="Full Transcript", lines=15, interactive=False),
        gr.DataFrame(headers=['Speaker', 'Start (s)', 'End (s)', 'Text'], label="Speaker Segments", interactive=False, wrap=True),
        gr.File(label="Download TXT"),
        gr.File(label="Download SRT")
    ],
    title="WhisperX Diarization UI",
    description="Transcribe and diarize audio using WhisperX. Upload an audio file or record from microphone. Optionally set language and speaker count hints.",
    allow_flagging="never"
)

if __name__ == "__main__":
    # This block is for the Gradio app launch
    # The smoke test will be in a separate file.
    iface.launch()
