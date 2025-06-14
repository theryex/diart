from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Text, Any, Tuple

import torch
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate

# Import whisperx and other necessary modules
import whisperx
import numpy as np
from pyannote.core import Segment
from whisperx.diarize import DiarizationPipeline as WXDiarizationPipeline # Correct import
import os
from dotenv import load_dotenv
import pandas as pd # Added for DataFrame check

from .base import PipelineConfig, Pipeline, HyperParameter
# Import AudioLoader if needed for duration/padding, similar to PipelineConfig
from ..audio import AudioLoader # For get_file_padding, if we keep that method as is
from .. import utils


@dataclass
class WhisperXDiarizationConfig(PipelineConfig):
    """Configuration for the WhisperX diarization pipeline.

    Parameters
    ----------
    model_name : str, optional
        Name of the WhisperX model to use (e.g., "tiny", "base", "large-v2").
        Defaults to "base".
    language_code : str, optional
        Language code for transcription (e.g., "en", "es").
        If None, WhisperX will attempt to auto-detect the language. Defaults to None.
    device : torch.device, optional
        Device to run the model on (e.g., "cuda", "cpu").
        Defaults to "cuda" if available, otherwise "cpu".
    compute_type : str, optional
        Compute type for the model (e.g., "float16", "int8").
        Defaults to "float16" if CUDA is available, "int8" otherwise.
    batch_size : int, optional
        Batch size for transcription. Defaults to 16.
    hf_token : str | bool, optional
        HuggingFace token for models that require authentication.
        Defaults to True to use token from environment or login.
    align_model : str, optional
        Name of the alignment model. Defaults to None.
    interpolate_method : str, optional
        Method for interpolating timestamps. Defaults to "nearest".
    return_char_alignments : bool, optional
        Whether to return character-level alignments. Defaults to False.

    duration : float, optional
        Duration of audio chunks processed by the pipeline in seconds.
        Defaults to 5.0. This might be less relevant if WhisperX processes whole files
        or larger segments, but kept for consistency with diart's PipelineConfig.
    step : float, optional
        Step between consecutive audio chunks in seconds.
        Defaults to 0.5. Similar to duration, its relevance depends on WhisperX's processing.
    latency : float, optional
        Pipeline latency in seconds. Defaults to `step`.
    sample_rate : int, optional
        Sample rate of the input audio. Defaults to 16000.
    """
    model_name: str = "base"
    language_code: str | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    compute_type: str | None = None # Will be set based on device in __post_init__
    batch_size: int = 16
    hf_token: str | None = None # Default to None, will be loaded from env if not provided
    align_model: str | None = None
    diarization_model_name: str = "pyannote/speaker-diarization-3.1"
    interpolate_method: str = "nearest"
    return_char_alignments: bool = False

    # Standard diart pipeline parameters
    _duration: float = 5.0
    _step: float = 0.5
    _latency: float | None = None  # Will be set based on step in __post_init__
    _sample_rate: int = 16000
    # Minimum audio duration for WhisperX processing, in seconds.
    # WhisperX works better on longer segments.
    min_audio_duration: float = 1.0
    # Speaker count constraints for diarization
    min_speakers: int | None = None
    max_speakers: int | None = None

    def __post_init__(self):
        # load_dotenv() # Removed: .env loading is now expected to happen at app startup
        if self.hf_token is None or self.hf_token == "":
            self.hf_token = os.getenv("HF_TOKEN")
            # If still None, it means it wasn't in constructor or environment (already loaded by app)
            # WhisperX/pyannote will handle None hf_token (might use global cache or fail for gated models)

        if self.compute_type is None:
            self.compute_type = "float16" if self.device.type == "cuda" else "int8"
        if self._latency is None:
            self._latency = self._step

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        # Ensure _latency is not None, which should be handled by __post_init__
        return self._latency if self._latency is not None else self._step


    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # get_file_padding might need adjustment if WhisperX handles files differently
    # For now, keeping it similar to base PipelineConfig
    def get_file_padding(self, filepath) -> Tuple[float, float]:
        file_duration = AudioLoader(self.sample_rate, mono=True).get_duration(filepath)
        right = utils.get_padding_right(self.latency, self.step)
        left = utils.get_padding_left(file_duration + right, self.duration)
        return left, right


class WhisperXDiarization(Pipeline):
    """
    WhisperX diarization pipeline.
    Combines WhisperX transcription and alignment with speaker diarization.
    """

    def __init__(self, config: WhisperXDiarizationConfig | None = None):
        self._config = WhisperXDiarizationConfig() if config is None else config

        try:
            # Silencing prints for tests
            # print(f"Loading WhisperX ASR model '{self.config.model_name}' "
            #       f"on device '{self.config.device}' with compute_type '{self.config.compute_type}'")
            self.wx_model = whisperx.load_model(
                self.config.model_name,
                str(self.config.device), # Pass device as string
                compute_type=self.config.compute_type,
                language=self.config.language_code,
                #asr_options={"initial_prompt": ...} # Add if needed
                #download_root= # Specify if custom model download path is needed
            )
            # print("WhisperX ASR model loaded successfully.")

            if self.config.language_code is None:
                # print("No language code provided for alignment model. Using ASR model's detected language.")
                self.align_model = None
                self.align_metadata = None # Ensure metadata is also None initially
            else:
                # print(f"Loading WhisperX alignment model for language '{self.config.language_code}'")
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.config.language_code,
                    device=self.config.device,
                    # model_name= # Specify if custom alignment model name is needed
                    # download_root= # Specify if custom model download path is needed
                )
                # print("WhisperX alignment model loaded successfully.")

            # print("Initializing WhisperX DiarizationPipeline...")
            self.diarize_model = WXDiarizationPipeline(
                model_name=self.config.diarization_model_name,
                use_auth_token=self.config.hf_token, # hf_token can be True, False, or a string token
                device=str(self.config.device) # Ensure device is passed as string
                # min_speakers and max_speakers are passed during the call, not init for DiarizationPipeline
            )
            # print("WhisperX DiarizationPipeline initialized successfully.")

        except Exception as e:
            # print(f"Error during WhisperX model loading: {e}")
            # Depending on desired behavior, either raise e or set models to None and handle in __call__
            self.wx_model = None
            self.align_model = None
            self.diarize_model = None
            # Consider raising the exception to signal a critical failure
            raise RuntimeError(f"Failed to initialize WhisperX models: {e}")

        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return WhisperXDiarizationConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        # Using DiarizationErrorRate as a common metric for diarization tasks
        return DiarizationErrorRate(collar=0.0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[HyperParameter]:
        # For now, returning an empty list as WhisperX parameters are not typically tuned
        # in the same way as pyannote's HMM-based pipelines.
        # This could be expanded if specific WhisperX parameters are found to be tunable
        # in a similar hyperparameter optimization context.
        return []

    @property
    def config(self) -> WhisperXDiarizationConfig:
        return self._config

    def reset(self):
        """Reset pipeline to initial state."""
        # This method would re-initialize any stateful components.
        # For WhisperX, this might mean clearing caches or resetting model states if applicable.
        # print("WhisperXDiarization pipeline reset.")
        # DiarizationPipeline in WhisperX does not have an explicit reset method in the same way.
        # It's typically stateless per call or manages its state internally for a given audio.
        # If we were to buffer audio across calls, that buffering logic would be reset here.
        self.buffered_audio_chunks = [] # Example if buffering needed
        self.current_audio_duration = 0.0
        pass

    def set_timestamp_shift(self, shift: float):
        """Set a timestamp shift for the output annotations.
        Not typically used directly with WhisperX if it processes whole files,
        but included for compatibility with diart's streaming concepts.
        """
        # This would need careful consideration on how it interacts with WhisperX's outputs.
        # If processing is done chunk by chunk, and annotations are relative to the chunk start,
        # this shift might be applied when creating the final Annotation object.
        # However, WhisperX usually processes longer segments, so the impact of this needs thought.
        self.timestamp_shift = shift
        # print(f"Timestamp shift set to {shift}s. This will be applied to output annotations.")


    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[Tuple[Annotation, SlidingWindowFeature]]:
        """
        Perform diarization using WhisperX on a sequence of audio waveforms.
        NOTE: WhisperX is generally designed for processing longer audio files rather than
        short, sequential chunks. This implementation will attempt to concatenate
        the input chunks to form a more suitable segment for WhisperX.
        The output will be a single annotation for the concatenated audio, associated
        with the last waveform chunk in the sequence for consistency with diart's API,
        though it represents the whole processed segment.

        Parameters
        ----------
        waveforms : Sequence[SlidingWindowFeature]
            A sequence of audio chunks. These will be concatenated.

        Returns
        -------
        Sequence[Tuple[Annotation, SlidingWindowFeature]]
            A list containing a single tuple:
            - An Annotation object with speaker diarization results for the concatenated audio.
            - The last SlidingWindowFeature from the input sequence.
            If multiple calls are made, previously buffered audio is included.
        """
        if not self.wx_model or not self.diarize_model: # self.align_model can be None initially
            # print("WhisperX models not loaded properly. Cannot process audio.")
            return [(Annotation(uri="error_models_not_loaded"), w) for w in waveforms]

        if not waveforms:
            return []

        # Concatenate new waveforms with any buffered audio
        current_waveform_data = [torch.from_numpy(w.data.T) for w in waveforms]

        # If there's buffered audio, prepend it.
        # This simple concatenation assumes mono audio and consistent channel layout.
        # WhisperX expects a 1D numpy array or a path to an audio file.
        # We'll handle conversion to a single audio segment.

        # For simplicity in this iteration, we'll process the concatenation of the *current* batch of waveforms.
        # A more robust solution would handle continuous buffering and processing based on duration.

        # Convert sequence of SlidingWindowFeatures to a single audio segment (numpy array)
        # Assuming mono audio, data is (samples, 1) or (samples,)
        # We need to make sure it's a flat numpy array for WhisperX
        full_audio_data_list = []
        total_samples = 0
        for wf_feat in waveforms:
            # wf_feat.data is typically (num_samples, num_channels)
            # Whisper expects a flat array (mono)
            if wf_feat.data.ndim == 2 and wf_feat.data.shape[1] == 1: # Mono
                full_audio_data_list.append(wf_feat.data.squeeze())
            elif wf_feat.data.ndim == 1: # Already mono
                full_audio_data_list.append(wf_feat.data)
            else: # Stereo or more, average to mono
                full_audio_data_list.append(np.mean(wf_feat.data, axis=1))
            total_samples += wf_feat.data.shape[0]

        if not full_audio_data_list:
            return [(Annotation(uri="empty_input"), waveforms[-1])] # Or handle as error

        full_audio_np = np.ascontiguousarray(np.concatenate(full_audio_data_list).astype(np.float32))

        # Ensure audio meets minimum duration for WhisperX, if specified
        current_audio_duration_sec = total_samples / self.config.sample_rate
        if current_audio_duration_sec < self.config.min_audio_duration:
             # Not enough audio data yet, buffer or return empty
            # print(f"Audio duration {current_audio_duration_sec:.2f}s is less than minimum {self.config.min_audio_duration:.2f}s. Skipping.")
            return [(Annotation(uri="audio_too_short"), waveforms[-1])]


        try:
            # 1. Transcribe
            # print(f"Transcribing audio segment of {current_audio_duration_sec:.2f}s...")
            asr_result = self.wx_model.transcribe(
                full_audio_np,
                batch_size=self.config.batch_size,
            )
            # print("Transcription finished.")

            current_align_model = self.align_model
            current_align_metadata = self.align_metadata

            if current_align_model is None or (self.config.language_code is None and current_align_metadata is None):
                detected_language = asr_result.get("language")
                if detected_language:
                    # print(f"ASR detected language: {detected_language}. Loading alignment model...")
                    try:
                        current_align_model, current_align_metadata = whisperx.load_align_model(
                            language_code=detected_language, device=self.config.device
                        )
                    except Exception as e:
                        # print(f"Failed to load alignment model for detected language {detected_language}: {e}")
                        # Fallback: create a basic annotation or return error
                        return [(Annotation(uri="error_loading_align_model"), waveforms[-1])]
                else:
                    # print("Could not detect language, and no language code provided for alignment. Skipping alignment.")
                    return [(Annotation(uri="error_no_language_for_align"), waveforms[-1])]

            if not current_align_model or not current_align_metadata:
                 # print("Alignment model or metadata not available. Skipping alignment.")
                 return "Error: Alignment model or metadata not available.", [], None, None


            # 2. Align
            if not asr_result["segments"]:
                # print("No segments found by ASR. Skipping alignment and diarization.")
                return "No segments found by ASR. Skipping alignment and diarization.", [], None, None

            # print("Aligning transcript...")
            aligned_result = whisperx.align(
                asr_result["segments"],
                current_align_model,
                current_align_metadata,
                full_audio_np,
                self.config.device,
                return_char_alignments=self.config.return_char_alignments,
            )
            # print("Alignment finished.")

            # 3. Diarize
            # print("Performing speaker diarization...")
            audio_input_for_diarize = {"waveform": torch.from_numpy(full_audio_np).float(), "sample_rate": self.config.sample_rate}

            diarization_result = self.diarize_model(
                audio_input_for_diarize,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            # print("Diarization finished.")

            # Convert diarization_result to pyannote.core.Annotation if it's a DataFrame
            di_annotation = Annotation(uri="whisperx_diarization_output")
            if isinstance(diarization_result, pd.DataFrame) and not diarization_result.empty:
                for _, row in diarization_result.iterrows():
                    start_time = float(row['start'])
                    end_time = float(row['end'])
                    speaker_label = str(row['speaker']) # Ensure speaker label is a string
                    segment = Segment(start_time, end_time)
                    di_annotation[segment, speaker_label] = speaker_label # Store speaker as track & label
            elif isinstance(diarization_result, Annotation):
                # If it's already an annotation, use it directly, perhaps copy to ensure URI
                di_annotation = diarization_result.rename_labels(copy=True)
                di_annotation.uri = "whisperx_diarization_output"

            # 4. Assign word speakers
            # Use the converted di_annotation here
            if di_annotation.get_timeline().duration() > 0:
                # print("Assigning word speakers...")
                final_result = whisperx.assign_word_speakers(di_annotation, aligned_result)
                # print("Word speaker assignment finished.")
            else:
                # print("No speaker turns from diarization pipeline or empty result. Using aligned result without speaker info.")
                final_result = aligned_result

            # 5. Convert final_result (which now includes speaker assignments on word level)
            # to the output pyannote.core.Annotation.
            # The final_result["segments"] should contain the text and speaker for each segment.
            output_annotation = Annotation(uri="diart_whisperx_output")

            for segment_data in final_result.get("segments", []):
                start = segment_data.get("start")
                end = segment_data.get("end")

                if start is None or end is None:
                    continue # Skip segments without valid start/end times

                start_time = float(start) + self.timestamp_shift
                end_time = float(end) + self.timestamp_shift
                if end_time < start_time: end_time = start_time

                speaker_label = segment_data.get("speaker", "SPEAKER_UNKNOWN") # Changed default
                text = segment_data.get("text", "").strip()

                # pyannote.core.Segment(start, end)
                segment = Segment(start_time, end_time)
                output_annotation[segment, speaker_label] = text # Store text as label, or speaker as label
                # Or, more conventionally for diarization:
                # output_annotation[segment] = speaker_label
                # And perhaps store text separately if needed, or in segment.metadata if pyannote supports it.
                # For now, using speaker as track label and text as the value for that track/segment.
                # This might need adjustment based on how diart consumers expect the Annotation.
                # A common way: output_annotation[segment, track_id_for_speaker] = speaker_label

            # For diart, the output is a list of (Annotation, SlidingWindowFeature)
            # Since we concatenated, we associate the single annotation with the last input chunk.
            # This is a simplification; a more complex streaming approach would be needed for true chunk-wise output.
            outputs = [(output_annotation, waveforms[-1])]

        except Exception as e:
            print(f"Error during WhisperX processing: {e}")
            # Fallback: return empty annotation for the last waveform
            outputs = [(Annotation(uri=waveforms[-1].uri if waveforms[-1].uri else "error_processing"), waveforms[-1])]

        return outputs
