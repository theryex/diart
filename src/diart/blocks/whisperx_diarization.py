from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Text, Any, Tuple

import torch
from pyannote.core import Annotation, SlidingWindowFeature, Segment # Ensure Segment is imported
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate

import whisperx
import numpy as np
# from pyannote.core import Segment # Already imported above
from whisperx.diarize import DiarizationPipeline as WXDiarizationPipeline
import os
from dotenv import load_dotenv
import pandas as pd

from .base import PipelineConfig, Pipeline, HyperParameter
from ..audio import AudioLoader
from .. import utils


@dataclass
class WhisperXDiarizationConfig(PipelineConfig):
    model_name: str = "base"
    language_code: str | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    compute_type: str | None = None
    batch_size: int = 16
    hf_token: str | None = None
    align_model: str | None = None
    diarization_model_name: str = "pyannote/speaker-diarization-3.1"
    interpolate_method: str = "nearest"
    return_char_alignments: bool = False
    _duration: float = 5.0
    _step: float = 0.5
    _latency: float | None = None
    _sample_rate: int = 16000
    min_audio_duration: float = 1.0
    min_speakers: int | None = None
    max_speakers: int | None = None

    def __post_init__(self):
        if self.hf_token is None or self.hf_token == "":
            self.hf_token = os.getenv("HF_TOKEN")
        if self.compute_type is None:
            self.compute_type = "float16" if self.device.type == "cuda" else "int8"
        if self._latency is None:
            self._latency = self._step

    @property
    def duration(self) -> float: return self._duration
    @property
    def step(self) -> float: return self._step
    @property
    def latency(self) -> float: return self._latency if self._latency is not None else self._step
    @property
    def sample_rate(self) -> int: return self._sample_rate
    def get_file_padding(self, filepath) -> Tuple[float, float]:
        file_duration = AudioLoader(self.sample_rate, mono=True).get_duration(filepath)
        right = utils.get_padding_right(self.latency, self.step)
        left = utils.get_padding_left(file_duration + right, self.duration)
        return left, right

class WhisperXDiarization(Pipeline):
    def __init__(self, config: WhisperXDiarizationConfig | None = None):
        self._config = WhisperXDiarizationConfig() if config is None else config
        try:
            self.wx_model = whisperx.load_model(
                self.config.model_name, str(self.config.device),
                compute_type=self.config.compute_type, language=self.config.language_code,
            )
            if self.config.language_code is None or self.config.language_code == "":
                self.align_model, self.align_metadata = None, None
            else:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.config.language_code, device=str(self.config.device), # device to string
                )

            hf_token_for_diarization = self.config.hf_token
            if isinstance(hf_token_for_diarization, bool):
                 hf_token_for_diarization = None

            self.diarize_model = WXDiarizationPipeline(
                model_name=self.config.diarization_model_name,
                use_auth_token=hf_token_for_diarization,
                device=str(self.config.device)
            )
        except Exception as e:
            self.wx_model, self.align_model, self.diarize_model = None, None, None
            raise RuntimeError(f"Failed to initialize WhisperX models: {e}")
        self.reset()

    @staticmethod
    def get_config_class() -> type: return WhisperXDiarizationConfig
    @staticmethod
    def suggest_metric() -> BaseMetric: return DiarizationErrorRate(collar=0.0, skip_overlap=False)
    @staticmethod
    def hyper_parameters() -> Sequence[HyperParameter]: return []
    @property
    def config(self) -> WhisperXDiarizationConfig: return self._config
    def reset(self): self.timestamp_shift = 0.0; pass
    def set_timestamp_shift(self, shift: float): self.timestamp_shift = shift

    def __call__(self, waveforms: Sequence[SlidingWindowFeature]) -> Sequence[Tuple[Annotation, SlidingWindowFeature]]:
        last_waveform_or_none = waveforms[-1] if waveforms else None

        if not self.wx_model:
            err_ann = Annotation(uri="error_models_not_loaded")
            err_ann[Segment(0, 0.01), "ERROR"] = "WhisperX ASR model not loaded properly."
            return [(err_ann, last_waveform_or_none)]
        if not self.diarize_model:
            err_ann = Annotation(uri="error_diarize_model_not_loaded")
            err_ann[Segment(0, 0.01), "ERROR"] = "WhisperX Diarization model not loaded properly."
            return [(err_ann, last_waveform_or_none)]
        if not waveforms:
            return []

        full_audio_data_list = []
        total_samples = 0
        for wf_feat in waveforms:
            if wf_feat.data.ndim == 2 and wf_feat.data.shape[1] == 1: full_audio_data_list.append(wf_feat.data.squeeze())
            elif wf_feat.data.ndim == 1: full_audio_data_list.append(wf_feat.data)
            else: full_audio_data_list.append(np.mean(wf_feat.data, axis=1))
            total_samples += wf_feat.data.shape[0]

        if not full_audio_data_list:
            err_ann = Annotation(uri="error_empty_audio_data")
            err_ann[Segment(0, 0.01), "ERROR"] = "Waveform processing resulted in empty audio data."
            return [(err_ann, last_waveform_or_none)]

        full_audio_np = np.ascontiguousarray(np.concatenate(full_audio_data_list).astype(np.float32))
        current_audio_duration_sec = full_audio_np.shape[0] / self.config.sample_rate

        if current_audio_duration_sec < self.config.min_audio_duration:
            msg = f"Audio duration {current_audio_duration_sec:.2f}s is less than minimum {self.config.min_audio_duration:.2f}s."
            err_ann = Annotation(uri="error_audio_too_short")
            err_ann[Segment(0, current_audio_duration_sec if current_audio_duration_sec > 0 else 0.01), "ERROR"] = msg
            return [(err_ann, last_waveform_or_none)]

        try:
            asr_result = self.wx_model.transcribe(full_audio_np, batch_size=self.config.batch_size)

            current_align_model, current_align_metadata = self.align_model, self.align_metadata
            if current_align_model is None:
                detected_language = asr_result.get("language")
                if not detected_language:
                    err_ann = Annotation(uri="error_no_language_detected")
                    err_ann[Segment(0, 0.01), "ERROR"] = "Could not detect language for alignment."
                    return [(err_ann, last_waveform_or_none)]
                try:
                    current_align_model, current_align_metadata = whisperx.load_align_model(
                        language_code=detected_language, device=str(self.config.device))
                except Exception as e:
                    msg = f"Failed to load alignment model for lang '{detected_language}': {e}"
                    err_ann = Annotation(uri="error_load_align_model")
                    err_ann[Segment(0, 0.01), "ERROR"] = msg
                    return [(err_ann, last_waveform_or_none)]

            if not current_align_model or not current_align_metadata:
                 err_ann = Annotation(uri="error_align_model_unavailable")
                 err_ann[Segment(0, 0.01), "ERROR"] = "Alignment model or metadata not available."
                 return [(err_ann, last_waveform_or_none)]
            if not asr_result["segments"]:
                err_ann = Annotation(uri="no_asr_segments")
                err_ann[Segment(0, 0.01), "ERROR"] = "No speech segments found by ASR."
                return [(err_ann, last_waveform_or_none)]

            aligned_result = whisperx.align(
                asr_result["segments"], current_align_model, current_align_metadata,
                full_audio_np, str(self.config.device), return_char_alignments=self.config.return_char_alignments
            )

            diarization_result = self.diarize_model(
                full_audio_np,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )

            di_annotation = Annotation(uri="whisperx_diarization_output")
            if isinstance(diarization_result, pd.DataFrame) and not diarization_result.empty:
                for _, row in diarization_result.iterrows():
                    start_time, end_time = float(row['start']), float(row['end'])
                    original_speaker_id = str(row['speaker'])
                    # Prevent double prefixing
                    if original_speaker_id.startswith('SPEAKER_'):
                        track_label = original_speaker_id
                    else:
                        track_label = f"SPEAKER_{original_speaker_id}"
                    segment = Segment(start_time, end_time)
                    # Use this track_label for both the track and the segment's label on that track
                    di_annotation[segment, track_label] = track_label
            elif isinstance(diarization_result, Annotation):
                di_annotation = diarization_result.rename_labels(copy=True)
                di_annotation.uri = "whisperx_diarization_output"

            print(f"[DEBUG] di_annotation track labels: {list(di_annotation.labels())}")
            if di_annotation.get_timeline().duration() > 0:
                final_result = whisperx.assign_word_speakers(di_annotation, aligned_result)
            else:
                final_result = aligned_result

            output_annotation = Annotation(uri="diart_whisperx_output")
            for segment_data in final_result.get("segments", []):
                start, end = segment_data.get("start"), segment_data.get("end")
                if start is None or end is None: continue
                start_time, end_time = float(start) + self.timestamp_shift, float(end) + self.timestamp_shift
                if end_time < start_time: end_time = start_time
                speaker_label, text = segment_data.get("speaker", "SPEAKER_UNKNOWN"), segment_data.get("text", "").strip()
                output_annotation[Segment(start_time, end_time), speaker_label] = text

            return [(output_annotation, last_waveform_or_none)]

        except Exception as e:
            print(f"Error during WhisperX processing: {e}")
            import traceback
            traceback.print_exc()
            error_annotation = Annotation(uri="critical_error_in_whisperx_pipeline_call")
            error_segment_duration = current_audio_duration_sec if 'current_audio_duration_sec' in locals() and current_audio_duration_sec > 0 else 0.01
            error_segment = Segment(0, error_segment_duration)
            error_annotation[error_segment, "ERROR"] = str(e)
            return [(error_annotation, last_waveform_or_none)]
