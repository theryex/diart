import argparse
from pathlib import Path
import dataclasses # Added for field inspection

import torch

from diart import argdoc
from diart import models as m
from diart import sources as src
from diart import utils
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter
# Import WhisperX classes
from diart.blocks.whisperx_diarization import WhisperXDiarization, WhisperXDiarizationConfig
# Sink classes (TXTWriter, SRTWriter)
from rx.core import Observer
from pyannote.core import Annotation, SlidingWindowFeature # Needed for type hints in sinks
from typing import Text, Union, Tuple # Needed for type hints in sinks

# Minimal placeholder for TXTWriter
class TXTWriter(Observer):
    def __init__(self, path: Union[Path, Text]):
        super().__init__()
        self.path = Path(path).expanduser()
        # Ensure the directory exists and the file is cleared if it already exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("") # Clear file on initialization
        # print(f"TXTWriter initialized for {self.path}") # Silenced for testing

    def on_next(self, value: Union[Tuple[Annotation, SlidingWindowFeature], Annotation]):
        prediction = value[0] if isinstance(value, tuple) else value
        if prediction is None:
            return

        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for segment, track, label_or_text in prediction.itertracks(yield_label=True):
                    speaker = track if track else "UNKNOWN"
                    text = label_or_text if label_or_text else ""
                    f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {speaker}: {text}\n")
        except Exception as e:
            print(f"Error writing to TXT file {self.path}: {e}")

    def on_error(self, error: Exception):
        print(f"TXTWriter error: {error}")

    def on_completed(self):
        # print(f"TXTWriter completed for {self.path}") # Silenced for testing
        pass


# Minimal placeholder for SRTWriter
class SRTWriter(Observer):
    def __init__(self, path: Union[Path, Text]):
        super().__init__()
        self.path = Path(path).expanduser()
        self.segment_index = 1
        # Ensure the directory exists and the file is cleared if it already exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("") # Clear file on initialization
        # print(f"SRTWriter initialized for {self.path}") # Silenced for testing

    def _format_timestamp_srt(self, seconds: float) -> str:
        assert seconds >= 0, "SRT timestamp must be non-negative"
        millis = round(seconds * 1000)
        secs = millis // 1000
        millis %= 1000
        mins = secs // 60
        secs %= 60
        hours = mins // 60
        mins %= 60
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    def on_next(self, value: Union[Tuple[Annotation, SlidingWindowFeature], Annotation]):
        prediction = value[0] if isinstance(value, tuple) else value
        if prediction is None:
            return

        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for segment, track, label_or_text in prediction.itertracks(yield_label=True):
                    speaker = track if track else "Speaker"
                    text = label_or_text if label_or_text else ""

                    start_time_str = self._format_timestamp_srt(segment.start)
                    end_time_str = self._format_timestamp_srt(segment.end)

                    f.write(f"{self.segment_index}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{speaker}: {text}\n\n")
                    self.segment_index += 1
        except Exception as e:
            print(f"Error writing to SRT file {self.path}: {e}")

    def on_error(self, error: Exception):
        print(f"SRTWriter error: {error}")

    def on_completed(self):
        # print(f"SRTWriter completed for {self.path}") # Silenced for testing
        pass


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=str,
        help="Path to an audio file | 'microphone' | 'microphone:<DEVICE_ID>'",
    )
    parser.add_argument(
        "--pipeline",
        default="SpeakerDiarization",
        type=str,
        help="Class of the pipeline to optimize. Defaults to 'SpeakerDiarization'. Use 'WhisperXDiarization' for WhisperX.",
    )
    parser.add_argument(
        "--pipeline_type",
        default="diart",
        type=str,
        choices=["diart", "whisperx"],
        help="Type of pipeline to use ('diart' or 'whisperx'). Defaults to 'diart'.",
    )
    parser.add_argument(
        "--segmentation",
        default="pyannote/segmentation",
        type=str,
        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation. (diart pipeline only)",
    )
    parser.add_argument(
        "--embedding",
        default="pyannote/embedding",
        type=str,
        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding. (diart pipeline only)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5,
        help=f"{argdoc.DURATION}. Defaults to training segmentation duration",
    )
    parser.add_argument(
        "--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5"
    )
    parser.add_argument(
        "--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5"
    )
    parser.add_argument(
        "--tau-active", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5. (diart pipeline only)"
    )
    parser.add_argument(
        "--rho-update", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3. (diart pipeline only)"
    )
    parser.add_argument(
        "--delta-new", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1. (diart pipeline only)"
    )
    parser.add_argument(
        "--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3. (diart pipeline only)"
    )
    parser.add_argument(
        "--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10. (diart pipeline only)"
    )
    parser.add_argument(
        "--max-speakers",
        default=20,
        type=int,
        help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20. (diart pipeline only)",
    )
    parser.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help="Skip plotting for faster inference",
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        action="store_true",
        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise",
    )
    parser.add_argument(
        "--output",
        type=str,
        help=f"{argdoc.OUTPUT}. Defaults to home directory if SOURCE == 'microphone' or parent directory if SOURCE is a file",
    )
    parser.add_argument(
        "--hf-token",
        default="true",
        type=str,
        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote/WhisperX diarization model)",
    )
    parser.add_argument(
        "--normalize-embedding-weights", # Specific to pyannote based pipeline
        action="store_true",
        help=f"{argdoc.NORMALIZE_EMBEDDING_WEIGHTS}. Defaults to False. (diart pipeline only)",
    )

    # Arguments for WhisperXDiarizationConfig
    parser.add_argument(
        "--whisperx_model",
        type=str,
        default="base",
        help="Name of the WhisperX ASR model (e.g., 'tiny', 'base', 'large-v2'). Defaults to 'base'. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--language_code",
        type=str,
        default=None,
        help="Language code for WhisperX transcription (e.g., 'en', 'es'). Auto-detect if None. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="Compute type for WhisperX model (e.g., 'float16', 'int8'). (whisperx pipeline only)",
    )
    parser.add_argument(
        "--whisperx_batch_size",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription. Defaults to 16. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--align_model",
        type=str,
        default=None,
        help="Name of the WhisperX alignment model. Defaults to None (auto-select based on language). (whisperx pipeline only)",
    )
    parser.add_argument(
        "--interpolate_method",
        type=str,
        default="nearest",
        help="Method for interpolating timestamps in WhisperX. Defaults to 'nearest'. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--return_char_alignments",
        action="store_true",
        help="Whether to return character-level alignments from WhisperX. Defaults to False. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--min_audio_duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds for WhisperX processing. Defaults to 1.0. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--wsx_min_speakers",
        type=int,
        default=None,
        help="Minimum number of speakers expected by WhisperX diarization. (whisperx pipeline only)",
    )
    parser.add_argument(
        "--wsx_max_speakers",
        type=int,
        default=None,
        help="Maximum number of speakers expected by WhisperX diarization. (whisperx pipeline only)",
    )
    # Arguments for TXT and SRT output
    txt_srt_group = parser.add_mutually_exclusive_group()
    txt_srt_group.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Full path to save the output in TXT format.",
    )
    txt_srt_group.add_argument(
        "--output_srt",
        type=str,
        default=None,
        help="Full path to save the output in SRT format.",
    )

    args = parser.parse_args()

    args.device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = utils.parse_hf_token_arg(args.hf_token)

    if args.pipeline_type == "diart":
        pipeline_class = utils.get_pipeline_class(args.pipeline)
        config_class = pipeline_class.get_config_class()

        diart_config_kwargs = vars(args).copy()
        diart_config_kwargs['segmentation'] = m.SegmentationModel.from_pretrained(args.segmentation, use_hf_token=hf_token)
        diart_config_kwargs['embedding'] = m.EmbeddingModel.from_pretrained(args.embedding, use_hf_token=hf_token)
        diart_config_kwargs['device'] = args.device

        from diart.blocks.diarization import SpeakerDiarizationConfig as DiartDefaultConfig
        valid_diart_fields = {f.name for f in dataclasses.fields(DiartDefaultConfig)}
        filtered_diart_config_kwargs = {
            k: v for k, v in diart_config_kwargs.items() if k in valid_diart_fields
        }
        config = config_class(**filtered_diart_config_kwargs)
        pipeline = pipeline_class(config)
    elif args.pipeline_type == "whisperx":
        whisperx_config_kwargs = {
            "model_name": args.whisperx_model,
            "language_code": args.language_code,
            "device": args.device,
            "compute_type": args.compute_type,
            "batch_size": args.whisperx_batch_size,
            "hf_token": hf_token,
            "align_model": args.align_model,
            "interpolate_method": args.interpolate_method,
            "return_char_alignments": args.return_char_alignments,
            "_duration": args.duration,
            "_step": args.step,
            "_latency": args.latency,
            "_sample_rate": 16000,
            "min_audio_duration": args.min_audio_duration,
            "min_speakers": args.wsx_min_speakers,
            "max_speakers": args.wsx_max_speakers,
        }
        filtered_whisperx_config_kwargs = {}
        for k, v in whisperx_config_kwargs.items():
            if v is not None:
                filtered_whisperx_config_kwargs[k] = v
            elif k in ["language_code", "align_model", "compute_type", "min_speakers", "max_speakers"]:
                 filtered_whisperx_config_kwargs[k] = v
        config = WhisperXDiarizationConfig(**filtered_whisperx_config_kwargs)
        pipeline = WhisperXDiarization(config)
    else:
        raise ValueError(f"Unsupported pipeline_type: {args.pipeline_type}")

    source_components = args.source.split(":")
    if source_components[0] != "microphone":
        args.source = Path(args.source).expanduser()
        args.output = args.source.parent if args.output is None else Path(args.output)
        padding = config.get_file_padding(args.source)
        audio_source = src.FileAudioSource(args.source, config.sample_rate, padding, config.step)
        pipeline.set_timestamp_shift(-padding[0])
    else:
        args.output = Path("~/").expanduser() if args.output is None else Path(args.output)
        microphone_device = int(source_components[1]) if len(source_components) > 1 else None
        audio_source = src.MicrophoneAudioSource(config.step, microphone_device)

    inference = StreamingInference(
        pipeline,
        audio_source,
        batch_size=1,
        do_profile=True,
        do_plot=not args.no_plot,
        show_progress=True,
    )

    if args.output is not None: # This is the directory for RTTM
        rttm_path = args.output / f"{audio_source.uri}.rttm"
        inference.attach_observers(RTTMWriter(audio_source.uri, rttm_path))
        # print(f"Writing RTTM output to {rttm_path}") # Silenced for testing

    if args.output_txt is not None:
        txt_path = Path(args.output_txt).expanduser()
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        inference.attach_observers(TXTWriter(txt_path))
        # print(f"Writing TXT output to {txt_path}") # Silenced for testing

    if args.output_srt is not None:
        srt_path = Path(args.output_srt).expanduser()
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        inference.attach_observers(SRTWriter(srt_path))
        # print(f"Writing SRT output to {srt_path}") # Silenced for testing

    try:
        inference()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
