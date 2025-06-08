import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import tempfile
import shutil # For cleaning up temp directories

import torch
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature

# Classes to test
from src.diart.blocks.whisperx_diarization import WhisperXDiarizationConfig, WhisperXDiarization
from src.diart.console.stream import TXTWriter, SRTWriter # Assuming they are accessible for import
from src.diart.audio import AudioLoader # To load audio for testing __call__
from pyannote.core import SlidingWindow # For TestOutputWriters

# Sample rate typically used
DEFAULT_SAMPLE_RATE = 16000

class TestWhisperXDiarizationConfig(unittest.TestCase):
    def test_initialization_defaults(self):
        config = WhisperXDiarizationConfig()
        self.assertEqual(config.model_name, "base")
        self.assertIsNone(config.language_code)
        self.assertEqual(config.device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.assertEqual(config.compute_type, "float16" if torch.cuda.is_available() else "int8")
        self.assertEqual(config.batch_size, 16)
        self.assertTrue(config.hf_token)
        self.assertIsNone(config.align_model)
        self.assertEqual(config.interpolate_method, "nearest")
        self.assertFalse(config.return_char_alignments)
        self.assertEqual(config._duration, 5.0)
        self.assertEqual(config._step, 0.5)
        self.assertEqual(config._latency, 0.5) # Default latency is step
        self.assertEqual(config._sample_rate, 16000)
        self.assertEqual(config.min_audio_duration, 1.0)
        self.assertIsNone(config.min_speakers)
        self.assertIsNone(config.max_speakers)

    def test_initialization_custom_values(self):
        custom_device = torch.device("cpu")
        config = WhisperXDiarizationConfig(
            model_name="large-v2",
            language_code="en",
            device=custom_device,
            compute_type="int8", # Explicitly CPU compute type
            batch_size=32,
            hf_token="test_token",
            align_model="WAV2VEC2_ASR_LARGE_LV60K_960H",
            interpolate_method="linear",
            return_char_alignments=True,
            _duration=10.0,
            _step=1.0,
            _latency=2.0,
            _sample_rate=8000,
            min_audio_duration=0.5,
            min_speakers=1,
            max_speakers=5
        )
        self.assertEqual(config.model_name, "large-v2")
        self.assertEqual(config.language_code, "en")
        self.assertEqual(config.device, custom_device)
        self.assertEqual(config.compute_type, "int8")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.hf_token, "test_token")
        self.assertEqual(config.align_model, "WAV2VEC2_ASR_LARGE_LV60K_960H")
        self.assertEqual(config.interpolate_method, "linear")
        self.assertTrue(config.return_char_alignments)
        self.assertEqual(config.duration, 10.0)
        self.assertEqual(config.step, 1.0)
        self.assertEqual(config.latency, 2.0)
        self.assertEqual(config.sample_rate, 8000)
        self.assertEqual(config.min_audio_duration, 0.5)
        self.assertEqual(config.min_speakers, 1)
        self.assertEqual(config.max_speakers, 5)

    def test_post_init_latency_default(self):
        config = WhisperXDiarizationConfig(_step=0.7, _latency=None)
        self.assertEqual(config.latency, 0.7)

    def test_post_init_compute_type_default_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            config = WhisperXDiarizationConfig(device=torch.device("cpu"), compute_type=None)
            self.assertEqual(config.compute_type, "int8")

    def test_post_init_compute_type_default_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            config = WhisperXDiarizationConfig(device=torch.device("cuda"), compute_type=None)
            self.assertEqual(config.compute_type, "float16")


class TestWhisperXDiarizationPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for any files produced during tests
        self.test_dir = tempfile.mkdtemp()
        self.sample_audio_path = Path(self.test_dir) / "sample.wav"

        # Create a dummy WAV file (sine wave)
        sample_rate = DEFAULT_SAMPLE_RATE
        duration_s = 2.0  # 2 seconds
        frequency = 440  # A4 note
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
        wave_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        # WhisperX expects float32
        self.wave_data_float32 = wave_data.astype(np.float32)

        # Save as a proper WAV file using a library if possible, or a simpler mock
        # For now, we'll mostly mock the audio loading part for pipeline tests
        # but having a real file path can be useful.
        # Let's assume AudioLoader().get_duration will be mocked or the file is real.
        # We'll mock the actual whisperx model loading and processing.

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)

    @patch('src.diart.blocks.whisperx_diarization.whisperx')
    def test_pipeline_initialization_successful(self, mock_whisperx_module):
        mock_asr_model = MagicMock()
        mock_align_model_tuple = (MagicMock(), {"language": "en", "dictionary": MagicMock()}) # model, metadata with dictionary
        mock_diarize_instance = MagicMock()

        mock_whisperx_module.load_model.return_value = mock_asr_model
        mock_whisperx_module.load_align_model.return_value = mock_align_model_tuple
        mock_whisperx_module.DiarizationPipeline.return_value = mock_diarize_instance

        config = WhisperXDiarizationConfig(language_code="en") # Need lang for align model load
        pipeline = WhisperXDiarization(config)

        self.assertIsNotNone(pipeline.wx_model)
        self.assertIsNotNone(pipeline.align_model)
        self.assertIsNotNone(pipeline.diarize_model)
        mock_whisperx_module.load_model.assert_called_once_with(
            config.model_name, config.device, compute_type=config.compute_type, language=config.language_code
        )
        mock_whisperx_module.load_align_model.assert_called_once_with(
            language_code=config.language_code, device=config.device
        )
        mock_whisperx_module.DiarizationPipeline.assert_called_once_with(
            model_name="pyannote/speaker-diarization-huggingface",
            use_auth_token=config.hf_token,
            device=config.device
        )

    @patch('src.diart.blocks.whisperx_diarization.whisperx')
    def test_pipeline_initialization_no_lang_for_align(self, mock_whisperx_module):
        # Test case where language_code is None, so align_model is not loaded at init
        mock_whisperx_module.load_model.return_value = MagicMock()
        mock_whisperx_module.DiarizationPipeline.return_value = MagicMock()

        config = WhisperXDiarizationConfig(language_code=None)
        pipeline = WhisperXDiarization(config)

        self.assertIsNotNone(pipeline.wx_model)
        self.assertIsNone(pipeline.align_model) # Should be None
        self.assertIsNotNone(pipeline.diarize_model)
        mock_whisperx_module.load_align_model.assert_not_called()


    @patch('src.diart.blocks.whisperx_diarization.whisperx') # Mock the whole whisperx module used by the pipeline
    @patch.object(WhisperXDiarization, '__init__', lambda self, config=None: None) # Bypass actual model loading in __init__
    def test_call_method_flow(self, mock_whisperx_module_for_call):
        # mock_whisperx_module_for_call is for functions like whisperx.align, whisperx.assign_word_speakers
        # Setup mocks for models that would have been loaded in __init__
        # Create a config instance to pass to the pipeline instance, even if __init__ is mocked
        test_config = WhisperXDiarizationConfig(min_audio_duration=0.1, language_code="en") # lang_code for align metadata
        pipeline = WhisperXDiarization(config=test_config)
        pipeline._config = test_config # Explicitly set config as __init__ is mocked

        pipeline.wx_model = MagicMock() # This would be set by a (mocked) whisperx.load_model
        # Simulate that align_model and metadata are loaded if language_code is present
        pipeline.align_model = MagicMock()  # This would be set by a (mocked) whisperx.load_align_model
        pipeline.align_metadata = {"language": "en", "dictionary": MagicMock()}
        pipeline.diarize_model = MagicMock() # This would be set by a (mocked) whisperx.DiarizationPipeline
        pipeline.timestamp_shift = 0 # Set manually as __init__ is bypassed

        # Mock ASR output
        mock_asr_result = {
            "segments": [{"text": "Hello world", "start": 0.1, "end": 1.5, "words": [{"word":"Hello", "start":0.1, "end":0.5, "score":0.9}, {"word":"world", "start":0.6, "end":1.2, "score":0.8}]}],
            "language": "en"
        }
        pipeline.wx_model.transcribe.return_value = mock_asr_result

        # Mock alignment output
        mock_aligned_result = {
            "segments": [{
                "text": "Hello world", "start": 0.1, "end": 1.5,
                "words": [{"word":"Hello", "start":0.1, "end":0.5, "score":0.9}, {"word":"world", "start":0.6, "end":1.2, "score":0.8}]
            }],
            "word_segments": [{"word":"Hello", "start":0.1, "end":0.5, "score":0.9}, {"word":"world", "start":0.6, "end":1.2, "score":0.8}]
        }
        # Configure the mock_whisperx_module_for_call that is passed to this test method
        mock_whisperx_module_for_call.align = MagicMock(return_value=mock_aligned_result)

        # Mock diarization output (pyannote.core.Annotation)
        mock_diarization_anno = Annotation(uri="test_audio")
        mock_diarization_anno[Segment(0.1, 1.5)] = "SPEAKER_00"
        # Ensure get_timeline().duration() > 0
        pipeline.diarize_model.return_value = mock_diarization_anno

        # Mock assign_word_speakers output
        mock_final_result = mock_aligned_result.copy() # Start with aligned result
        for seg in mock_final_result["segments"]:
            seg["speaker"] = "SPEAKER_00" # Add speaker info
            for word in seg.get("words",[]): word["speaker"] = "SPEAKER_00"

        mock_whisperx_module_for_call.assign_word_speakers = MagicMock(return_value=mock_final_result)

        # If language_code was None in config, load_align_model would be called here
        mock_whisperx_module_for_call.load_align_model = MagicMock(return_value=(pipeline.align_model, pipeline.align_metadata))

        # Create dummy audio input
        duration_s = 1.6 # seconds, > min_audio_duration
        num_samples = int(DEFAULT_SAMPLE_RATE * duration_s)
        waveform_data = np.random.randn(num_samples).astype(np.float32)

        # SlidingWindowFeature requires (num_samples, num_channels)
        # For mono, num_channels = 1
        waveform_data_2d = waveform_data.reshape(-1, 1)

        # Create SlidingWindowFeature (single chunk for this test)
        # The extent (start, duration) of SWF is important for URI generation if not set in annotation
        input_swf = SlidingWindowFeature(waveform_data_2d, SlidingWindow(start=0, duration=duration_s, step=duration_s))
        input_waveforms = [input_swf]

        # Call the pipeline
        output = pipeline(input_waveforms)

        # Assertions
        self.assertEqual(len(output), 1)
        self.assertIsInstance(output[0][0], Annotation)
        self.assertIsInstance(output[0][1], SlidingWindowFeature)

        output_annotation = output[0][0]
        self.assertEqual(len(list(output_annotation.itersegments())), 1)

        segments_data = list(output_annotation.itertracks(yield_label=True))
        self.assertEqual(len(segments_data), 1)
        segment, speaker, text = segments_data[0]

        self.assertAlmostEqual(segment.start, 0.1, places=2)
        self.assertAlmostEqual(segment.end, 1.5, places=2)
        self.assertEqual(speaker, "SPEAKER_00")
        self.assertEqual(text.strip(), "Hello world")

        pipeline.wx_model.transcribe.assert_called_once()
        # Access the mocked 'align' and 'assign_word_speakers' via the mock_whisperx_module_for_call
        mock_whisperx_module_for_call.align.assert_called_once()
        pipeline.diarize_model.assert_called_once()
        mock_whisperx_module_for_call.assign_word_speakers.assert_called_once()


class TestOutputWriters(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_annotation = Annotation(uri="test_anno")
        self.sample_annotation[Segment(0.1, 1.5), "SPEAKER_00"] = "Hello world."
        self.sample_annotation[Segment(1.8, 3.2), "SPEAKER_01"] = "This is a test."
        self.sample_annotation[Segment(3.5, 4.0), "SPEAKER_00"] = "Okay."

        # Create a dummy SlidingWindowFeature to pair with annotation for writer tests
        self.dummy_waveform = SlidingWindowFeature(np.array([[0.0]]), SlidingWindow(start=0, duration=1, step=1))


    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_txt_writer(self):
        txt_file_path = Path(self.temp_dir) / "output.txt"
        writer = TXTWriter(txt_file_path)

        # Simulate receiving data (Annotation, SlidingWindowFeature)
        writer.on_next((self.sample_annotation, self.dummy_waveform))
        writer.on_completed() # Or on_error, depending on what triggers file finalization if any

        self.assertTrue(txt_file_path.exists())
        with open(txt_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_content = (
            "[0.10s - 1.50s] SPEAKER_00: Hello world.\n"
            "[1.80s - 3.20s] SPEAKER_01: This is a test.\n"
            "[3.50s - 4.00s] SPEAKER_00: Okay.\n"
        )
        self.assertEqual(content, expected_content)

    def test_srt_writer(self):
        srt_file_path = Path(self.temp_dir) / "output.srt"
        writer = SRTWriter(srt_file_path)

        writer.on_next((self.sample_annotation, self.dummy_waveform))
        writer.on_completed()

        self.assertTrue(srt_file_path.exists())
        with open(srt_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Expected:
        # 1
        # 00:00:00,100 --> 00:00:01,500
        # SPEAKER_00: Hello world.
        #
        # 2
        # 00:00:01,800 --> 00:00:03,200
        # SPEAKER_01: This is a test.
        #
        # 3
        # 00:00:03,500 --> 00:00:04,000
        # SPEAKER_00: Okay.

        lines = content.strip().split('\n')
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "00:00:00,100 --> 00:00:01,500")
        self.assertEqual(lines[2], "SPEAKER_00: Hello world.")
        self.assertEqual(lines[3], "") # Empty line
        self.assertEqual(lines[4], "2")
        self.assertEqual(lines[5], "00:00:01,800 --> 00:00:03,200")
        self.assertEqual(lines[6], "SPEAKER_01: This is a test.")
        self.assertEqual(lines[7], "")
        self.assertEqual(lines[8], "3")
        self.assertEqual(lines[9], "00:00:03,500 --> 00:00:04,000")
        self.assertEqual(lines[10], "SPEAKER_00: Okay.")


if __name__ == "__main__":
    unittest.main()
