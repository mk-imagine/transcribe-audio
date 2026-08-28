import argparse
import torch
import json
import subprocess
import os
import tempfile
import soundfile as sf # type: ignore
import logging
import librosa
import re
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyannote.audio import Pipeline
try:
    from pyannote.audio.core.task import Specifications, Problem
except ImportError:
    Specifications = None   # type: ignore
    Problem = None          # type: ignore
from transformers import logging as hf_logging

# mlx_whisper is Apple-Silicon only; import lazily so this module loads on
# Linux/CUDA hosts where the MLX backend is unavailable.
try:
    import mlx_whisper  # type: ignore
except ImportError:
    mlx_whisper = None  # type: ignore

hf_logging.set_verbosity_error()

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# 1. Load Environment Variables (Force Override)
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True) 
    logger.info(f"Loaded .env file from: {env_path}")
else:
    logger.warning(f"No .env file found at: {env_path}")

# Check for Token
if os.getenv("HF_TOKEN"):
    logger.info("HF_TOKEN detected in environment.")
else:
    logger.warning("HF_TOKEN is missing from environment.")

class AudioHandler:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.temp_files: List[Path] = []

    def prepare_segment(self, input_path: Path, start_time: Optional[str], end_time: Optional[str]) -> Path:
        if input_path.is_dir():
            raise IsADirectoryError(f"Input path points to a directory and not a file: {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        if not start_time:
            return input_path

        safe_start = start_time.replace(':', '')
        # The segment is always re-encoded as PCM s16le below, so it must land in
        # a container that accepts PCM. Reusing the source suffix wrote PCM into
        # e.g. .m4a, which the MP4 muxer rejects outright.
        temp_filename = f"temp_segment_{safe_start}_{input_path.stem}.wav"
        temp_path = self.output_dir / temp_filename
        
        logger.info(f"Creating temporary audio segment: {start_time} to {end_time or 'EOF'}...")
        
        command = ["ffmpeg", "-y", "-i", str(input_path), "-ss", str(start_time)]
        if end_time:
            command.extend(["-to", str(end_time)])
        
        command.extend(["-c:a", "pcm_s16le", str(temp_path)])
        
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not temp_path.exists() or temp_path.stat().st_size < 1000:
                raise RuntimeError("FFmpeg created an empty/invalid file.")
            self.temp_files.append(temp_path)
            return temp_path
        except FileNotFoundError:
            logger.error("ffmpeg not found on PATH. It is required for --start_time/--end_time.")
            raise
        except subprocess.CalledProcessError as exc:
            # Previously stderr was routed into a discarded stdout, so real
            # ffmpeg failures surfaced only as an exit code.
            detail = (exc.stderr or "").strip().splitlines()[-8:]
            logger.error("ffmpeg failed creating audio segment:\n%s", "\n".join(detail))
            raise

    def cleanup(self):
        for path in self.temp_files:
            if path.exists():
                try:
                    os.remove(path)
                    logger.info(f"Removed temporary file: {path.name}")
                except OSError:
                    pass

class TextCleaner:
    DISFLUENCY_MODEL_ID = "hafidev/bert-base-uncased-filled-pauses-disfluency-detection-beta-v1"

    def __init__(self, mode: str, device: Union[str, torch.device]):
        self.mode = mode
        self.device = device
        self.tokenizer = None
        self.model = None
        
        if self.mode == "intelligent":
            self._load_bert_model()

    def _load_bert_model(self):
        logger.info(f"Loading BERT Disfluency Model: {self.DISFLUENCY_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.DISFLUENCY_MODEL_ID)
        self.model = AutoModelForTokenClassification.from_pretrained(self.DISFLUENCY_MODEL_ID).to(self.device)

    def clean(self, text: str) -> str:
        if not text or not text.strip(): return ""
        if self.mode == "intelligent": return self._clean_intelligent(text)
        elif self.mode == "basic": return self._clean_regex(text)
        return text.strip()

    def _clean_regex(self, text: str) -> str:
        fillers = [r"\bum\b", r"\buh\b", r"\bah\b", r"\bhm+\b", r"\ber\b"]
        for filler in fillers: text = re.sub(filler, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    def _clean_intelligent(self, text: str) -> str:
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert self.model is not None, "Model not loaded"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad(): logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        kept_tokens = []
        for token, label_id in zip(tokens, predictions):
            if token in self.tokenizer.all_special_tokens: continue
            if label_id.item() == 0: kept_tokens.append(token)
        return self.tokenizer.convert_tokens_to_string(kept_tokens).strip()

class BaseTranscriber:
    """Abstract base class for transcription models"""
    SEGMENT_SIZE = 300  # 5 minutes in seconds

    def __init__(self, model_name: str, device: Union[str, torch.device]):
        self.model_name = model_name
        self.device = device

    def transcribe(self, audio_path: Union[str, Path]) -> List[Dict]:
        """Transcribe audio file with automatic segmentation."""
        logger.info(f"Transcribing audio file...")

        # Check audio duration
        duration = self._get_audio_duration(audio_path)
        if duration == 0:
            logger.error("Audio file is empty!")
            return []

        # Short audio - process directly
        if duration <= self.SEGMENT_SIZE:
            return self._transcribe_segment(str(audio_path), 0, duration)

        # Long audio - process in segments
        logger.info(f"Audio is {duration:.1f}s long. Processing in {self.SEGMENT_SIZE}s segments...")
        all_chunks = []

        for start_time in range(0, int(duration), self.SEGMENT_SIZE):
            end_time = min(start_time + self.SEGMENT_SIZE, duration)
            logger.info(f"Processing segment: {start_time}s - {end_time}s ({start_time/60:.1f}min - {end_time/60:.1f}min)")

            try:
                chunks = self._transcribe_segment(str(audio_path), start_time, end_time)
                # Adjust timestamps to absolute time
                self._adjust_timestamps(chunks, start_time, end_time)
                all_chunks.extend(chunks)
                logger.info(f"✓ Segment {start_time}s - {end_time}s completed ({len(chunks)} chunks)")
            except Exception as e:
                logger.error(f"✗ Error at segment {start_time}s - {end_time}s: {e}")
                all_chunks.append(self._create_error_chunk(start_time, end_time, e))
                continue

        return all_chunks

    def _get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """Get audio duration in seconds."""
        try:
            duration = librosa.get_duration(path=str(audio_path))
            logger.info(f"Audio duration: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Failed to load audio for validation: {e}")
            return 0

    def _adjust_timestamps(self, chunks: List[Dict], start_time: float, end_time: float):
        """Adjust chunk timestamps to absolute time (modifies chunks in-place)."""
        for chunk in chunks:
            if "timestamp" in chunk and chunk["timestamp"]:
                ts = chunk["timestamp"]
                if isinstance(ts, (list, tuple)) and len(ts) == 2:
                    ts_start = (ts[0] + start_time) if ts[0] is not None else start_time
                    ts_end = (ts[1] + start_time) if ts[1] is not None else end_time
                    chunk["timestamp"] = (ts_start, ts_end)

    def _create_error_chunk(self, start_time: float, end_time: float, error: Exception) -> Dict:
        """Create error placeholder chunk."""
        return {
            "timestamp": (start_time, end_time),
            "text": f"[ERROR: Transcription failed: {str(error)[:100]}. Missing audio from {start_time}s to {end_time}s ({start_time/60:.1f}min - {end_time/60:.1f}min)]"
        }

    def _transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> List[Dict]:
        """Transcribe a segment of audio. Must be implemented by subclasses."""
        raise NotImplementedError

class WhisperTranscriber(BaseTranscriber):
    """Transcriber using OpenAI Whisper models"""
    def __init__(self, model_name: str, device: Union[str, torch.device], return_timestamps: str | bool = "word"):
        super().__init__(model_name, device)
        self.return_timestamps = return_timestamps
        self._load_pipeline()

    def _load_pipeline(self):
        # whisper-large-v3 derivatives are ~1.55B params, which is ~6.2GB in
        # float32 -- more than a small card can host alongside the diarizer.
        # Half precision halves that to ~3.1GB (measured); CPU keeps float32
        # since fp16 matmuls there are slow and often unimplemented.
        device_str = str(self.device)
        if device_str.startswith("cpu"):
            dtype = torch.float32
        elif device_str.startswith("cuda") and torch.cuda.is_bf16_supported():
            # Ampere and newer: bf16 costs the same memory as fp16 but keeps
            # fp32's exponent range, so long jobs cannot silently overflow.
            # Matches the dtype the Granite path already uses on GPU.
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        logger.info(f"Loading ASR Model: {self.model_name} on {self.device} ({dtype})")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device,
            torch_dtype=dtype,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True
        )

    def _transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> List[Dict]:
        """Transcribe a segment using Whisper."""
        try:
            # librosa loads directly to a numpy array (float32)
            # sr=16000 ensures we match Whisper's expected rate
            audio_array, sr = librosa.load(
                audio_path, 
                sr=16000, 
                offset=start_time, 
                duration=end_time - start_time
            )
        except Exception as e:
            logger.error(f"Failed to load audio segment: {e}")
            return []

        if len(audio_array) == 0:
            return []
        
        # Instead of a filename, we pass a dictionary with 'raw' audio and 'sampling_rate'.
        # This tells the pipeline "Here is the data, don't look for a file."
        input_data = {"raw": audio_array, "sampling_rate": sr}

        try:
            # Whisper transcription with configurable timestamp granularity
            result = self.pipe(
                input_data,
                return_timestamps=self.return_timestamps,  # "word" for word-level, True for chunk-level
                generate_kwargs={
                    "language": "en",
                    "task": "transcribe",
                    # The following two kwargs are to attempt to preserve disfluencies with a base openai whisper model
                    # "initial_prompt": "Umm, uh, like, I mean, well, sort of, you know...",
                    # "suppress_tokens": []  # Preserve disfluencies ("um", "uh", etc.)
                }
            )

            # Log result
            logger.info(f"  Pipeline result keys: {result.keys() if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict) and "text" in result:
                logger.info(f"  Got text (length={len(result['text'])}): {result['text'][:100]}...")

            # Extract chunks
            chunks = result.get("chunks", [])
            
            # Fallback if no chunks but text exists
            if not chunks and "text" in result and result["text"].strip():
                logger.warning(f"  No chunks returned, but got text. Creating single chunk.")
                chunks = [{
                    "timestamp": (0.0, end_time - start_time),
                    "text": result["text"]
                }]

            logger.info(f"  Returning {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Whisper pipeline error: {e}")
            raise e

class CrisperWhisperTranscriber(WhisperTranscriber):
    """Transcriber for CrisperWhisper models with pause adjustment.

    CrisperWhisper is optimized for accurate word-level timestamps with proper
    pause handling. This class extends WhisperTranscriber with CrisperWhisper-specific
    post-processing to adjust pause timings between words.

    Based on: https://huggingface.co/nyrahealth/CrisperWhisper
    """

    def __init__(self, model_name: str, device: Union[str, torch.device]):
        # Use nyrahealth/CrisperWhisper if user specified unsloth variant
        # if "unsloth" in model_name.lower() and "crisper" in model_name.lower():
        #     logger.warning(f"Converting {model_name} to nyrahealth/CrisperWhisper (recommended by model card)")
        #     model_name = "nyrahealth/CrisperWhisper"

        # Use chunk-level timestamps to avoid MPS compatibility issues with word-level timestamps
        super().__init__(model_name, device, return_timestamps=True)
        self.split_threshold = 0.12  # Default pause split threshold from model card
        logger.info(f"CrisperWhisper initialized with chunk-level timestamps and pause split threshold: {self.split_threshold}s")

    def _transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> List[Dict]:
        """Transcribe a segment using CrisperWhisper with pause adjustment."""
        # Get chunks from parent WhisperTranscriber
        chunks = super()._transcribe_segment(audio_path, start_time, end_time)

        # Apply CrisperWhisper-specific pause adjustment
        adjusted_chunks = self._adjust_pauses(chunks)

        return adjusted_chunks

    def _adjust_pauses(self, chunks: List[Dict]) -> List[Dict]:
        """
        Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.

        This implements the pause adjustment algorithm from the CrisperWhisper model card.
        Pauses between words are redistributed to avoid awkward timing gaps:
        - Pauses <= split_threshold: distributed evenly (50/50) between adjacent words
        - Pauses > split_threshold: only split_threshold/2 distributed to each word

        Args:
            chunks: List of transcription chunks with timestamps

        Returns:
            List of chunks with adjusted timestamps
        """
        if len(chunks) <= 1:
            return chunks

        adjusted_chunks = chunks.copy()

        for i in range(len(adjusted_chunks) - 1):
            current_chunk = adjusted_chunks[i]
            next_chunk = adjusted_chunks[i + 1]

            # Extract timestamps
            current_ts = current_chunk.get("timestamp")
            next_ts = next_chunk.get("timestamp")

            # Skip if timestamps are missing or malformed
            if not current_ts or not next_ts:
                continue
            if not isinstance(current_ts, (list, tuple)) or len(current_ts) != 2:
                continue
            if not isinstance(next_ts, (list, tuple)) or len(next_ts) != 2:
                continue

            current_start, current_end = current_ts
            next_start, next_end = next_ts

            # Skip if None values
            if current_end is None or next_start is None:
                continue

            # Calculate pause duration
            pause_duration = next_start - current_end

            if pause_duration > 0:
                # Determine how much to distribute
                if pause_duration > self.split_threshold:
                    distribute = self.split_threshold / 2
                else:
                    distribute = pause_duration / 2

                # Adjust current chunk end time
                adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

                # Adjust next chunk start time
                adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)

        return adjusted_chunks

class MLXCrisperWhisperTranscriber(BaseTranscriber):
    """Transcriber for CrisperWhisper using MLX (optimized for Apple Silicon).

    This implementation uses the mlx_whisper library which is optimized for Apple Silicon
    and avoids the compatibility issues with the standard transformers pipeline.

    Model: kyr0/crisperwhisper-unsloth-mlx
    """

    def __init__(self, model_name: str, device: Union[str, torch.device]):
        super().__init__(model_name, device)
        # Convert to MLX model path if needed
        if "kyr0" not in model_name.lower():
            self.mlx_model = "kyr0/crisperwhisper-unsloth-mlx"
            logger.info(f"Using MLX model: {self.mlx_model}")
        else:
            self.mlx_model = model_name

        if mlx_whisper is None:
            raise RuntimeError(
                "mlx_whisper is not installed. The MLX CrisperWhisper backend "
                "requires Apple Silicon; use --model unsloth/CrisperWhisper for "
                "the transformers/CUDA backend."
            )

        logger.info(f"MLX CrisperWhisper initialized")

    def _transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> List[Dict]:
        """Transcribe a segment using MLX CrisperWhisper with word-level timestamps."""
        try:
            # Load audio segment
            audio_array, sr = librosa.load(
                audio_path,
                sr=16000,
                offset=start_time,
                duration=end_time - start_time
            )
        except Exception as e:
            logger.error(f"Failed to load audio segment: {e}")
            return []

        if len(audio_array) == 0:
            return []

        try:
            # MLX Whisper transcription with word-level timestamps
            result = mlx_whisper.transcribe(
                audio_array,
                path_or_hf_repo=self.mlx_model,
                word_timestamps=True,
                verbose=False
            )

            logger.info(f"  MLX result keys: {result.keys() if isinstance(result, dict) else type(result)}")

            # Extract word-level segments
            segments = result.get("segments", [])
            chunks = []

            for segment in segments:
                # Get words from segment
                words = segment.get("words", [])
                if words:
                    # Create chunk for each word
                    for word_info in words:
                        chunks.append({
                            "timestamp": (word_info["start"], word_info["end"]),
                            "text": word_info["word"]
                        })
                else:
                    # Fallback to segment-level if no words
                    chunks.append({
                        "timestamp": (segment["start"], segment["end"]),
                        "text": segment["text"]
                    })

            # Fallback if no segments but text exists
            if not chunks and "text" in result and result["text"].strip():
                logger.warning(f"  No segments returned, but got text. Creating single chunk.")
                chunks = [{
                    "timestamp": (0.0, end_time - start_time),
                    "text": result["text"]
                }]

            logger.info(f"  Returning {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"MLX CrisperWhisper error: {e}")
            raise e

class GraniteTranscriber(BaseTranscriber):
    """Transcriber using IBM Granite Speech models"""
    def __init__(self, model_name: str, device: Union[str, torch.device]):
        super().__init__(model_name, device)
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading IBM Granite ASR Model: {self.model_name} on {self.device}")
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            device_map=self.device if isinstance(self.device, str) else str(self.device),
            # Fallback for CPU: bfloat16 is often not supported on CPU, use float32
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32
        )
        
        self.tokenizer = self.processor.tokenizer
        
        logger.info(f"IBM Granite model loaded successfully")

    def _transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> List[Dict]:
        """Transcribe a segment using IBM Granite."""
        import torchaudio   # type: ignore
        
        granite_target_sample_rate = 16000

        try:
            waveform, sr = librosa.load(
                audio_path, 
                sr=granite_target_sample_rate, 
                offset=start_time, 
                duration=end_time - start_time
            )
        except Exception as e:
            logger.error(f"Failed to load audio segment: {e}")
            return []

        if len(waveform) == 0:
            return []

        today = datetime.now()
        
        system_prompt = (f"Knowledge Cutoff Date: April 2024.\nToday's Date: {today.month}/{today.day}/{today.year}.\n"
                        f"You are Transcriber, an expert in providing accurate transcriptions of audio.")
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        
        # Apply chat template to get the text input for the model
        prompt_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Process audio through processor
        inputs = self.processor(
            prompt_text,
            waveform,
            sampling_rate=granite_target_sample_rate,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)

        # Decode
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        logger.info(f"  Got text (length={len(transcription)}): {transcription[:100]}...")

        # Granite doesn't provide word-level timestamps by default
        # Return as single chunk for now
        chunks = [{
            "timestamp": (0.0, end_time - start_time),
            "text": transcription
        }]

        logger.info(f"  Returning {len(chunks)} chunks")
        return chunks

class TranscriberFactory:
    """Factory to create the appropriate transcriber based on model name"""
    @staticmethod
    def create(model_name: str, device: Union[str, torch.device],
               timestamp_mode: str = "word") -> BaseTranscriber:
        return_timestamps: Union[str, bool] = "word" if timestamp_mode == "word" else True
        if "granite-speech" in model_name.lower():
            logger.info("Creating IBM Granite transcriber")
            return GraniteTranscriber(model_name, device)
        elif "crisper" in model_name.lower() and ("kyr0" in model_name.lower() or "mlx" in model_name.lower()):
            logger.info("Creating MLX CrisperWhisper transcriber (Apple Silicon optimized)")
            return MLXCrisperWhisperTranscriber(model_name, device)
        elif "crisper" in model_name.lower():
            logger.info("Creating CrisperWhisper transcriber (transformers pipeline)")
            return CrisperWhisperTranscriber(model_name, device)
        else:
            logger.info(f"Creating standard Whisper transcriber ({timestamp_mode}-level timestamps)")
            return WhisperTranscriber(model_name, device, return_timestamps=return_timestamps)

class Diarizer:
    def __init__(self, model_name: str, auth_token: Optional[str], device: Union[str, torch.device]):
        self.model_name = model_name
        self.auth_token = auth_token
        self.device = device
        self.pipeline = None

    def load(self):
        if not self.auth_token:
            logger.warning("No HF Token provided. Diarization will be skipped.")
            return

        logger.info("Loading Diarization Pipeline (pyannote/speaker-diarization-3.1)...")
        
        safe_globals = [torch.torch_version.TorchVersion]
        if Specifications is not None: safe_globals.append(Specifications)
        if Problem is not None: safe_globals.append(Problem)
        torch.serialization.add_safe_globals(safe_globals)

        original_load = torch.load
        def safe_load(*args, **kwargs):
            if 'weights_only' in kwargs: del kwargs['weights_only']
            return original_load(*args, **kwargs, weights_only=False)
        torch.load = safe_load

        try:
            target_device = torch.device(self.device) if isinstance(self.device, str) else self.device
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=self.auth_token 
                ).to(target_device)
            except Exception as e:
                if "403" in str(e):
                    logger.error("Hugging Face 403 Error: Check your token permissions.")
                    raise e
                logger.info(f"Retrying with 'use_auth_token'...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    use_auth_token=self.auth_token
                ).to(target_device)
        except Exception as e:
            logger.error(f"Failed to load Diarization pipeline: {e}")
            self.pipeline = None
        finally:
            torch.load = original_load

    def run(self, audio_path: Union[str, Path]) -> List[Dict]:
        if not self.pipeline: return []
        logger.info("Running speaker diarization...")
        segments = []
        try:
            diarization = self.pipeline(str(audio_path))
            logger.info(f"Diarization type: {type(diarization)}")
            
            for turn, speaker in diarization.speaker_diarization:
                segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            logger.info(f"Found {len(segments)} speaker segments")

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            logger.error(f"Diarization object type: {type(diarization)}")
            logger.error(f"Available methods: {[attr for attr in dir(diarization) if not attr.startswith('_')]}")
        return segments

class TranscriptionOrchestrator:
    def __init__(self, args):
        self.args = args
        self.input_path = Path(args.input_path)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # self.device = "cpu"
        self.device = self._get_device()
        
        self.audio_handler = AudioHandler(self.output_dir)
        self.transcriber = TranscriberFactory.create(
            args.model, self.device, getattr(args, "timestamp_mode", "word")
        )
        self.cleaner = TextCleaner(args.clean_mode, self.device)
        self.diarizer = Diarizer(args.diarizer_model, args.hf_token, self.device)
        
        if not args.no_diarize:
            self.diarizer.load()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using MPS device")
            return "mps"
        else:
            logger.info("Using CPU")
            return "cpu"

    def _assign_speaker(self, start: float, end: float, diarization_segments: List[Dict]) -> str:
        if not diarization_segments: return "Unknown"
        best_speaker, max_overlap = "Unknown", 0
        for seg in diarization_segments:
            overlap_start = max(start, seg["start"])
            overlap_end = min(end, seg["end"])
            dur = max(0, overlap_end - overlap_start)
            if dur > max_overlap:
                max_overlap = dur
                best_speaker = seg["speaker"]
        return best_speaker

    def _format_time(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    def run(self):
        try:
            processing_path = self.audio_handler.prepare_segment(
                self.input_path, self.args.start_time, self.args.end_time
            )
            raw_chunks = self.transcriber.transcribe(processing_path)
            speaker_segments = []
            if not self.args.no_diarize:
                speaker_segments = self.diarizer.run(processing_path)

            final_output = []
            logger.info(f"Processing text (Mode: {self.args.clean_mode})...")

            for chunk in raw_chunks:
                raw_text = chunk["text"]
                timestamp = chunk.get("timestamp")
                if timestamp and len(timestamp) == 2:
                    start, end = timestamp
                else:
                    start, end = 0.0, 0.0

                cleaned_text = self.cleaner.clean(raw_text)
                if not cleaned_text.strip(): continue
                segment_data = {"start": start, "end": end, "text": cleaned_text}
                if speaker_segments:
                    segment_data["speaker"] = self._assign_speaker(start, end, speaker_segments)
                final_output.append(segment_data)

            self._save_results(final_output)
        finally:
            self.audio_handler.cleanup()

    def _save_results(self, data: List[Dict]):
        base_name = self.input_path.stem
        if self.args.start_time:
            base_name += f"_seg_{self.args.start_time.replace(':', '')}"
        
        if self.args.job_id:
            base_name += f"_job{self.args.job_id}"
        else:
            current_time = datetime.now()
            base_name += f"_{current_time.month}-{current_time.day}-{current_time.hour}:{current_time.minute}"
        
        json_path = self.output_dir / f"{base_name}_data.json"
        with open(json_path, "w") as f: json.dump(data, f, indent=2)
        
        txt_path = self.output_dir / f"{base_name}.txt"
        with open(txt_path, "w") as f:
            for seg in data:
                line = ""
                if not self.args.no_timestamps:
                    line += f"[{self._format_time(seg['start'])} --> {self._format_time(seg['end'])}] "
                if "speaker" in seg: line += f"({seg['speaker']}): "
                line += seg["text"]
                f.write(line + "\n")
        logger.info(f"Processing complete. Results saved to: {txt_path}")

def main():
    parser = argparse.ArgumentParser(description="Cluster Speech-to-Text Pipeline")
    parser.add_argument("-i", "--input_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    # ibm-granite/granite-speech-3.3-2b
    # openai/whisper-small
    # unsloth/crisperwhisper
    parser.add_argument("--model", type=str, default="unsloth/crisperwhisper")
    parser.add_argument("--diarizer_model", type=str, default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--no_diarize", action="store_true")
    parser.add_argument("--no_timestamps", action="store_true")
    # Word-level timestamps require the decoder to emit per-token alignments,
    # which needs well over 8GB of VRAM on whisper-large-v3 class models.
    # "chunk" asks for segment-level timestamps instead and fits comfortably.
    parser.add_argument("--timestamp_mode", type=str, choices=["word", "chunk"],
                        default="word")
    parser.add_argument("--clean_mode", type=str, choices=["none", "basic", "intelligent"], default="none")
    parser.add_argument("--start_time", type=str, default=None)
    parser.add_argument("--end_time", type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    
    args = parser.parse_args()

    # --- FIX: Handle Empty Strings from Shell ---
    # We use 'if not args.hf_token' to catch both None and empty strings ""
    if not args.hf_token:
        args.hf_token = os.getenv("HF_TOKEN")

    orchestrator = TranscriptionOrchestrator(args)
    orchestrator.run()

if __name__ == "__main__":
    main()