import argparse
import torch
import json
import subprocess
import os
import logging
import functools
import librosa
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyannote.audio import Pipeline
try:
    from pyannote.audio.core.task import Specifications, Problem
except ImportError:
    Specifications = None
    Problem = None
from transformers import logging as hf_logging

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
        if not start_time:
            return input_path

        safe_start = start_time.replace(':', '')
        temp_filename = f"temp_segment_{safe_start}_{input_path.name}"
        temp_path = self.output_dir / temp_filename
        
        logger.info(f"Creating temporary audio segment: {start_time} to {end_time or 'EOF'}...")
        
        command = ["ffmpeg", "-y", "-i", str(input_path), "-ss", str(start_time)]
        if end_time:
            command.extend(["-to", str(end_time)])
        
        command.extend(["-c:a", "pcm_s16le", str(temp_path)])
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if not temp_path.exists() or temp_path.stat().st_size < 1000:
                raise RuntimeError("FFmpeg created an empty/invalid file.")
            self.temp_files.append(temp_path)
            return temp_path
        except subprocess.CalledProcessError:
            logger.error("Error creating audio segment. Ensure 'ffmpeg' is installed.")
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
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad(): logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        kept_tokens = []
        for token, label_id in zip(tokens, predictions):
            if token in self.tokenizer.all_special_tokens: continue
            if label_id.item() == 0: kept_tokens.append(token)
        return self.tokenizer.convert_tokens_to_string(kept_tokens).strip()

class Transcriber:
    def __init__(self, model_name: str, device: Union[str, torch.device]):
        self.model_name = model_name
        self.device = device
        self._load_pipeline()

    def _load_pipeline(self):
        logger.info(f"Loading ASR Model: {self.model_name} on {self.device}")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device,
            chunk_length_s=30, 
            stride_length_s=5,
            return_timestamps=True
        )
        
        # --- Stability Patch (Nuclear Option) ---
        # We try to patch the model directly, but we don't rely on it.
        # We will ALSO pass these parameters in 'transcribe' below.
        if hasattr(self.pipe.model, "generation_config"):
            self.pipe.model.generation_config.condition_on_prev_tokens = False
            # These disable the "fallback" logic that crashes on CPU
            self.pipe.model.generation_config.compression_ratio_threshold = None
            self.pipe.model.generation_config.logprob_threshold = None
            self.pipe.model.generation_config.no_speech_threshold = None

    def transcribe(self, audio_path: Union[str, Path]) -> List[Dict]:
        logger.info(f"Transcribing audio file...")

        # --- DEBUG: Check audio file ---
        try:
            duration = librosa.get_duration(path=str(audio_path))
            logger.info(f"Audio duration: {duration:.2f}s")

            if duration == 0:
                logger.error("Audio file is empty!")
                return []

        except Exception as e:
            logger.error(f"Failed to load audio for validation: {e}")
            # Continue anyway, let the pipeline try
            duration = 0

        # For long audio files, process in segments to identify problematic chunks
        SEGMENT_SIZE = 300  # 5 minutes in seconds

        try:
            # If we couldn't get duration, or it's short, just try processing normally
            if duration > 0 and duration <= SEGMENT_SIZE:
                return self._transcribe_segment(str(audio_path), 0, duration if duration else None)

            # For long audio, process in segments
            logger.info(f"Audio is {duration:.1f}s long. Processing in {SEGMENT_SIZE}s segments...")
            all_chunks = []

            for start_time in range(0, int(duration), SEGMENT_SIZE):
                end_time = min(start_time + SEGMENT_SIZE, duration)
                logger.info(f"Processing segment: {start_time}s - {end_time}s ({start_time/60:.1f}min - {end_time/60:.1f}min)")

                try:
                    chunks = self._transcribe_segment(str(audio_path), start_time, end_time)
                    # Adjust timestamps to absolute time
                    none_count = 0
                    for chunk in chunks:
                        if "timestamp" in chunk and chunk["timestamp"]:
                            ts = chunk["timestamp"]
                            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                                # Handle None values in timestamps
                                if ts[0] is None or ts[1] is None:
                                    none_count += 1
                                ts_start = (ts[0] + start_time) if ts[0] is not None else start_time
                                ts_end = (ts[1] + start_time) if ts[1] is not None else end_time
                                chunk["timestamp"] = (ts_start, ts_end)
                    if none_count > 0:
                        logger.warning(f"  ⚠ {none_count} chunks had None timestamps, using segment boundaries as fallback")
                    all_chunks.extend(chunks)
                    logger.info(f"✓ Segment {start_time}s - {end_time}s completed ({len(chunks)} chunks)")
                except IndexError as e:
                    logger.error(f"✗ IndexError at segment {start_time}s - {end_time}s ({start_time/60:.1f}min - {end_time/60:.1f}min): {e}")
                    logger.error("Skipping this segment and continuing...")
                    # Insert error placeholder in transcript
                    all_chunks.append({
                        "timestamp": (start_time, end_time),
                        "text": f"[ERROR: Transcription failed for this segment due to IndexError. Missing audio from {start_time}s to {end_time}s ({start_time/60:.1f}min - {end_time/60:.1f}min)]"
                    })
                    continue
                except Exception as e:
                    logger.error(f"✗ Error at segment {start_time}s - {end_time}s: {e}")
                    logger.error("Skipping this segment and continuing...")
                    # Insert error placeholder in transcript
                    all_chunks.append({
                        "timestamp": (start_time, end_time),
                        "text": f"[ERROR: Transcription failed for this segment: {str(e)[:100]}. Missing audio from {start_time}s to {end_time}s ({start_time/60:.1f}min - {end_time/60:.1f}min)]"
                    })
                    continue

            return all_chunks

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise e

    def _transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> List[Dict]:
        """Transcribe a specific segment of audio"""
        # Load only the specific segment
        audio_data, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time - start_time)

        if len(audio_data) == 0:
            logger.warning(f"Segment {start_time}s-{end_time}s is empty")
            return []

        try:
            # --- FIX: Pass kwargs explicitly ---
            result = self.pipe(
                audio_data,
                return_timestamps=True,
                generate_kwargs={
                    "language": "en",
                    "task": "transcribe",
                    "condition_on_prev_tokens": False,
                    "compression_ratio_threshold": None,
                    "logprob_threshold": None,
                    "no_speech_threshold": None,
                    "temperature": 0.0
                }
            )
            return result.get("chunks", [])
        except IndexError as e:
            # Re-raise so the caller can log which segment failed
            raise e

class Diarizer:
    def __init__(self, auth_token: Optional[str], device: Union[str, torch.device]):
        self.auth_token = auth_token
        self.device = device
        self.pipeline = None

    def load(self):
        if not self.auth_token:
            logger.warning("No HF Token provided. Diarization will be skipped.")
            return

        logger.info("Loading Diarization Pipeline (pyannote/speaker-diarization-3.1)...")
        
        safe_globals = [torch.torch_version.TorchVersion]
        if Specifications: safe_globals.append(Specifications)
        if Problem: safe_globals.append(Problem)
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
                    "pyannote/speaker-diarization-3.1",
                    token=self.auth_token 
                ).to(target_device)
            except Exception as e:
                if "403" in str(e):
                    logger.error("Hugging Face 403 Error: Check your token permissions.")
                    raise e
                logger.info(f"Retrying with 'use_auth_token'...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
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
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        except Exception as e:
            logger.error(f"Diarization runtime error: {e}")
        return segments

class TranscriptionOrchestrator:
    def __init__(self, args):
        self.args = args
        self.input_path = Path(args.input_path)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._get_device()
        
        self.audio_handler = AudioHandler(self.output_dir)
        self.transcriber = Transcriber(args.model, self.device)
        self.cleaner = TextCleaner(args.clean_mode, self.device)
        self.diarizer = Diarizer(args.hf_token, self.device)
        
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
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="unsloth/CrisperWhisper")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--no_diarize", action="store_true")
    parser.add_argument("--no_timestamps", action="store_true")
    parser.add_argument("--clean_mode", type=str, choices=["none", "basic", "intelligent"], default="none")
    parser.add_argument("--start_time", type=str, default=None)
    parser.add_argument("--end_time", type=str, default=None)
    
    args = parser.parse_args()

    # --- FIX: Handle Empty Strings from Shell ---
    # We use 'if not args.hf_token' to catch both None and empty strings ""
    if not args.hf_token:
        args.hf_token = os.getenv("HF_TOKEN")

    orchestrator = TranscriptionOrchestrator(args)
    orchestrator.run()

if __name__ == "__main__":
    main()