import argparse
import torch
import json
import re
import subprocess
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyannote.audio import Pipeline
import functools

# --- FIX: Suppress Hugging Face generation config warnings ---
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# --- Import Pyannote classes explicitly to whitelist them ---
try:
    from pyannote.audio.core.task import Specifications, Problem
except ImportError:
    Specifications = None
    Problem = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

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
        
        # Use pcm_s16le for compatibility
        command.extend(["-c:a", "pcm_s16le", str(temp_path)])
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # --- FIX: Verify the file actually has content ---
            if not temp_path.exists() or temp_path.stat().st_size < 1000:
                raise RuntimeError(f"FFmpeg created an empty or invalid file. Check if start_time {start_time} is beyond the audio duration.")
                
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
        # Force CPU for ASR if using MPS to avoid chunking crashes
        run_device = self.device
        if str(self.device) == "mps":
            logger.warning("Apple Silicon (MPS) detected: Forcing ASR to run on CPU to avoid chunking crashes.")
            run_device = "cpu"

        logger.info(f"Loading ASR Model: {self.model_name} on {run_device}")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=run_device,
            chunk_length_s=30,
            return_timestamps=True
        )

    def transcribe(self, audio_path: Union[str, Path]) -> List[Dict]:
        logger.info(f"Transcribing audio file...")
        
        # FIX: "word" timestamps are more stable with CrisperWhisper
        # FIX: condition_on_prev_tokens=False prevents crash on independent chunks
        gen_kwargs = {
            "language": "en",
            "condition_on_prev_tokens": False
        }

        try:
            result = self.pipe(
                str(audio_path), 
                return_timestamps="word", 
                generate_kwargs=gen_kwargs
            )
            return result.get("chunks", [])
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            logger.info("Attempting fallback with standard timestamp mode...")
            
            # Last resort fallback
            try:
                result = self.pipe(
                    str(audio_path), 
                    return_timestamps=True, 
                    generate_kwargs=gen_kwargs
                )
                return result.get("chunks", [])
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise e2

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

        # Monkeypatch torch.load
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
                    logger.error("Hugging Face 403 Error: Please accept the license at https://hf.co/pyannote/speaker-diarization-3.1")
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
        try:
            if torch.accelerator.is_available():
                device = torch.accelerator.current_accelerator()
                logger.info(f"Global accelerator detected: {device}")
                return device
        except AttributeError:
            pass
        if torch.cuda.is_available(): return "cuda"
        elif torch.backends.mps.is_available(): return "mps"
        else: return "cpu"

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
                # Handle word-level timestamps (dict) vs standard timestamps (list)
                timestamp = chunk.get("timestamp")
                if isinstance(timestamp, tuple) or isinstance(timestamp, list):
                    start, end = timestamp
                else:
                    # Fallback if timestamp is missing or malformed
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
    parser = argparse.ArgumentParser(description="OOP Speech-to-Text Pipeline (CUDA/MPS/CPU)")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to input audio file")
    parser.add_argument("-o", "--output_dir", type=str, default="./", help="Directory to save outputs")
    parser.add_argument("--model", type=str, default="unsloth/CrisperWhisper", help="HuggingFace ASR model name")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Token")
    parser.add_argument("--no_diarize", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--no_timestamps", action="store_true", help="Exclude timestamps in text output")
    parser.add_argument("--clean_mode", type=str, choices=["none", "basic", "intelligent"], default="none", help="Level of disfluency removal")
    parser.add_argument("--start_time", type=str, default=None, help="Start time (e.g., 00:05:00 or 300)")
    parser.add_argument("--end_time", type=str, default=None, help="End time (e.g., 00:10:00 or 600)")
    
    args = parser.parse_args()
    orchestrator = TranscriptionOrchestrator(args)
    orchestrator.run()

if __name__ == "__main__":
    main()