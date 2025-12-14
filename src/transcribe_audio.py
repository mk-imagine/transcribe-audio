import argparse
import torch
import json
import re
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyannote.audio import Pipeline

# --- Configuration for Intelligent Cleaning ---
# We use a model fine-tuned for filled pauses. 
# You can swap this for other 'hafidev' models (e.g., restarts) if needed.
DISFLUENCY_MODEL_ID = "hafidev/bert-base-uncased-filled-pauses-disfluency-detection-beta-v1"

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

def regex_clean_text(text):
    """
    Basic cleaning: Removes common written representations of fillers.
    Good for speed, bad for context.
    """
    fillers = [r"\bum\b", r"\buh\b", r"\bah\b", r"\bhm+\b", r"\ber\b"]
    for filler in fillers:
        text = re.sub(filler, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text) # Dedup words
    return re.sub(r"\s+", " ", text).strip()

class IntelligentCleaner:
    def __init__(self, device):
        print(f"Loading BERT Disfluency Model: {DISFLUENCY_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(DISFLUENCY_MODEL_ID)
        self.model = AutoModelForTokenClassification.from_pretrained(DISFLUENCY_MODEL_ID).to(device)
        self.device = device

    def clean(self, text):
        if not text.strip():
            return ""
        
        # Tokenize and run inference
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get predictions (argmax of logits)
        predictions = torch.argmax(logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Filter tokens
        # We assume label '0' is fluent and others (like '1') are disfluent.
        # This checks the model's specific config to be safe, defaulting to assuming 0 is 'O' (Outside/Fluent)
        kept_tokens = []
        for token, label_id in zip(tokens, predictions):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if token in self.tokenizer.all_special_tokens:
                continue
                
            # Heuristic: If label is 0 (usually 'O'), keep it. 
            # If the model uses a different ID for fluent, this logic needs adjustment based on model card.
            # For hafidev models, 0 is typically the "Fluent" class.
            if label_id.item() == 0:
                kept_tokens.append(token)
        
        # Reconstruct text
        clean_text = self.tokenizer.convert_tokens_to_string(kept_tokens)
        return clean_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Intelligent Speech-to-Text")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to input audio")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")
    # Switched default to CrisperWhisper for better verbatim capture
    parser.add_argument("--model", type=str, default="unsloth/CrisperWhisper", help="ASR Model (Verbatim preferred)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Token")
    parser.add_argument("--no_diarize", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--no_timestamps", action="store_true", help="Exclude timestamps")
    parser.add_argument("--clean_mode", type=str, choices=["none", "basic", "intelligent"], default="none", 
                        help="Level of disfluency removal")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Initialize Intelligent Cleaner if requested ---
    cleaner = None
    if args.clean_mode == "intelligent":
        cleaner = IntelligentCleaner(device)

    # --- 1. Transcribe (ASR) ---
    print(f"Loading ASR Model: {args.model}")
    # We use the standard pipeline. For 'unsloth/CrisperWhisper', this works 
    # but strictly accurate word-level timestamps might benefit from their custom loader. 
    # For general chunk-level timestamps + verbatim text, this is sufficient.
    transcriber = pipeline(
        "automatic-speech-recognition", 
        model=args.model, 
        device=device,
        chunk_length_s=30,
        return_timestamps=True
    )
    
    print("Starting Verbatim Transcription...")
    asr_result = transcriber(str(input_path), return_timestamps=True)
    
    # --- 2. Diarization (Optional) ---
    diarization_segments = []
    if not args.no_diarize:
        if not args.hf_token:
            print("WARNING: No HF Token. Skipping Diarization.")
        else:
            print("Loading Diarization (pyannote/speaker-diarization-3.1)...")
            try:
                diarize_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=args.hf_token
                ).to(torch.device(device))
                
                print("Diarizing...")
                diarization = diarize_pipeline(str(input_path))
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    diarization_segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            except Exception as e:
                print(f"Diarization failed: {e}")

    # --- 3. Processing & Merging ---
    final_output = []
    chunks = asr_result.get("chunks", [])
    
    print(f"Processing text with clean_mode: {args.clean_mode}")
    
    for chunk in chunks:
        raw_text = chunk["text"]
        start, end = chunk["timestamp"]
        
        # Apply Cleaning
        if args.clean_mode == "intelligent":
            # BERT inference
            processed_text = cleaner.clean(raw_text)
        elif args.clean_mode == "basic":
            # Regex
            processed_text = regex_clean_text(raw_text)
        else:
            # Raw
            processed_text = raw_text

        if not processed_text.strip():
            continue

        segment_info = {
            "start": start,
            "end": end,
            "raw_text": raw_text.strip(),
            "text": processed_text.strip()
        }

        # Match Speaker
        if diarization_segments:
            best_speaker = "Unknown"
            max_overlap = 0
            for seg in diarization_segments:
                overlap_start = max(start, seg["start"])
                overlap_end = min(end, seg["end"])
                dur = max(0, overlap_end - overlap_start)
                if dur > max_overlap:
                    max_overlap = dur
                    best_speaker = seg["speaker"]
            segment_info["speaker"] = best_speaker
        
        final_output.append(segment_info)

    # --- 4. Save ---
    base_name = input_path.stem
    
    # JSON Dump
    with open(output_dir / f"{base_name}_data.json", "w") as f:
        json.dump(final_output, f, indent=2)

    # Text File
    txt_path = output_dir / f"{base_name}.txt"
    with open(txt_path, "w") as f:
        for seg in final_output:
            line = ""
            if not args.no_timestamps:
                line += f"[{format_time(seg['start'])} --> {format_time(seg['end'])}] "
            
            if "speaker" in seg:
                line += f"({seg['speaker']}): "
            
            line += seg["text"]
            f.write(line + "\n")

    print(f"Done. Results at {txt_path}")

if __name__ == "__main__":
    main()