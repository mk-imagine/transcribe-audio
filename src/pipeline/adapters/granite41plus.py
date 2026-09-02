"""Granite Speech 4.1-2b-plus (Apache 2.0), through transformers.

The second adapter (D5). It exists to exercise the three capability paths
CrisperWhisper does not -- ``end_only`` timestamps, silence tokens, native
speaker labels -- which is what shows the contract was not written around one
model.

Everything here follows the model card's own usage code (D19), re-read on
2026-09-01. Two facts the earlier note in plan §3 had missed:

* **Timestamps are centiseconds modulo 1000.** The ``[T:N]`` tag rolls over
  every ten seconds; ``unwrap_timestamps`` is the card's own reconstruction.
  Read as plain centiseconds, an adapter is right for the first ten seconds of
  every window and wrong thereafter.
* **Timestamp mode is rated to 3.5 minutes of audio**, ASR and SAA to 9. So
  this is a ``needs_chunking`` model with a window well under the chunker's
  300 s default.

Speaker numbers are ordinals by first appearance and **restart per chunk**;
the card's incremental decoding (``prefix_text``) is what carries them across
segments. What that means for the declaration is settled by the Phase 3 spike.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from pipeline.adapters.base import Adapter, AdapterResult, Word
from pipeline.capabilities import Capabilities

logger = logging.getLogger(__name__)

MODEL_DEFAULT = "ibm-granite/granite-speech-4.1-2b-plus"

# The card's, verbatim. A prompt-steered model; the wording is the interface.
SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
PROMPT_ASR = "<|audio|> can you transcribe the speech into a written format?"
PROMPT_SAA = ("<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding "
              "[Speaker 1]: and [Speaker 2]: tags before speaker turns.")
PROMPT_TS = ("<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag "
             "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]")

# Card: "up to 9 minutes for ASR and SAA, and up to 3.5 minutes for timestamps".
# Windows sit inside those with headroom.
TS_WINDOW_S = 200.0
SAA_WINDOW_S = 500.0
MAX_NEW_TOKENS = {"asr": 2000, "saa": 4000, "ts": 10000}

SILENCE = "_"
_TS_TAG = re.compile(r"\[T:(\d+)\]")
_SPK_TAG = re.compile(r"\[Speaker (\d+)\]:")


def unwrap_timestamps(text: str) -> Tuple[List[Word], bool, str]:
    """Split ``word [T:N] word [T:N] ...`` into words with absolute end times.

    The card's algorithm: N is ``round(t*100) mod 1000``; whenever a tag reads
    lower than the running end, ten seconds have rolled over. Returns the words
    (``start=None`` -- the orchestrator derives starts from the previous token's
    end, which the silence tokens make sound), whether the unwrapped ends were
    monotonic, and any trailing text that carried no tag.
    """
    parts = _TS_TAG.split(text)
    words: List[Word] = []
    last, offset, mono = 0.0, 0.0, True
    for chunk, ts in zip(parts[::2], parts[1::2]):
        token = chunk.strip()
        end = float(ts) / 100.0
        while end + offset < last:
            offset += 10.0
        t_abs = end + offset
        if t_abs < last:
            mono = False
        last = t_abs
        if not token:
            continue
        flags = ("silence",) if token == SILENCE else ()
        words.append(Word(text=token, start=None, end=round(t_abs, 3), flags=flags))
    trailing = parts[-1].strip() if len(parts) % 2 == 1 else ""
    return words, mono, trailing


def parse_speakers(text: str) -> List[Tuple[Optional[str], str]]:
    """``[Speaker 1]: hello [Speaker 2]: hi`` -> [("Speaker 1", "hello"), ("Speaker 2", "hi")].

    Text before the first tag is returned with speaker ``None``.
    """
    parts = _SPK_TAG.split(text)
    out: List[Tuple[Optional[str], str]] = []
    if parts[0].strip():
        out.append((None, parts[0].strip()))
    for num, seg in zip(parts[1::2], parts[2::2]):
        out.append((f"Speaker {num}", seg.strip()))
    return out


def load_audio_16k(path: Union[str, Path], start: float, duration: float):
    """16 kHz mono float32, the model's input. librosa, else soundfile + torchaudio."""
    try:
        import librosa
        wav, _ = librosa.load(str(path), sr=16000, offset=start, duration=duration)
        return wav
    except ImportError:
        import soundfile as sf
        import torch
        import torchaudio
        info = sf.info(str(path))
        wav, _ = sf.read(str(path), start=int(start * info.samplerate),
                         frames=int(duration * info.samplerate), dtype="float32", always_2d=True)
        mono = torch.from_numpy(wav.T).mean(0, keepdim=True)
        return torchaudio.functional.resample(mono, info.samplerate, 16000)[0].numpy()


class Granite41PlusAdapter(Adapter):
    """Declaration and dispatch are filled in once the Phase 3 spike settles
    whether timestamp and speaker modes combine, and how speaker numbering
    survives chunking. The loader and the prompt plumbing below are the card's."""

    capabilities = Capabilities(
        word_timestamps="end_only", verbatim="no", speaker_labels=True,
        silence_tokens=True, longform="needs_chunking", confidence=False, hotwords="trained",
    )

    def __init__(self, model_id: str, device, *, language: str = "en", **options: Any):
        super().__init__(model_id, device, **options)
        self.language = language
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info("Loading Granite Speech: %s on %s (%s)", self.model_id, device, dtype)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, device_map=device, dtype=dtype,
        )
        self.model.eval()

    def _generate(self, audio, prompt: str, max_new_tokens: int, prefix_text: Optional[str] = None) -> str:
        """The card's ``transcribe()``: chat template, processor, greedy, decode new tokens only."""
        import torch

        chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        extra = {"prefix_text": prefix_text} if prefix_text is not None else {}
        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **extra
        )
        device = str(self.model.device)
        inputs = self.processor(prompt_text, audio, device=device, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, num_beams=1)
        new = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new, add_special_tokens=False, skip_special_tokens=True)

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        raise NotImplementedError("dispatch is written after the Phase 3 spike")
