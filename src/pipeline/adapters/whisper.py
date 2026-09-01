"""Generic Whisper checkpoints through `transformers.pipeline`.

For `openai/whisper-*` and lookalikes only. CrisperWhisper does **not** come
through here (D17): the pipeline has no ``mode`` parameter, so verbatim output
cannot be requested and the model looks like an ordinary cleaned-text ASR.
That mistake produced a ten-model comparison with the wrong conclusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from pipeline.adapters.base import Adapter, AdapterResult, Word
from pipeline.capabilities import Capabilities
from pipeline.chunking import SEGMENT_SIZE, transcribe_windows

logger = logging.getLogger(__name__)

# Whisper's decoder can fall into a repetition trap: measured on real audio it
# repeated "anterior commissure" 37 times in one 90s window, duplicating 59% of
# its 8-grams. These make it detect the trap and retry the segment hotter.
#
# Deliberately NOT no_repeat_ngram_size: banning repeated n-grams would also
# delete genuine verbatim repetitions ("as as as a"), which are a disfluency
# category this project needs to keep.
LOOP_GUARDS = {
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}


def _transformers_version() -> tuple:
    import transformers
    parts = transformers.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


class WhisperAdapter(Adapter):
    capabilities = Capabilities(
        word_timestamps="start_end",
        verbatim="no",          # normalises disfluencies; prompting does not help
        speaker_labels=False,
        silence_tokens=False,
        longform="needs_chunking",
        confidence=False,
        hotwords="no",
    )

    def __init__(self, model_id: str, device, *, granularity: str = "word",
                 language: str = "en", **options: Any):
        super().__init__(model_id, device, **options)
        if granularity not in ("word", "chunk"):
            raise ValueError(f"granularity must be 'word' or 'chunk', got {granularity!r}")
        # "chunk" is a VRAM concession, not a capability difference: word-level
        # alignments OOM on an 8GB card for whisper-large-v3-class models. It
        # coarsens the record, so the schema records which was produced.
        self.granularity = granularity
        self.language = language
        self.pipe = None
        self.supports_loop_guards = False

    def load(self) -> None:
        # Imported here, not at module scope: an optional backend must not make
        # this module unimportable. A module-scope pyannote import already took
        # the whole script down once (bug 15), and the mock adapters exist so
        # dispatch can be exercised on a machine with no ML stack at all.
        import torch
        from transformers import pipeline

        self.supports_loop_guards = _transformers_version() >= (4, 39)
        device_str = str(self.device)
        if device_str.startswith("cpu"):
            dtype = torch.float32
        elif device_str.startswith("cuda") and torch.cuda.is_bf16_supported():
            # Ampere and newer: bf16 costs the same memory as fp16 but keeps
            # fp32's exponent range, so long jobs cannot silently overflow.
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        logger.info("Loading ASR model: %s on %s (%s)", self.model_id, self.device, dtype)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            torch_dtype=dtype,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
        )

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        assert self.pipe is not None, "load() must be called before transcribe()"
        path = str(audio_path)
        params = {
            "language": self.language,
            "granularity": self.granularity,
            "chunk_length_s": 30,
            "stride_length_s": 5,
            "segment_size_s": SEGMENT_SIZE,
            "loop_guards": LOOP_GUARDS if self.supports_loop_guards else None,
        }

        words, errors = transcribe_windows(
            lambda s, e: self._window(path, s, e), duration
        )
        return AdapterResult(
            words=words,
            text=" ".join(w.text for w in words),
            errors=errors,
            params=params,
            backend="transformers",
        )

    def _window(self, path: str, start: float, end: float) -> List[Word]:
        import librosa

        audio, sr = librosa.load(path, sr=16000, offset=start, duration=end - start)
        if len(audio) == 0:
            return []
        result = self.pipe(
            {"raw": audio, "sampling_rate": sr},
            return_timestamps="word" if self.granularity == "word" else True,
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
                **(LOOP_GUARDS if self.supports_loop_guards else {}),
            },
        )
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        if not chunks:
            text = (result.get("text", "") if isinstance(result, dict) else "").strip()
            if not text:
                return []
            logger.warning("No chunks returned; recording the window as one token.")
            return [Word(text=text, start=0.0, end=end - start)]

        words: List[Word] = []
        for c in chunks:
            ts = c.get("timestamp") or (None, None)
            text = (c.get("text") or "").strip()
            if not text:
                continue
            # A None bound stays None. Substituting the window edge (bug 6)
            # invents a timestamp and hides the gap from everything downstream.
            words.append(Word(text=text, start=ts[0], end=ts[1]))
        return words
