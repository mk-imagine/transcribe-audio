"""Parakeet-TDT-0.6b-v3 (NVIDIA NeMo lineage) through its transformers port, a §7b dissenter.

The card's ``AutoModelForTDT`` usage, unchunked -- but in **30 s windows without
overlap**. Measured 2026-09-01 (plan §7b): on 90 s the port returned 141 words
where every other model gave ~157, 159 in 30 s sub-windows; and the pipeline's
``chunk_length_s`` stride duplicated words at every boundary (210). The card's
long-form and streaming recipes are NeMo's and do not apply to the port.
A transducer needs no overlap. The card also documents word timestamps via
``durations``; not used here yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

from pipeline.adapters.base import Adapter, AdapterResult, ErrorRange, Word
from pipeline.capabilities import Capabilities
from pipeline.chunking import transcribe_windows

logger = logging.getLogger(__name__)

WINDOW_S = 30.0


class ParakeetTdtAdapter(Adapter):
    capabilities = Capabilities(
        word_timestamps="none", verbatim="no", speaker_labels=False,
        silence_tokens=False, longform="needs_chunking", confidence=False, hotwords="no",
    )

    def __init__(self, model_id: str, device, **options: Any):
        super().__init__(model_id, device, **options)
        self.model = self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForTDT, AutoProcessor

        device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        logger.info("Loading Parakeet-TDT: %s on %s", self.model_id, device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForTDT.from_pretrained(self.model_id, dtype="auto", device_map=device).eval()

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        assert self.model is not None
        path = str(audio_path)
        params = {"window_s": WINDOW_S, "overlap": 0, "decoding": "generate() default"}
        words, errors = transcribe_windows(lambda s, e: self._window(path, s, e), duration, segment_size=WINDOW_S)
        if not words and not errors and duration > 5.0:
            # A model that emits nothing for a stretch of speech has failed,
            # whatever its exit status. Recording it as a lost range makes the
            # run exit 1 -- the first ARK run through this adapter produced zero
            # words and reported success.
            errors = [ErrorRange(0.0, duration, "model produced no text for the whole file")]
        return AdapterResult(words=words, text=" ".join(w.text for w in words), errors=errors,
                             params=params, backend="transformers")

    def _window(self, path: str, start: float, end: float) -> List[Word]:
        import torch
        from pipeline.adapters.granite41plus import load_audio_16k

        audio = load_audio_16k(path, start, end - start)
        inputs = self.processor([audio], sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, return_dict_in_generate=True)
        dec = self.processor.decode(out.sequences, skip_special_tokens=True)
        text = (dec[0] if isinstance(dec, list) else dec).strip()
        return [Word(text=t, start=None, end=None) for t in text.split()]
