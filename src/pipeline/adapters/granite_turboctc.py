"""Granite Speech 5.0 TurboCTC (IBM), a §7b dissenter. Card usage, verbatim.

CTC is frame-synchronous and must **not** be chunked through the pipeline's
seq2seq ``chunk_length_s``: that returned 68 words of ~166 on a 90 s window
(bug 14). ``AutoModelForCTC`` on a whole window is correct. Windows here are
for memory only, and a CTC model does not drop content at their edges the way
a seq2seq model does -- it may split a word (bug 3), never lose a stretch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

from pipeline.adapters.base import Adapter, AdapterResult, Word
from pipeline.capabilities import Capabilities
from pipeline.chunking import transcribe_windows

logger = logging.getLogger(__name__)

WINDOW_S = 300.0


class GraniteTurboCtcAdapter(Adapter):
    capabilities = Capabilities(
        word_timestamps="none", verbatim="no", speaker_labels=False,
        silence_tokens=False, longform="needs_chunking", confidence=False, hotwords="no",
    )

    def __init__(self, model_id: str, device, **options: Any):
        super().__init__(model_id, device, **options)
        self.model = self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        logger.info("Loading Granite TurboCTC: %s on %s", self.model_id, device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(self.model_id, device_map=device).eval()

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        assert self.model is not None
        path = str(audio_path)
        params = {"window_s": WINDOW_S, "chunk_length_s": None, "decoding": "CTC greedy"}
        words, errors = transcribe_windows(lambda s, e: self._window(path, s, e), duration, segment_size=WINDOW_S)
        return AdapterResult(words=words, text=" ".join(w.text for w in words), errors=errors,
                             params=params, backend="transformers")

    def _window(self, path: str, start: float, end: float) -> List[Word]:
        import torch
        from pipeline.adapters.granite41plus import load_audio_16k

        audio = load_audio_16k(path, start, end - start)
        sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor([audio], sampling_rate=sr, device=self.model.device)
        inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return [Word(text=t, start=None, end=None) for t in text.split()]
