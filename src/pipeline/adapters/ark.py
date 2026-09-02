"""ARK-ASR-0.6B (AutoArk), a §7b dissenter. Card script, verbatim (D19).

Three rules from the card, each of which this project once got wrong (plan §3):
cast **only** ``inputs["audios"]`` to fp16, cap audio at 30 s
(``audio_max_length``), and generate greedily with the model's special tokens
banned. Get any of them wrong and the output collapses into repeated CJK
characters -- ``hpc/logs/ark_48646.log`` has the specimen. Verified 2026-09-01.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

from pipeline.adapters.base import Adapter, AdapterResult, Word
from pipeline.capabilities import Capabilities
from pipeline.chunking import transcribe_windows

logger = logging.getLogger(__name__)

WINDOW_S = 30.0          # the card's documented cap
PROMPT = "Please transcribe this audio."


class ArkAsrAdapter(Adapter):
    capabilities = Capabilities(
        word_timestamps="none", verbatim="no", speaker_labels=False,
        silence_tokens=False, longform="needs_chunking", confidence=False, hotwords="no",
    )

    def __init__(self, model_id: str, device, **options: Any):
        super().__init__(model_id, device, **options)
        self.model = self.processor = self.tokenizer = None
        self.bad_words_ids: List[List[int]] = []
        self._dtype = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading ARK-ASR: %s on %s (%s)", self.model_id, device, self._dtype)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=self._dtype, attn_implementation="sdpa",
        ).to(device).eval()
        tok = self.tokenizer
        eos = tok.eos_token_id
        keep = {eos} if isinstance(eos, int) else set(eos or [])
        bad = set(tok.all_special_ids) - keep
        bad.update(tid for t, tid in tok.get_added_vocab().items()
                   if t.startswith("<") and t.endswith(">") and tid not in keep)
        self.bad_words_ids = [[t] for t in sorted(bad)]

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        assert self.model is not None
        path = str(audio_path)
        params = {"prompt": PROMPT, "window_s": WINDOW_S, "dtype": str(self._dtype),
                  "audio_max_length": 30 * 16000, "do_sample": False, "max_new_tokens": 256,
                  "bad_words_ids": len(self.bad_words_ids)}
        words, errors = transcribe_windows(lambda s, e: self._window(path, s, e), duration, segment_size=WINDOW_S)
        return AdapterResult(words=words, text=" ".join(w.text for w in words), errors=errors,
                             params=params, backend="transformers")

    def _window(self, path: str, start: float, end: float) -> List[Word]:
        import torch
        from pipeline.adapters.granite41plus import load_audio_16k

        audio = load_audio_16k(path, start, end - start)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": audio},
                                              {"type": "text", "text": PROMPT}]}]
        inp = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt", sampling_rate=16000,
            audio_padding="longest", text_kwargs={"padding": "longest"}, audio_max_length=30 * 16000,
        ).to(self.model.device)
        if "audios" in inp:
            inp["audios"] = inp["audios"].to(dtype=self._dtype)      # only the audio tensor
        with torch.inference_mode():
            out = self.model.generate(**inp, do_sample=False, max_new_tokens=256,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      bad_words_ids=self.bad_words_ids)
        text = self.tokenizer.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        return [Word(text=t, start=None, end=None) for t in text.split()]
