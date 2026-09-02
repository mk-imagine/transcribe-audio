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

**Spike, 2026-09-01 (plan §3):** the timestamp and speaker-attribution modes
do not combine in one prompt -- a fused prompt yields timestamp output with a
single vestigial ``[Speaker 1]:`` and no attribution. So the adapter runs two
passes per window: timestamps for the words, SAA for the turns, aligned by
token sequence. The SAA turns go into the record's ``speaker_turns`` slot,
where stage 2 treats them exactly like a diarizer's.

Speaker numbers are ordinals by first appearance and restart per window; the
card's incremental decoding (``prefix_text``, audio accumulating, capped at
nine minutes) is what carries them across segments and cannot span an 80-min
file. Labels are therefore namespaced by window when there is more than one
(``w3:Speaker 1``), and the render-time speaker map merges them.

Timestamp mode is lowercase and unpunctuated; ASR and SAA modes are not. The
record carries the timestamped words as the model emitted them (D3), so
stage 2's punctuation rule gets little from this model and the silence tokens
carry the segmentation instead -- which is what they are for.
"""

from __future__ import annotations

import difflib
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pipeline.adapters.base import Adapter, AdapterResult, Word
from pipeline.capabilities import Capabilities
from pipeline.chunking import transcribe_windows

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


_NORM = re.compile(r"[^a-z0-9]+")


def _norm(token: str) -> str:
    return _NORM.sub("", token.lower())


def align_speakers(words: List[Word], segments: List[Tuple[Optional[str], str]]) -> List[str]:
    """Give each timestamped word the speaker of its SAA counterpart.

    The two passes transcribe the same audio with the same model, so their
    token sequences are near-identical; ``SequenceMatcher`` on normalised
    tokens pairs them. A word with no match inherits the previous word's
    speaker. Silence tokens are skipped in the match and inherit too. Returns
    one label per word, never None: a pass that emitted no tag at all is a
    single speaker.
    """
    spoken = [(k, _norm(w.text)) for k, w in enumerate(words) if "silence" not in w.flags and _norm(w.text)]
    saa: List[Tuple[str, str]] = []
    for label, seg in segments:
        for tok in seg.split():
            n = _norm(tok)
            if n:
                saa.append((label or "Speaker 1", n))
    labels: List[Optional[str]] = [None] * len(words)
    if spoken and saa:
        sm = difflib.SequenceMatcher(a=[t for _, t in spoken], b=[t for _, t in saa], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for d in range(i2 - i1):
                    labels[spoken[i1 + d][0]] = saa[j1 + d][0]
    first = next((lab for lab in labels if lab), (saa[0][0] if saa else "Speaker 1"))
    out: List[str] = []
    cur = first
    for lab in labels:
        if lab:
            cur = lab
        out.append(cur)
    return out


def turns_from_labels(words: List[Word], labels: List[str]) -> List[Dict[str, Any]]:
    """Collapse per-word labels into time ranges. A word's start is the
    previous token's end -- the same derivation the orchestrator applies -- and
    the token before a turn is usually a silence, so a turn starts where the
    silence ended."""
    turns: List[Dict[str, Any]] = []
    prev_end = 0.0
    for w, lab in zip(words, labels):
        start = prev_end
        if w.end is not None:
            prev_end = w.end
        if "silence" in w.flags:
            continue
        if turns and turns[-1]["speaker"] == lab:
            turns[-1]["end"] = w.end if w.end is not None else turns[-1]["end"]
        else:
            turns.append({"start": start, "end": w.end if w.end is not None else start, "speaker": lab})
    return turns


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
    capabilities = Capabilities(
        word_timestamps="end_only", verbatim="no", speaker_labels=True,
        silence_tokens=True, longform="needs_chunking", confidence=False,
        # Keyword-list biasing is a documented, evaluated feature of this
        # checkpoint (the card reports keyword F1), unlike CrisperWhisper's.
        hotwords="trained",
    )

    def __init__(self, model_id: str, device, *, language: str = "en",
                 hotwords: Optional[List[str]] = None, **options: Any):
        super().__init__(model_id, device, **options)
        self.language = language
        self.hotwords = list(hotwords or [])
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._turns: List[Dict[str, Any]] = []
        self._perf: List[Tuple[float, float]] = []
        self._notes: List[str] = []

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
        assert self.model is not None, "load() must be called before transcribe()"
        self._turns, self._perf, self._notes = [], [], []
        suffix = f" Keywords: {', '.join(self.hotwords)}" if self.hotwords else ""
        n_windows = max(1, math.ceil(duration / TS_WINDOW_S))
        params = {
            "language": self.language,
            "passes": ["ts", "saa"],
            "prompt_ts": PROMPT_TS + suffix,
            "prompt_saa": PROMPT_SAA + suffix,
            "hotwords": self.hotwords or None,
            "window_s": TS_WINDOW_S,
            "windows": n_windows,
            "max_new_tokens": {"ts": MAX_NEW_TOKENS["ts"], "saa": MAX_NEW_TOKENS["saa"]},
            "decoding": "greedy",
            "speaker_labels_namespaced": n_windows > 1,
        }
        path = str(audio_path)
        words, errors = transcribe_windows(
            lambda s, e: self._window(path, s, e, namespaced=n_windows > 1),
            duration, segment_size=TS_WINDOW_S,
        )
        if n_windows > 1:
            self._notes.append(
                f"speaker labels are per-window ordinals over {n_windows} windows of "
                f"{TS_WINDOW_S:.0f}s: the model numbers speakers by first appearance and "
                "restarts each window, so w1:Speaker 1 and w2:Speaker 1 may or may not be the "
                "same person. Merge them with a speaker map at render time."
            )
        if self._notes:
            params["model_warnings"] = self._notes
        gen = sum(dt for _, dt in self._perf)
        perf = ({"processing_time_s": round(gen, 3), "realtime_factor": round(duration / gen, 1)}
                if gen else {})
        return AdapterResult(
            words=words,
            text=" ".join(w.text for w in words if "silence" not in w.flags),
            speaker_turns=self._turns,
            errors=errors,
            params=params,
            backend="transformers",
            performance=perf,
            diarization={
                "model_id": self.model_id,
                "source": "asr",
                "params": {"prompt": PROMPT_SAA + suffix, "window_s": TS_WINDOW_S,
                           "alignment": "SequenceMatcher on normalised tokens, per window"},
            },
        )

    def _window(self, path: str, start: float, end: float, *, namespaced: bool) -> List[Word]:
        """Both passes on one window. Returns window-relative words; turns are
        shifted to absolute time and kept on the adapter."""
        audio = load_audio_16k(path, start, end - start)
        suffix = f" Keywords: {', '.join(self.hotwords)}" if self.hotwords else ""
        t0 = time.perf_counter()
        ts_text = self._generate(audio, PROMPT_TS + suffix, MAX_NEW_TOKENS["ts"])
        saa_text = self._generate(audio, PROMPT_SAA + suffix, MAX_NEW_TOKENS["saa"])
        dt = time.perf_counter() - t0
        self._perf.append((end - start, dt))

        words, mono, trailing = unwrap_timestamps(ts_text)
        k = int(round(start / TS_WINDOW_S)) + 1
        if not mono:
            self._notes.append(f"window {k}: timestamps not monotonic after unwrap")
        if trailing:
            self._notes.append(f"window {k}: {len(trailing.split())} trailing token(s) carried no "
                               f"timestamp and were dropped: {trailing[:60]!r}")
        labels = align_speakers(words, parse_speakers(saa_text))
        if namespaced:
            labels = [f"w{k}:{lab}" for lab in labels]
        for t in turns_from_labels(words, labels):
            self._turns.append({"start": t["start"] + start, "end": t["end"] + start, "speaker": t["speaker"]})
        logger.info("Window %.0f-%.0fs: %d tokens (%d silences), %d speaker(s), %.1fs",
                    start, end, len(words), sum(1 for w in words if "silence" in w.flags),
                    len(set(labels)), dt)
        return words
