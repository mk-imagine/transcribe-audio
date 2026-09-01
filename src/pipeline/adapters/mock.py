"""Capability-declaring fakes. No weights, no ML dependencies, no network.

These exist to prove the contract generalises rather than having been written
around one model (plan §7). Each mock declares a different combination and
returns deterministic words, so the orchestrator paths that are hardest to
reach with real models -- ``end_only`` start derivation, forced-alignment
gap-fill, ``needs_chunking``, unrecoverable-range timeline preservation --
are exercised on any machine in milliseconds.

Determinism matters: the words are a pure function of the duration, so a
golden fixture regenerated a year from now is byte-identical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

from pipeline.adapters.base import Adapter, AdapterResult, ErrorRange, Stream, Word
from pipeline.capabilities import Capabilities

logger = logging.getLogger(__name__)

# A short verbatim-looking script with the shapes stage 2 has to handle: filled
# pauses, a partial word, a vocalization, terminal punctuation, and a repetition.
_SCRIPT = [
    "So", "[UM]", "the", "the", "first", "thing", "is", "that", "we", "look",
    "at", "the", "p-", "process", "here.", "[UH]", "Right,", "and", "then",
    "Courchesne", "showed", "the", "commissure", "was", "intact.", "[laughter]",
]
_WORD_S = 0.35
_GAP_S = 0.05
_PAUSE_EVERY = 8       # a longer gap every N words, so pause rules have something to find
_PAUSE_S = 0.6


def _script_words(duration: float, *, with_silences: bool = False) -> List[Word]:
    """Lay the script down on the timeline until the duration is filled."""
    words: List[Word] = []
    t = 0.0
    i = 0
    while t + _WORD_S <= duration:
        text = _SCRIPT[i % len(_SCRIPT)]
        words.append(Word(text=text, start=round(t, 3), end=round(t + _WORD_S, 3)))
        t += _WORD_S
        gap = _PAUSE_S if (i + 1) % _PAUSE_EVERY == 0 else _GAP_S
        if gap > _GAP_S and with_silences and t + gap <= duration:
            # Granite-style explicit silence token: this is what makes deriving
            # a start from the previous token's end sound.
            words.append(Word(text="_", start=round(t, 3), end=round(t + gap, 3)))
        t += gap
        i += 1
    return words


class _MockBase(Adapter):
    """Shared plumbing. Subclasses differ only in their declaration."""

    with_silences = False
    fail_ranges: tuple = ()

    def __init__(self, model_id: str, device, *, mode: str = "verbatim",
                 dual_stream: bool = False, **options: Any):
        super().__init__(model_id, device, **options)
        self.mode = mode
        self.dual_stream = dual_stream
        self.loaded = False

    def load(self) -> None:
        logger.info("Loading mock adapter %s (no weights)", self.model_id)
        self.loaded = True

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        assert self.loaded, "load() must be called before transcribe()"
        words = self._shape(_script_words(duration, with_silences=self.with_silences))
        errors = [
            ErrorRange(s, e, "mock: simulated unrecoverable range")
            for s, e in self.fail_ranges
            if s < duration
        ]
        if errors:
            lo, hi = errors[0].start, errors[-1].end
            words = [w for w in words if not (w.start is not None and lo <= w.start < hi)]
        secondary = None
        if self.dual_stream:
            # The clean rendering: same timeline, markers dropped -- which is
            # what distinguishes the two streams and what stage 2's two profiles
            # actually differ over.
            clean = [w for w in words if not w.text.startswith("[")]
            secondary = Stream(
                mode="intended" if self.mode == "verbatim" else "verbatim",
                text=" ".join(w.text for w in clean),
                words=clean,
            )
        return AdapterResult(
            words=words,
            text=" ".join(w.text for w in words),
            secondary=secondary,
            speaker_turns=self._turns(duration),
            errors=errors,
            params={"mock": True, "model_id": self.model_id, "mode": self.mode},
            revision="mock-0",
            backend="mock",
        )

    def _shape(self, words: List[Word]) -> List[Word]:
        return words

    def _turns(self, duration: float) -> List[dict]:
        return []


class MockStartEndAdapter(_MockBase):
    """The CrisperWhisper 2 shape: native start and end, native long-form."""

    capabilities = Capabilities(
        word_timestamps="start_end", verbatim="selectable", speaker_labels=False,
        silence_tokens=False, longform="native", confidence=False, hotwords="untrained",
    )


class MockEndOnlyAdapter(_MockBase):
    """The Granite 4.1-plus shape: end times only, with explicit silence tokens.

    Start derivation is sound here precisely *because* the silences are
    tokenised -- that coupling is the reason `silence_tokens` is in the contract.
    """

    capabilities = Capabilities(
        word_timestamps="end_only", verbatim="no", speaker_labels=True,
        silence_tokens=True, longform="needs_chunking", confidence=False, hotwords="no",
    )
    with_silences = True

    def _shape(self, words: List[Word]) -> List[Word]:
        for w in words:
            w.start = None      # the model reports only the end of each word
        return words

    def _turns(self, duration: float) -> List[dict]:
        return []


class MockEndOnlyNoSilenceAdapter(MockEndOnlyAdapter):
    """The unsound combination: end times with no silence tokens.

    Nothing forbids a model from being built this way, so the contract must not
    reject it at declaration time -- it must warn that every pause folds into
    the following word. This mock is how that warning gets exercised.
    """

    capabilities = Capabilities(
        word_timestamps="end_only", verbatim="no", speaker_labels=False,
        silence_tokens=False, longform="needs_chunking", confidence=False, hotwords="no",
    )
    with_silences = False


class MockNoTimestampsAdapter(_MockBase):
    """No timing at all: the orchestrator must route this to forced alignment."""

    capabilities = Capabilities(
        word_timestamps="none", verbatim="yes", speaker_labels=False,
        silence_tokens=False, longform="native", confidence=False, hotwords="no",
    )

    def _shape(self, words: List[Word]) -> List[Word]:
        for w in words:
            w.start = w.end = None
        return words


class MockSpeakerLabelsAdapter(_MockBase):
    """Declares its own speaker attribution, so no diarizer should be loaded."""

    capabilities = Capabilities(
        word_timestamps="start_end", verbatim="no", speaker_labels=True,
        silence_tokens=False, longform="native", confidence=True, hotwords="no",
    )

    def _shape(self, words: List[Word]) -> List[Word]:
        for n, w in enumerate(words):
            w.speaker = f"Speaker {1 + (n // 12) % 2}"
            w.conf = 0.9
        return words


class MockFailingAdapter(_MockBase):
    """Loses a stretch of audio, so the timeline-preservation and non-zero-exit
    paths are testable without waiting for a real model to misbehave (bug 13)."""

    capabilities = Capabilities(
        word_timestamps="start_end", verbatim="selectable", speaker_labels=False,
        silence_tokens=False, longform="needs_chunking", confidence=False, hotwords="no",
    )
    fail_ranges = ((30.0, 60.0),)
