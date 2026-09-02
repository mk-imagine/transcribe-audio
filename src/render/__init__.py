"""Stage 2: the raw JSON record in, a printable transcript out.

Everything here is a pure function over a schema-v1 document (plan §6). No
model, no GPU, no audio: re-rendering is instant, so a pause threshold or a
speaker correction is tuned against real transcripts rather than guessed at,
and never means re-queuing a cluster job (D1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RWord:
    """A word as the renderer sees it: timed, speaker-assigned, flagged."""

    i: int
    text: str
    start: Optional[float]
    end: Optional[float]
    flags: Tuple[str, ...] = ()
    speaker: Optional[str] = None        # after assignment, smoothing and the name map
    speaker_raw: Optional[str] = None    # straight from max-overlap, before either

    @property
    def is_marker(self) -> bool:
        return self.text.startswith("[") and self.text.endswith("]")


@dataclass
class Sentence:
    words: List[RWord]
    boundary: str                        # "punct" | "pause" | "speaker" | "end"

    @property
    def speaker(self) -> Optional[str]:
        return self.words[0].speaker if self.words else None

    @property
    def start(self) -> Optional[float]:
        return next((w.start for w in self.words if w.start is not None), None)

    @property
    def end(self) -> Optional[float]:
        return next((w.end for w in reversed(self.words) if w.end is not None), None)


@dataclass
class Turn:
    """A contiguous stretch by one speaker -- the primary render unit (D9)."""

    speaker: Optional[str]
    sentences: List[Sentence] = field(default_factory=list)
    anchors: Dict[int, str] = field(default_factory=dict)   # word index -> "sentence" | "word"

    @property
    def start(self) -> Optional[float]:
        return next((s.start for s in self.sentences if s.start is not None), None)

    @property
    def end(self) -> Optional[float]:
        return next((s.end for s in reversed(self.sentences) if s.end is not None), None)

    @property
    def words(self) -> List[RWord]:
        return [w for s in self.sentences for w in s.words]


@dataclass(frozen=True)
class Profile:
    """The two render profiles (plan §6)."""

    name: str
    stream: str                 # "verbatim" | "intended"
    line_numbers: bool
    anchor_interval: float      # seconds between within-turn anchors
    margin: str                 # "coding" (wide right column, D23) | "normal"
    keep_turns_together: bool   # avoid a page break inside a turn


PROFILES: Dict[str, Profile] = {
    "coding": Profile(
        name="coding", stream="verbatim", line_numbers=True,
        anchor_interval=30.0, margin="coding", keep_turns_together=True,
    ),
    "lecture": Profile(
        name="lecture", stream="intended", line_numbers=False,
        anchor_interval=60.0, margin="normal", keep_turns_together=False,
    ),
}


@dataclass
class RenderParams:
    profile: Profile
    pause_threshold: float = 0.5
    anchor_interval: Optional[float] = None    # None -> the profile's default
    smoothing: bool = True
    max_island: int = 2
    speaker_map_path: Optional[str] = None
    apply_offset: bool = True
    width: int = 88

    @property
    def interval(self) -> float:
        return self.anchor_interval if self.anchor_interval is not None else self.profile.anchor_interval


def select_stream(doc: Dict[str, Any], wanted: str) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Pick the words for the requested stream.

    Returns (words, mode_used, warning). The record's primary stream is whatever
    mode stage 1 ran in; the secondary, if present, is the other. A profile that
    asks for a stream the record does not carry gets the primary and a warning
    that goes into the header stamp -- a lecture render of a verbatim-only
    record is still useful, but it must say what it is.
    """
    primary_mode = (doc.get("asr", {}).get("params") or {}).get("mode") or "verbatim"
    if primary_mode == wanted:
        return doc["words"], primary_mode, None
    sec = doc.get("secondary_stream")
    if sec and sec.get("mode") == wanted and sec.get("words"):
        return sec["words"], wanted, None
    return doc["words"], primary_mode, (
        f"profile wants the {wanted!r} stream, but this record carries only "
        f"{primary_mode!r}; rendered from that instead. Re-run stage 1 with "
        "--dual_stream to get both."
    )


def to_rwords(words: List[Dict[str, Any]]) -> List[RWord]:
    return [
        RWord(i=w["i"], text=w["text"], start=w.get("start"), end=w.get("end"),
              flags=tuple(w.get("flags") or ()))
        for w in words
    ]
