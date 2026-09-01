"""The adapter interface every model implements.

An adapter's job is to turn audio into words. It does not clean, filter,
assign speakers, or decide anything the orchestrator can decide from the
declaration (D3: stage 1 is verbatim and lossless).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pipeline.capabilities import Capabilities


@dataclass
class Word:
    """One token with its timing. ``start`` may be None when the model gives
    only end times; the orchestrator derives it and records how."""

    text: str
    start: Optional[float]
    end: Optional[float]
    conf: Optional[float] = None
    speaker: Optional[str] = None
    flags: tuple = ()


@dataclass
class ErrorRange:
    """A stretch of audio that produced no transcript.

    Kept in the record rather than dropped: a hole that is not in the file is
    a hole nobody finds. One lecture lost 12.7% of its audio this way (bug 13).
    """

    start: float
    end: float
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "message": self.message}


@dataclass
class AdapterResult:
    words: List[Word] = field(default_factory=list)
    text: str = ""
    intended_text: Optional[str] = None
    speaker_turns: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[ErrorRange] = field(default_factory=list)
    # Exactly what was passed to the model, for the provenance block. Recording
    # the invocation is what makes a benchmark reproducible -- three of this
    # project's wrong conclusions came from an invocation nobody wrote down.
    params: Dict[str, Any] = field(default_factory=dict)
    revision: Optional[str] = None
    backend: Optional[str] = None

    @property
    def failed_ranges(self) -> List[tuple]:
        return [(e.start, e.end) for e in self.errors]


class Adapter(ABC):
    """Base for every model adapter.

    ``capabilities`` is a class attribute so the registry can declare and the
    orchestrator can dispatch without instantiating -- and so a mock can lie
    about a capability on purpose to exercise a path no real model reaches.
    """

    capabilities: Capabilities

    def __init__(self, model_id: str, device: Union[str, "object"], **options: Any):
        self.model_id = model_id
        self.device = device
        self.options = options

    @abstractmethod
    def load(self) -> None:
        """Load weights. Must fail loudly if the checkpoint does not match the
        declared capabilities."""

    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        ...

    def close(self) -> None:  # pragma: no cover - most adapters need nothing
        pass
