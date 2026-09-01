"""The capability contract (D2).

An adapter declares what its model provides; the orchestrator reads the
declaration and fills the gaps. Nothing dispatches on a model *name* -- that
was bug 7, where ``"crisper" in name`` routed a v1 checkpoint into a v2-only
code path and the run succeeded while silently ignoring ``--mode`` and
``--hotwords``.

The declaration is data, so the dispatch decisions it drives (`plan_for`) are a
pure function of it. That is what makes the contract testable against the mock
adapters without downloading a weight.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Literal, Optional, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:  # avoid a circular import at runtime
    from pipeline.adapters.base import Adapter

WordTimestamps = Literal["none", "end_only", "start_end"]
Verbatim = Literal["no", "yes", "selectable"]
Longform = Literal["native", "needs_chunking"]
Hotwords = Literal["no", "untrained", "trained"]

_WORD_TIMESTAMPS = ("none", "end_only", "start_end")
_VERBATIM = ("no", "yes", "selectable")
_LONGFORM = ("native", "needs_chunking")
_HOTWORDS = ("no", "untrained", "trained")


@dataclass(frozen=True)
class Capabilities:
    """What a model provides. Declared per checkpoint, never inferred.

    ``word_timestamps``
        ``"start_end"`` -- both bounds native (CrisperWhisper 2).
        ``"end_only"``  -- word end only; starts must be derived from the
        previous token's end, which is sound *only* when ``silence_tokens``
        is true (Granite 4.1-plus transcribes silences as ``_``). Without
        them a pause folds into the following word.
        ``"none"``      -- no timing; needs forced alignment.
    ``verbatim``
        ``"selectable"`` means a mode parameter exists (CW2's
        ``mode="verbatim"``); ``"yes"`` means always verbatim; ``"no"`` means
        the model normalises disfluencies away and cannot be asked not to.
        Prompting does not count -- measured, it never works (plan §3).
    ``hotwords``
        ``"trained"``   -- the checkpoint was trained with hotword prompts.
        ``"untrained"`` -- the parameter is accepted but the checkpoint was
        never trained on it, so it can *degrade* transcription. The
        ``crisperwhisper`` package raises only a ``UserWarning`` here, which
        is exactly the kind of silent-success this contract exists to catch.
        ``"no"``        -- no such parameter.
    """

    word_timestamps: WordTimestamps
    verbatim: Verbatim
    speaker_labels: bool
    silence_tokens: bool
    longform: Longform
    confidence: bool
    hotwords: Hotwords = "no"

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("word_timestamps", self.word_timestamps, _WORD_TIMESTAMPS),
            ("verbatim", self.verbatim, _VERBATIM),
            ("longform", self.longform, _LONGFORM),
            ("hotwords", self.hotwords, _HOTWORDS),
        ):
            if value not in allowed:
                raise ValueError(
                    f"Capabilities.{name}={value!r} is not one of {allowed}"
                )
        for name, value in (
            ("speaker_labels", self.speaker_labels),
            ("silence_tokens", self.silence_tokens),
            ("confidence", self.confidence),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"Capabilities.{name} must be a bool, got {value!r}")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    """A registry entry: one checkpoint, its adapter, and its declaration.

    ``unsupported`` marks a checkpoint that is deliberately refused. Refusing
    loudly is the point: the alternative is what bug 7 did, which was to route
    it somewhere that ran fine and quietly did the wrong thing.
    """

    model_id: str
    adapter: Optional[Type["Adapter"]] = None
    capabilities: Optional[Capabilities] = None
    defaults: Dict[str, Any] = field(default_factory=dict)
    unsupported: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.unsupported is None and (self.adapter is None or self.capabilities is None):
            raise ValueError(
                f"ModelSpec({self.model_id!r}) must declare both an adapter and "
                "capabilities, or an `unsupported` reason."
            )


@dataclass(frozen=True)
class TranscribePlan:
    """What the orchestrator must do, derived only from a declaration.

    Every field here answers a question the old factory answered by guessing
    from the model name.
    """

    chunking: bool
    derive_starts: bool
    forced_alignment: bool
    needs_diarizer: bool
    timing_source: str  # "native" | "derived" | "aligned"
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def plan_for(
    caps: Capabilities,
    *,
    diarize_requested: bool = True,
    mode: Optional[str] = None,
    hotwords: bool = False,
) -> TranscribePlan:
    """Turn a declaration plus the run's requests into a dispatch plan.

    Pure: no model, no audio, no device. Every mock adapter exercises it.
    """
    warnings: list = []
    errors: list = []

    if caps.word_timestamps == "start_end":
        timing_source, derive_starts, forced_alignment = "native", False, False
    elif caps.word_timestamps == "end_only":
        timing_source, derive_starts, forced_alignment = "derived", True, False
        if not caps.silence_tokens:
            warnings.append(
                "word_timestamps='end_only' without silence_tokens: word starts "
                "are derived from the previous token's end, so every pause folds "
                "into the following word and pause durations are unrecoverable."
            )
    else:
        timing_source, derive_starts, forced_alignment = "aligned", False, True

    # A verbatim request against a model that cannot honour it must fail, not
    # warn: the run would produce cleaned text that looks like a normal
    # transcript and is unusable for coding (D3).
    if mode == "verbatim" and caps.verbatim == "no":
        errors.append(
            "mode='verbatim' requested, but this model declares verbatim='no'. "
            "It normalises disfluencies away and cannot be asked not to; "
            "prompting does not change this (plan §3)."
        )
    if mode is not None and mode != "verbatim" and caps.verbatim == "yes":
        warnings.append(
            f"mode={mode!r} requested, but this model is always verbatim; "
            "the mode argument has no effect."
        )

    if hotwords:
        if caps.hotwords == "no":
            errors.append(
                "hotwords were supplied, but this model declares hotwords='no'."
            )
        elif caps.hotwords == "untrained":
            warnings.append(
                "hotwords were supplied to a checkpoint that was never trained "
                "with hotword prompts. The parameter is accepted and the run "
                "will succeed, but biasing is unsupported here and can degrade "
                "transcription rather than merely doing nothing."
            )

    return TranscribePlan(
        chunking=caps.longform == "needs_chunking",
        derive_starts=derive_starts,
        forced_alignment=forced_alignment,
        needs_diarizer=diarize_requested and not caps.speaker_labels,
        timing_source=timing_source,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
