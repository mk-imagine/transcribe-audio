"""The model registry: exact-match routing, never name-sniffing (retires bug 7).

The old factory decided with ``"crisper" in model_name.lower()``. That routed
``unsloth/crisperwhisper`` -- a **v1** checkpoint -- into the v2 code path,
where ``mode`` and ``hotwords`` are silently ignored: the package warns, the
run exits 0, and the transcript comes back with zero disfluency markers. The
same substring also matched ``kyr0/crisperwhisper-unsloth-mlx``, sending it to
a class that no longer exists.

Three rules make that shape impossible here:

1. **Lookup is exact** (case-folded, since the Hub resolves ids that way).
   An unknown id is refused with the list of known ones, not guessed at.
2. **Known-bad checkpoints are registered as refusals**, with the reason.
   A checkpoint that would run and quietly do the wrong thing is worse than
   one that will not run at all.
3. **The adapter re-verifies on load.** The registry is a claim about a
   checkpoint; the adapter checks the checkpoint itself
   (``CrisperWhisper2Adapter`` reads the tokenizer's version marker), so a
   renamed or mirrored copy of v1 weights is still caught.

Adding a model is a registry entry plus, at most, a thin adapter.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Type

from pipeline.capabilities import Capabilities, ModelSpec

logger = logging.getLogger(__name__)

_CW2 = "pipeline.adapters.crisperwhisper2:CrisperWhisper2Adapter"
_CW2_PRO = "pipeline.adapters.crisperwhisper2:CrisperWhisper2ProAdapter"
_WHISPER = "pipeline.adapters.whisper:WhisperAdapter"
_MOCK = "pipeline.adapters.mock:"

_CW2_CAPS = Capabilities(
    word_timestamps="start_end", verbatim="selectable", speaker_labels=False,
    silence_tokens=False, longform="native", confidence=False, hotwords="untrained",
)
_CW2_PRO_CAPS = Capabilities(
    word_timestamps="start_end", verbatim="selectable", speaker_labels=False,
    silence_tokens=False, longform="native", confidence=False, hotwords="trained",
)
_WHISPER_CAPS = Capabilities(
    word_timestamps="start_end", verbatim="no", speaker_labels=False,
    silence_tokens=False, longform="needs_chunking", confidence=False, hotwords="no",
)

_V1_REFUSAL = (
    "CrisperWhisper v1. The package accepts --mode and --hotwords against a v1 "
    "checkpoint, ignores both, emits no disfluency markers, and still exits 0 "
    "(verified: byte-identical 158-word output for verbatim, verbatim+hotwords "
    "and intended). Use nyralabs/CrisperWhisper2.0_large."
)
_MLX_REFUSAL = (
    "MLX/Apple-Silicon checkpoint. MPS and MLX are dropped (D13): Docker on "
    "macOS cannot expose Metal to Linux containers, so this needs a bare-metal "
    "install for a platform that is never a deployment target. Production is CUDA."
)


def _spec(model_id, adapter, caps, **kw) -> ModelSpec:
    return ModelSpec(model_id=model_id, adapter=adapter, capabilities=caps, **kw)


REGISTRY: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    key = spec.model_id.lower()
    if key in REGISTRY:
        raise ValueError(f"duplicate registry entry for {spec.model_id!r}")
    REGISTRY[key] = spec


# --- CrisperWhisper 2 (default) --------------------------------------------
# The size shorthands the package itself accepts are registered too, so
# `--model large` resolves the same way `CrisperWhisperModel("large")` does.
for _size in ("large", "turbo", "medium", "small"):
    _register(_spec(f"nyralabs/CrisperWhisper2.0_{_size}", _CW2, _CW2_CAPS,
                    defaults={"backend": "auto", "mode": "verbatim"}))
    _register(_spec(f"nyralabs/CrisperWhisper2.0_{_size}_pro", _CW2_PRO, _CW2_PRO_CAPS,
                    defaults={"backend": "auto", "mode": "verbatim"},
                    notes="Commercial licence only. Trained with hotword prompts."))

# --- CrisperWhisper v1 and MLX: registered refusals -------------------------
for _v1 in ("nyrahealth/CrisperWhisper", "unsloth/CrisperWhisper",
            "unsloth/crisperwhisper", "nyrahealth/CrisperWhisper-en"):
    if _v1.lower() not in REGISTRY:
        _register(ModelSpec(model_id=_v1, unsupported=_V1_REFUSAL))
_register(ModelSpec(model_id="kyr0/crisperwhisper-unsloth-mlx", unsupported=_MLX_REFUSAL))

# --- Generic Whisper --------------------------------------------------------
for _w in ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
           "large-v3-turbo"):
    _register(_spec(f"openai/whisper-{_w}", _WHISPER, _WHISPER_CAPS,
                    defaults={"granularity": "word"}))

# --- Planned, not yet implemented ------------------------------------------
# Registered so a Phase 3 model id fails with a sentence instead of a NameError,
# which is what the factory did after its adapter class was deleted.
_register(ModelSpec(
    model_id="ibm-granite/granite-speech-4.1-2b-plus",
    unsupported=(
        "Adapter not implemented yet (Phase 3). Declared shape: word_timestamps="
        "'end_only' (centisecond [T:N] tags marking word ends), silence_tokens=True "
        "(silences transcribed as '_'), speaker_labels=True, longform="
        "'needs_chunking', verbatim='no'."
    ),
))

# --- Mocks: no weights, no ML dependencies ---------------------------------
_MOCKS = {
    "mock/start-end": ("MockStartEndAdapter", "CrisperWhisper 2 shape"),
    "mock/end-only": ("MockEndOnlyAdapter", "Granite shape: end times + silence tokens"),
    "mock/end-only-no-silence": ("MockEndOnlyNoSilenceAdapter",
                                 "unsound derivation; must warn"),
    "mock/no-timestamps": ("MockNoTimestampsAdapter", "must route to forced alignment"),
    "mock/speaker-labels": ("MockSpeakerLabelsAdapter", "no diarizer should be loaded"),
    "mock/failing": ("MockFailingAdapter", "loses 30s; timeline must survive"),
}
for _id, (_cls, _note) in _MOCKS.items():
    _mod = importlib.import_module("pipeline.adapters.mock")
    _register(_spec(_id, _MOCK + _cls, getattr(_mod, _cls).capabilities, notes=_note))


class UnknownModelError(KeyError):
    """The id is not registered. Add an entry rather than guessing at runtime."""


class UnsupportedModelError(RuntimeError):
    """The id is registered as a deliberate refusal, with a reason."""


def known_models(include_unsupported: bool = False) -> List[str]:
    return sorted(
        s.model_id for s in REGISTRY.values()
        if include_unsupported or s.unsupported is None
    )


def resolve(model_id: str) -> ModelSpec:
    """Look up a checkpoint. Exact match only.

    Raises ``UnsupportedModelError`` for a registered refusal and
    ``UnknownModelError`` for anything else -- never a best guess.
    """
    spec = REGISTRY.get(model_id.strip().lower())
    if spec is None:
        raise UnknownModelError(
            f"{model_id!r} is not in the registry. Routing by name pattern is "
            "exactly the bug this replaced, so it is not attempted.\n"
            f"Supported: {', '.join(known_models())}\n"
            "To use another checkpoint, add a ModelSpec to "
            "src/pipeline/registry.py declaring its capabilities, or pass "
            "--adapter to name one explicitly for this run."
        )
    if spec.unsupported:
        raise UnsupportedModelError(f"{spec.model_id} is not supported: {spec.unsupported}")
    return spec


def load_adapter_class(spec: ModelSpec) -> Type:
    """Import the adapter and check it still declares what the registry claims.

    The two declarations are written in different files and drift silently
    otherwise -- and a stale capability declaration misroutes exactly the way a
    stale name pattern did.
    """
    assert spec.adapter is not None
    module_name, _, class_name = str(spec.adapter).partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    declared = getattr(cls, "capabilities", None)
    if declared != spec.capabilities:
        raise RuntimeError(
            f"capability drift for {spec.model_id!r}: registry declares "
            f"{spec.capabilities}, adapter {class_name} declares {declared}. "
            "One of them is wrong; dispatch is driven by the registry, so fix "
            "before running."
        )
    return cls


def adapter_class_by_name(name: str) -> Type:
    """Resolve an explicit ``--adapter`` override.

    An escape hatch for an unregistered checkpoint. It is deliberately explicit
    and it is recorded in the output's provenance (``adapter_source:
    "override"``), so an unusual routing decision is visible in the record
    rather than inferred from a name at runtime.
    """
    candidates = {
        "crisperwhisper2": _CW2,
        "crisperwhisper2-pro": _CW2_PRO,
        "whisper": _WHISPER,
    }
    candidates.update({k.split("/", 1)[1]: _MOCK + v[0] for k, v in _MOCKS.items()})
    target = candidates.get(name.strip().lower())
    if target is None:
        raise UnknownModelError(
            f"unknown adapter {name!r}; choose from {', '.join(sorted(candidates))}"
        )
    module_name, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_name), class_name)
