"""Stage 1 orchestration: declaration in, lossless record out.

Every dispatch decision here is a function of the adapter's declared
capabilities (`plan_for`), never of the model's name. That is the whole of
D2, and the reason bug 7 cannot recur: there is no code path that reads a
model id and infers what the model can do.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline import flags as flagmod
from pipeline import provenance, schema
from pipeline.adapters.base import AdapterResult, Word
from pipeline.capabilities import TranscribePlan, plan_for
from pipeline.diarize import Diarizer
from pipeline.registry import ModelSpec, adapter_class_by_name, load_adapter_class, resolve

logger = logging.getLogger(__name__)


class CapabilityConflict(RuntimeError):
    """The run asks for something the model declares it cannot do.

    An error rather than a warning: the alternative is a transcript that looks
    ordinary and is silently unusable for coding, which is the exact failure
    this contract exists to prevent.
    """


def select_device(requested: Optional[str] = None) -> str:
    """CUDA, else CPU.

    MPS is deliberately absent (D13). Docker on macOS cannot expose Metal to a
    Linux container, so supporting it means a bare-metal install for a platform
    that is never a deployment target -- and the repo already paid for that
    once, in a chunk-level timestamp workaround and an entire MLX code path.
    """
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            return "cuda"
    except ImportError:
        pass
    logger.info("Using CPU")
    return "cpu"


def build_adapter(
    model_id: str,
    device: str,
    *,
    adapter_override: Optional[str] = None,
    **options: Any,
):
    """Resolve a checkpoint to an adapter instance, or refuse with a reason."""
    if adapter_override:
        cls = adapter_class_by_name(adapter_override)
        spec = ModelSpec(
            model_id=model_id, adapter=f"override:{adapter_override}",
            capabilities=cls.capabilities,
            notes="explicit --adapter override; not a registry decision",
        )
        logger.warning(
            "Adapter forced to %r for %s. The registry did not make this "
            "routing decision; it is recorded in the output's provenance.",
            adapter_override, model_id,
        )
        source = "override"
    else:
        spec = resolve(model_id)
        cls = load_adapter_class(spec)
        source = "registry"

    merged = dict(spec.defaults)
    merged.update({k: v for k, v in options.items() if v is not None})
    return cls(model_id, device, **_filtered(cls, merged)), spec, source, merged


def _accepted(cls) -> set:
    import inspect
    return set(inspect.signature(cls.__init__).parameters)


def _filtered(cls, options: Dict[str, Any]) -> Dict[str, Any]:
    """Pass only what the adapter accepts; the rest would be a silent no-op."""
    accepted = _accepted(cls)
    dropped = {k: v for k, v in options.items() if k not in accepted}
    if dropped:
        logger.info("Options not accepted by %s, ignored: %s", cls.__name__, sorted(dropped))
    return {k: v for k, v in options.items() if k in accepted}


def apply_plan(result: AdapterResult, plan: TranscribePlan) -> List[str]:
    """Fill the gaps the declaration says exist. Returns warnings for the record."""
    notes: List[str] = []

    if plan.derive_starts:
        # Valid only because the model emits explicit silence tokens; without
        # them plan_for has already warned that pauses fold into the next word.
        previous_end = 0.0
        for w in result.words:
            if w.start is None:
                w.start = previous_end
            if w.end is not None:
                previous_end = w.end
        notes.append(
            f"derived start times for {len(result.words)} tokens from the "
            "previous token's end"
        )

    if plan.forced_alignment:
        # The dispatch decision is real; the aligner is not built yet. Say so
        # in the record rather than emitting untimed words that look timed.
        notes.append(
            "this model declares word_timestamps='none' and needs forced "
            "alignment, which is not implemented yet (plan §4, align.py). "
            "Tokens are recorded without timings; the text is complete."
        )
        logger.warning(notes[-1])

    for w in result.words:
        w.flags = flagmod.tag(w.text)
    if result.secondary is not None:
        # The second stream is a record in its own right, so it is tagged too --
        # an `intended` stream should come back with no filled pauses, and a
        # flag count that is not zero says the mode did not do what it claims.
        for w in result.secondary.words:
            w.flags = flagmod.tag(w.text)

    counts = flagmod.summarise(w.flags for w in result.words)
    if counts:
        logger.info("Disfluency flags: %s", counts)
    if counts.get("unknown_marker"):
        notes.append(
            f"{counts['unknown_marker']} bracketed token(s) matched no known "
            "marker; they are flagged 'unknown_marker' rather than passed "
            "through as speech. Extend pipeline/flags.py if the tag is real."
        )
    return notes


def transcribe(args) -> Dict[str, Any]:
    """Run stage 1 and return the schema-v1 document."""
    audio_path = Path(args.input_path)
    device = select_device(getattr(args, "device", None))

    adapter, spec, adapter_source, effective = build_adapter(
        args.model, device,
        adapter_override=getattr(args, "adapter", None),
        mode=getattr(args, "mode", None),
        hotwords=getattr(args, "hotwords_list", None) or None,
        granularity=getattr(args, "timestamp_mode", None),
        backend=getattr(args, "backend", None),
        dual_stream=getattr(args, "dual_stream", None) or None,
    )
    caps = adapter.capabilities

    # The mode checked against the declaration is the *effective* one: the
    # registry's default for this checkpoint, overridden by the CLI. Defaulting
    # --mode to "verbatim" in argparse made every generic-Whisper run refuse
    # itself over a mode the user never asked for.
    plan = plan_for(
        caps,
        diarize_requested=not args.no_diarize,
        mode=effective.get("mode"),
        hotwords=bool(getattr(args, "hotwords_list", None)),
        dual_stream=bool(getattr(args, "dual_stream", False)),
    )
    logger.info(
        "Plan: chunking=%s derive_starts=%s forced_alignment=%s diarizer=%s "
        "timing=%s dual_stream=%s",
        plan.chunking, plan.derive_starts, plan.forced_alignment,
        plan.needs_diarizer, plan.timing_source, plan.dual_stream,
    )
    for w in plan.warnings:
        logger.warning(w)
    if not plan.ok:
        for e in plan.errors:
            logger.error(e)
        raise CapabilityConflict(
            "The requested run contradicts the model's declared capabilities; "
            "refusing rather than producing output that looks right. "
            + " ".join(plan.errors)
        )

    duration = _duration(audio_path)

    diarizer = None
    if plan.needs_diarizer:
        diarizer = Diarizer(args.diarizer_model, args.hf_token, device)
        diarizer.load()
    elif not args.no_diarize:
        logger.info(
            "Diarizer not loaded: %s declares speaker_labels=True and attributes "
            "speakers itself.", spec.model_id,
        )

    adapter.load()
    result = adapter.transcribe(audio_path, duration)
    notes = apply_plan(result, plan)
    adapter.close()

    speaker_turns = list(result.speaker_turns)
    if diarizer is not None:
        speaker_turns.extend(diarizer.run(audio_path))

    asr: Dict[str, Any] = {
        "model_id": spec.model_id,
        "adapter": type(adapter).__name__,
        "adapter_source": adapter_source,
        "backend": result.backend,
        "granularity": getattr(adapter, "granularity", "word"),
        "timing_source": plan.timing_source,
        "capabilities": caps.as_dict(),
        "params": result.params,
        "performance": result.performance,
    }
    asr.update(provenance.hf_revision(spec.model_id))
    if result.revision:
        asr["revision"] = result.revision

    doc = schema.build(
        source=_source_block(args, audio_path, duration),
        run=provenance.run_block(device),
        asr=asr,
        diarization=diarizer.provenance() if diarizer is not None else _adapter_diarization(result, spec),
        words=result.words,
        speaker_turns=speaker_turns,
        errors=[e.as_dict() for e in result.errors],
        warnings=list(plan.warnings) + notes + _model_warnings(result),
        text=result.text,
        secondary_stream=_secondary_block(result, plan),
    )
    logger.info("Stage 1 complete: %s", schema.summary(doc))
    return doc


def _secondary_block(result: AdapterResult, plan: TranscribePlan) -> Optional[Dict[str, Any]]:
    if result.secondary is None:
        return None
    return {
        "mode": result.secondary.mode,
        "text": result.secondary.text,
        "words": schema.words_block(result.secondary.words, plan.timing_source),
    }


def _source_block(args, processed: Path, duration: float) -> Dict[str, Any]:
    """Identify the audio that was transcribed, and what it was cut from.

    With --start_time the working file is a temporary excerpt that is deleted
    as soon as the run ends, so recording *its* path and hash names something
    nobody can ever check again. The original is hashed instead, and the
    excerpt bounds are recorded alongside -- word timestamps are relative to
    the excerpt, which stage 2 cannot infer and must not assume.
    """
    original = Path(getattr(args, "source_path", None) or processed)
    block: Dict[str, Any] = {
        "audio_path": str(original.resolve()),
        "audio_sha256": provenance.sha256_file(original),
        "duration_s": round(duration, 3),
    }
    if getattr(args, "start_time", None):
        block["excerpt"] = {
            "start_time": args.start_time,
            "end_time": getattr(args, "end_time", None),
            "offset_s": _hhmmss(args.start_time),
            "timestamps_relative_to": "excerpt",
        }
    return block


def _hhmmss(value: str) -> Optional[float]:
    try:
        parts = [float(p) for p in str(value).split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _adapter_diarization(result: AdapterResult, spec: ModelSpec) -> Optional[Dict[str, Any]]:
    """Provenance for speaker turns the model produced itself (speaker_labels=True)."""
    if not result.diarization:
        return None
    block = dict(result.diarization)
    block.update(provenance.hf_revision(spec.model_id))
    return block


def _model_warnings(result: AdapterResult) -> List[str]:
    return [f"model: {m}" for m in (result.params.get("model_warnings") or [])]


def _duration(audio_path: Path) -> float:
    try:
        import librosa
        d = float(librosa.get_duration(path=str(audio_path)))
    except ImportError:
        import contextlib
        import wave
        with contextlib.closing(wave.open(str(audio_path))) as wf:
            d = wf.getnframes() / float(wf.getframerate())
    logger.info("Audio duration: %.2fs", d)
    return d
