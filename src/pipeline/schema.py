"""Raw JSON schema v1 (plan §5): the lossless stage-1 record.

Stage 1 is verbatim and lossless (D3). Nothing is cleaned, filtered, or
assigned here -- anything recomputable from this file is recomputed in stage 2,
so a pause threshold or a speaker correction never means re-queuing a GPU job.

What replaced what:

* the old flat ``[{start, end, text, speaker}]`` list carried no provenance at
  all, and baked speaker assignment into the only copy of the data;
* ``speaker_turns`` are now stored **raw**, exactly as the diarizer emitted
  them, and words carry no speaker unless the *model itself* attributed them
  (``speaker_source: "asr"``);
* ``timing_source`` records where a timestamp came from, so "native" and
  "derived from the previous token's end" are never confused;
* ``errors`` keep unrecoverable ranges in the timeline instead of dropping
  them, because a hole that is not in the file is a hole nobody finds;
* ``warnings`` keep every caveat the run raised -- an untrained hotword prompt,
  a start time derived without silence tokens -- in the file rather than only
  in a job log that gets rotated away.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

TIMING_SOURCES = ("native", "derived", "aligned", "none")
SPEAKER_SOURCES = (None, "asr")
GRANULARITIES = ("word", "chunk")


def build(
    *,
    source: Dict[str, Any],
    run: Dict[str, Any],
    asr: Dict[str, Any],
    words: Iterable[Any],
    diarization: Optional[Dict[str, Any]] = None,
    speaker_turns: Optional[List[Dict[str, Any]]] = None,
    intended_text: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
    text: str = "",
) -> Dict[str, Any]:
    """Assemble a schema-v1 document. ``words`` are `adapters.base.Word`."""
    doc: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "run": run,
        "asr": asr,
        "diarization": diarization,
        "words": [
            {
                "i": i,
                "text": w.text,
                "start": _round(w.start),
                "end": _round(w.end),
                "timing_source": getattr(w, "timing_source", asr.get("timing_source")),
                "speaker": w.speaker,
                "speaker_source": "asr" if w.speaker is not None else None,
                "conf": w.conf,
                "flags": list(w.flags),
            }
            for i, w in enumerate(words)
        ],
        "text": text,
        "speaker_turns": speaker_turns or [],
        "intended_text": intended_text,
        "errors": errors or [],
        "warnings": list(warnings or []),
    }
    return doc


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 3)


def write(path: Path, doc: Dict[str, Any]) -> None:
    problems = validate(doc)
    if problems:
        # Write anyway -- a flawed record beats no record when a GPU job has
        # already run -- but never let the flaw go unmentioned.
        for p in problems:
            logger.error("schema v1 violation: %s", p)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    logger.info("Wrote raw JSON (schema v%s): %s", doc.get("schema_version"), path)


def read(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        doc = json.load(fh)
    problems = validate(doc)
    if problems:
        raise ValueError(f"{path} is not a valid schema-v1 document: {problems}")
    return doc


def validate(doc: Dict[str, Any]) -> List[str]:
    """Structural check. Returns problems rather than raising, so a caller can
    decide whether a flawed record is still worth keeping."""
    problems: List[str] = []

    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version is {doc.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")

    for key in ("source", "run", "asr", "words", "speaker_turns", "errors", "warnings"):
        if key not in doc:
            problems.append(f"missing top-level key {key!r}")

    src = doc.get("source") or {}
    for key in ("audio_path", "audio_sha256", "duration_s"):
        if key not in src:
            problems.append(f"source is missing {key!r}")

    asr = doc.get("asr") or {}
    for key in ("model_id", "revision", "capabilities", "params", "granularity"):
        if key not in asr:
            problems.append(f"asr is missing {key!r}")
    if asr.get("granularity") not in GRANULARITIES:
        problems.append(f"asr.granularity {asr.get('granularity')!r} not in {GRANULARITIES}")

    run = doc.get("run") or {}
    for key in ("created_utc", "device", "pipeline_version"):
        if key not in run:
            problems.append(f"run is missing {key!r}")

    words = doc.get("words")
    if not isinstance(words, list):
        problems.append("words must be a list")
        return problems

    last_start = None
    for n, w in enumerate(words):
        if w.get("i") != n:
            problems.append(f"words[{n}].i is {w.get('i')!r}, expected {n}")
        if not isinstance(w.get("text"), str):
            problems.append(f"words[{n}].text must be a string")
        if w.get("timing_source") not in TIMING_SOURCES:
            problems.append(
                f"words[{n}].timing_source {w.get('timing_source')!r} not in {TIMING_SOURCES}"
            )
        if w.get("speaker_source") not in SPEAKER_SOURCES:
            problems.append(f"words[{n}].speaker_source {w.get('speaker_source')!r} invalid")
        if (w.get("speaker") is None) != (w.get("speaker_source") is None):
            problems.append(f"words[{n}] speaker and speaker_source disagree")
        s, e = w.get("start"), w.get("end")
        if s is not None and e is not None and e < s:
            problems.append(f"words[{n}] ends ({e}) before it starts ({s})")
        if s is not None:
            if last_start is not None and s < last_start:
                problems.append(f"words[{n}] starts at {s}, before words[{n-1}] at {last_start}")
            last_start = s

    for n, t in enumerate(doc.get("speaker_turns") or []):
        for key in ("start", "end", "speaker"):
            if key not in t:
                problems.append(f"speaker_turns[{n}] is missing {key!r}")
    for n, e in enumerate(doc.get("errors") or []):
        for key in ("start", "end", "message"):
            if key not in e:
                problems.append(f"errors[{n}] is missing {key!r}")
    return problems


def summary(doc: Dict[str, Any]) -> str:
    """One line for the log: enough to notice a run that went wrong."""
    words = doc.get("words") or []
    flags: Dict[str, int] = {}
    for w in words:
        for f in w.get("flags") or []:
            flags[f] = flags.get(f, 0) + 1
    lost = sum(e["end"] - e["start"] for e in (doc.get("errors") or []))
    return (
        f"{len(words)} tokens, flags={flags or '{}'}, "
        f"{len(doc.get('speaker_turns') or [])} speaker turns, "
        f"{len(doc.get('errors') or [])} error range(s) ({lost:.0f}s lost), "
        f"{len(doc.get('warnings') or [])} warning(s)"
    )
