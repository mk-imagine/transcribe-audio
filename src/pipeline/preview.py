"""A readable text preview of the raw JSON.

**Not the renderer.** Stage 2 (`render.py`, Phase 2) owns turn grouping,
within-turn anchors, line numbering, print CSS and the two profiles. This
exists so a stage-1 run is still inspectable by eye today, and it derives
everything from the JSON -- which is the point: the record is authoritative
and every view of it is recomputable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MAX_LINE_CHARS = 88


def _fmt(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--:--.---"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _speaker_at(t: Optional[float], turns: List[Dict[str, Any]]) -> Optional[str]:
    """Containing turn, else the nearest one.

    Diarizer turns do not tile the timeline -- they leave gaps at pauses -- so a
    containment-only lookup returns None for any word in a gap and breaks the
    line there. Stage 2 does this properly, with smoothing inside pause-bounded
    runs; falling back to the nearest turn is enough to keep a preview readable.
    """
    if t is None or not turns:
        return None
    for turn in turns:
        if turn["start"] <= t < turn["end"]:
            return turn["speaker"]
    nearest = min(turns, key=lambda x: min(abs(t - x["start"]), abs(t - x["end"])))
    return nearest["speaker"]


def render(doc: Dict[str, Any], *, timestamps: bool = True) -> str:
    words = doc.get("words") or []
    turns = sorted(doc.get("speaker_turns") or [], key=lambda t: t["start"])
    asr = doc.get("asr") or {}
    run = doc.get("run") or {}

    # Version-stamp the preview: a printout and a later re-render must visibly
    # differ, or a coder citing "line 47" is citing a different sentence.
    head = [
        "# stage-1 preview -- not the stage-2 render",
        f"# model:      {asr.get('model_id')} @ {asr.get('revision')}",
        f"# mode:       {(asr.get('params') or {}).get('mode') or 'n/a'}   "
        f"granularity: {asr.get('granularity')}   timing: {asr.get('timing_source')}",
        f"# pipeline:   {(run.get('pipeline_version') or {}).get('commit')}"
        f"{'  (dirty)' if (run.get('pipeline_version') or {}).get('dirty') else ''}",
        f"# created:    {run.get('created_utc')}",
    ]
    for w in doc.get("warnings") or []:
        head.append(f"# WARNING:    {w}")
    for e in doc.get("errors") or []:
        head.append(f"# ERROR:      {_fmt(e['start'])}-{_fmt(e['end'])} {e['message']}")
    head.append("")

    lines: List[str] = []
    buf: List[str] = []
    buf_start: Optional[float] = None
    buf_speaker: Optional[str] = None

    def flush() -> None:
        nonlocal buf, buf_start, buf_speaker
        if not buf:
            return
        prefix = f"[{_fmt(buf_start)}] " if timestamps else ""
        if buf_speaker:
            prefix += f"({buf_speaker}): "
        lines.append(prefix + " ".join(buf))
        buf, buf_start, buf_speaker = [], None, None

    for w in words:
        speaker = w.get("speaker") or _speaker_at(w.get("start"), turns)
        if buf and speaker != buf_speaker:
            flush()
        if not buf:
            buf_start, buf_speaker = w.get("start"), speaker
        buf.append(w["text"])
        text_len = sum(len(t) + 1 for t in buf)
        if text_len >= MAX_LINE_CHARS and w["text"].endswith((".", "?", "!")):
            flush()
        elif text_len >= MAX_LINE_CHARS * 2:
            flush()
    flush()

    return "\n".join(head + lines) + "\n"
