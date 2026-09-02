"""Output formats. Each exposes ``render(turns, stamp, params) -> str``.

Plain text is always emitted (D11); HTML is the default print format; LaTeX is
optional and not built; docx is deferred until a QDA tool is adopted (D16).
"""

from __future__ import annotations

from typing import Any, Dict, List


def fmt_time(seconds: float, offset: float = 0.0, decimals: int = 1) -> str:
    """hh:mm:ss.s in the *original recording's* time when an offset is known.

    Timestamps exist so the audio can be checked when a transcript looks wrong
    (D10). A time relative to a temporary excerpt is useless for that; the
    excerpt offset stage 1 recorded is applied here.
    """
    t = max(0.0, seconds + offset)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:0{3 + decimals}.{decimals}f}"


def header_lines(stamp: Dict[str, Any]) -> List[str]:
    """The stamp as comment lines, for the text formats."""
    r = stamp["render"]
    st = stamp["stats"]
    lines = [
        f"# transcribe_audio — {stamp['profile']} transcript",
        f"# source:    {stamp['source']}"
        + (f"  [{stamp['excerpt']}]" if stamp.get("excerpt") else "")
        + (f"  sha256 {stamp['audio_sha256']}" if stamp.get("audio_sha256") else ""),
        f"# model:     {stamp['model']} @ {stamp['revision']}  ({stamp['mode']} stream)",
        f"# diarizer:  {stamp['diarizer']} @ {stamp['diarizer_revision']}"
        if stamp.get("diarizer") else "# diarizer:  none",
        f"# render:    pause>{r['pause_threshold_s']}s  anchors every {r['anchor_interval_s']:.0f}s  "
        + (f"smoothing≤{r['max_island']} words ({r['words_reassigned']} reassigned)"
           if r["smoothing"] else "smoothing off")
        + (f"  speaker map: {r['speaker_map']}" if r.get("speaker_map") else ""),
        f"# structure: {st['turns']} turns, {st['sentences']} sentences "
        f"({', '.join(f'{k} {v}' for k, v in st['boundaries'].items())}), {st['anchors']} anchors",
        f"# stamp:     hash {stamp['content_hash']}  rendered {stamp['rendered_utc']}  "
        f"pipeline {stamp['pipeline_commit']}{' (dirty)' if stamp.get('pipeline_dirty') else ''}"
        + (f"  stage-1 job {stamp['stage1_job']}" if stamp.get("stage1_job") else ""),
    ]
    if stamp.get("offset_s"):
        lines.append(f"# times:     in the original recording (excerpt offset +{stamp['offset_s']:.0f}s applied)")
    for w in stamp.get("warnings") or []:
        lines.append(f"# WARNING:   {w}")
    lines.append("# the timestamp is the durable reference; line numbers hold only for this render")
    return lines
