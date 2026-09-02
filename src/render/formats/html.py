"""HTML with print CSS: the default print format (D11).

Zero dependencies. CSS counters number the sentences, ``@page`` sets the
coding margin (D23), and ``break-inside: avoid`` keeps a turn on one page.
The stylesheet is inlined, so the file is self-contained and prints from any
browser with Cmd+P.
"""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Any, Dict, List

from render import RenderParams, RWord, Turn
from render.formats import fmt_time

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _css(profile_name: str) -> str:
    base = (_TEMPLATES / "base.css").read_text()
    prof = (_TEMPLATES / f"{profile_name}.css").read_text()
    return base + "\n" + prof


def _word(w: RWord) -> str:
    text = _html.escape(w.text)
    if w.is_marker or "filled_pause" in w.flags or "vocalization" in w.flags:
        return f'<span class="marker">{text}</span>'
    if "partial_word" in w.flags:
        return f'<span class="partial">{text}</span>'
    return text


def _stamp_block(stamp: Dict[str, Any]) -> str:
    r, st = stamp["render"], stamp["stats"]
    rows = [
        ("source", f"{stamp['source']}" + (f" [{stamp['excerpt']}]" if stamp.get("excerpt") else "")
                   + (f" · sha256 {stamp['audio_sha256']}" if stamp.get("audio_sha256") else "")),
        ("model", f"{stamp['model']} @ {stamp['revision']} ({stamp['mode']} stream)"),
        ("diarizer", f"{stamp['diarizer']} @ {stamp['diarizer_revision']}" if stamp.get("diarizer") else "none"),
        ("render", f"pause&gt;{r['pause_threshold_s']}s · anchors every {r['anchor_interval_s']:.0f}s · "
                   + (f"smoothing ≤{r['max_island']} words ({r['words_reassigned']} reassigned)"
                      if r["smoothing"] else "smoothing off")
                   + (f" · speaker map {_html.escape(str(r['speaker_map']))}" if r.get("speaker_map") else "")),
        ("structure", f"{st['turns']} turns · {st['sentences']} sentences · {st['anchors']} anchors"),
        ("stamp", f"hash {stamp['content_hash']} · rendered {stamp['rendered_utc']} · "
                  f"pipeline {stamp['pipeline_commit']}{' (dirty)' if stamp.get('pipeline_dirty') else ''}"
                  + (f" · stage-1 job {stamp['stage1_job']}" if stamp.get("stage1_job") else "")),
    ]
    if stamp.get("offset_s"):
        rows.append(("times", f"in the original recording (excerpt offset +{stamp['offset_s']:.0f}s applied)"))
    parts = ["<header class=\"stamp\">", f"<h1>{_html.escape(stamp['profile'])} transcript</h1>", "<dl>"]
    for k, v in rows:
        parts.append(f"<dt>{k}</dt><dd>{v}</dd>")
    for w in stamp.get("warnings") or []:
        parts.append(f'<dt class="warning">warning</dt><dd class="warning">{_html.escape(w)}</dd>')
    parts.append("<dt>note</dt><dd>the timestamp is the durable reference; line numbers hold only for this render</dd>")
    parts.append("</dl></header>")
    return "\n".join(parts)


def render(turns: List[Turn], stamp: Dict[str, Any], params: RenderParams) -> str:
    prof = params.profile
    offset = stamp.get("offset_s") or 0.0
    out: List[str] = [
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        f"<title>{_html.escape(prof.name)} transcript · {_html.escape(str(stamp.get('source') or ''))}</title>",
        "<style>", _css(prof.name), "</style>", "</head>",
        f'<body class="profile-{prof.name}">', _stamp_block(stamp), "<main>",
    ]
    for t in turns:
        ts = fmt_time(t.start or 0.0, offset)
        out.append(f'<section class="turn" data-start="{ts}">')
        out.append(f'<h2 class="turn-head"><span class="ts">{ts}</span> '
                   f'<span class="spk">{_html.escape(t.speaker or "UNKNOWN")}</span></h2>')
        for s in t.sentences:
            pieces = []
            for w in s.words:
                if w.i in t.anchors:
                    pieces.append(f'<span class="anchor">[{fmt_time(w.start or 0.0, offset)}]</span>')
                pieces.append(_word(w))
            st = fmt_time(s.start or 0.0, offset)
            out.append(f'<p class="line" data-t="{st}">{" ".join(pieces)}</p>')
        out.append("</section>")
    out += ["</main>", "</body>", "</html>"]
    return "\n".join(out) + "\n"
