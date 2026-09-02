"""Turn-structured plain text: speaker labels, timestamps, and -- in the
coding profile -- one numbered line per sentence.

A sentence is the numbered unit rather than a physical line: physical lines
depend on the page width, and the number must survive a re-flow. Continuation
lines are indented under the number, the way a legal transcript wraps.

Wrapping never breaks inside a token. textwrap's default splits "self-report"
at the hyphen when it lands on a line end, which in a verbatim coding
transcript would put a line ending in "self-" -- indistinguishable from a
partial word, which is a disfluency category coders look for.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from render import RenderParams, Turn
from render.formats import fmt_time, header_lines


def render(turns: List[Turn], stamp: Dict[str, Any], params: RenderParams) -> str:
    prof = params.profile
    offset = stamp.get("offset_s") or 0.0
    numbered = prof.line_numbers
    total = sum(len(t.sentences) for t in turns)
    num_w = max(3, len(str(total)))
    indent = " " * (num_w + 3) if numbered else "    "
    body_w = max(40, params.width - len(indent))

    out: List[str] = header_lines(stamp)
    out.append("")
    def pieces_of(t: Turn, s) -> List[str]:
        ps = []
        for w in s.words:
            if w.i in t.anchors:
                ps.append(f"[{fmt_time(w.start or 0.0, offset)}]")
            ps.append(w.text)
            if "proper_noun_candidate" in w.flags:
                # §7b: a flag is worth more to a coder than a confident wrong spelling.
                ps.append("[NAME?]")
        return ps

    n = 0
    for t in turns:
        out.append(f"[{fmt_time(t.start or 0.0, offset)}] {t.speaker or 'UNKNOWN'}")
        if numbered:
            # Coding: one numbered line per sentence, continuation lines indented.
            for s in t.sentences:
                n += 1
                text = " ".join(pieces_of(t, s))
                wrapped = textwrap.wrap(text, width=body_w, break_on_hyphens=False,
                                        break_long_words=False) or [""]
                out.append(f"{n:>{num_w}}  " + wrapped[0])
                out.extend(indent + line for line in wrapped[1:])
        else:
            # Lecture: the turn is one paragraph. Sentence boundaries still exist
            # underneath -- anchors snap to them -- but a pause-split fragment
            # on its own line reads as broken prose, which this profile is not.
            text = " ".join(" ".join(pieces_of(t, s)) for s in t.sentences)
            out.extend(textwrap.wrap(text, width=body_w, initial_indent=indent,
                                     subsequent_indent=indent, break_on_hyphens=False,
                                     break_long_words=False))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"
