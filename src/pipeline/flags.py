"""Machine-identifiable disfluency flags (plan §5).

Only the categories a single token settles on its own. ``repetition`` and
``repair`` need surrounding context and belong to ``annotate.py``, so that
re-tagging with a better detector never means re-transcribing.

Flags are typed, not boolean (D8): the categories degrade independently across
models and coders may treat them differently.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# CrisperWhisper emits fillers as bracketed tokens. Measured over 12 windows the
# inventory was exactly {uh, um, laughter} -- but [UH]/[UM] came back uppercase
# and [laughter] lowercase, so matching is case-insensitive throughout. An
# uppercase-only regex silently misses every vocalization.
_FILLED_PAUSE = {
    "uh", "um", "uhm", "hm", "hmm", "mm", "mhm", "er", "erm", "ah", "eh", "huh",
}
_VOCALIZATION = {
    "laughter", "laugh", "laughs", "cough", "coughs", "breath", "breathing",
    "sigh", "sighs", "sniff", "throat", "clears throat", "applause", "music",
    "noise", "silence", "inaudible", "unintelligible", "crosstalk",
}

_BRACKETED = re.compile(r"^[\(\[<]\s*(?P<body>.*?)\s*[\)\]>]$")
# A partial word is cut off mid-utterance: "the p-", "de- developmental".
_PARTIAL = re.compile(r"[^\W\d_]-$", re.UNICODE)


def tag(text: str) -> Tuple[str, ...]:
    """Flags for one token. Never raises; unknown markers are flagged, not dropped."""
    token = (text or "").strip()
    if not token:
        return ()

    m = _BRACKETED.match(token)
    if m:
        body = m.group("body").strip().lower()
        if body in _FILLED_PAUSE:
            return ("filled_pause",)
        if body in _VOCALIZATION:
            return ("vocalization",)
        # A tag inventory that grows must announce itself rather than passing
        # through as ordinary speech.
        return ("unknown_marker",)

    stripped = token.strip("\"'“”‘’.,;:!?)(")
    if _PARTIAL.search(stripped):
        return ("partial_word",)
    return ()


def summarise(flag_lists) -> dict:
    """Counts per flag, for logging and for the run's own summary."""
    counts: dict = {}
    for flags in flag_lists:
        for f in flags:
            counts[f] = counts.get(f, 0) + 1
    return dict(sorted(counts.items()))
