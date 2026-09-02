"""Repetition and repair tagging (plan §5) -- the two disfluency categories a
single token cannot settle, derived from context and from the tags stage 1
already emitted (§3: "repair: derivable from marker positions + repetitions").

Deterministic rules, no model, no dependencies. The tags are metadata on the
record's words; the text is never touched (D3). Their value is consistency and
counting -- every coder works from the same inventory, and a hesitation *rate*
is computable -- more than reading, since a verbatim transcript already shows
"the the" as written.

What is tagged is the **abandoned** material: the first copy of a repetition,
the reparandum of a repair. The restart stands untagged. So one flag is one
event, and a per-speaker rate is a count.

Shapes recognised:

* **repetition** -- a span of 1-3 words immediately restated, fillers allowed in
  between: ``the the``, ``not not mirror``, ``the [UH] the``, ``as as as a as a``
  (greedy, left to right). Standard contractions expand before comparison, so
  ``You're you are`` is a restatement, not a mystery.
* **repair, completed partial** -- a ``partial_word`` whose stem the next word
  continues: ``de- developmental``, ``p- process``.
* **repair, substitution** -- a 2-3 word span restarted with its first word kept
  and one later word changed: ``I went I drove``. Skipped across a sentence
  boundary, where ``I came. I saw`` is parallelism, not repair.

Editing terms ("I mean", "sorry") are not modelled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

_PUNCT = re.compile(r"[\"“”‘’.,;:!?()\[\]{}…]")
_TERMINAL = (".", "?", "!")
_BRACKET = re.compile(r"^[\[\(<].*[\]\)>]$")

CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "it's": "it is", "that's": "that is", "there's": "there is", "he's": "he is", "she's": "she is",
    "what's": "what is", "who's": "who is", "here's": "here is", "let's": "let us",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will", "it'll": "it will",
    "i'd": "i would", "you'd": "you would", "we'd": "we would", "they'd": "they would",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "can not",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
}


def _norm(text: str) -> str:
    """Lowercase, punctuation gone, apostrophes kept for the contraction lookup,
    trailing hyphen kept so a partial matches a partial and never a whole word."""
    t = text.strip().lower().replace("’", "'")
    partial = t.endswith("-") and len(t) > 1
    t = _PUNCT.sub("", t).rstrip("-")
    return t + ("-" if partial else "")


def _expand(text: str) -> str:
    n = _norm(text)
    return CONTRACTIONS.get(n, n).replace("'", "")


def _is_filler(flags: Sequence[str], text: str) -> bool:
    return "filled_pause" in flags or "vocalization" in flags or bool(_BRACKET.match(text.strip()))


def _terminal(text: str) -> bool:
    return text.rstrip("\"')]”’").endswith(_TERMINAL)


@dataclass
class Event:
    kind: str                    # "repetition" | "repair"
    shape: str                   # "restatement" | "completed_partial" | "substitution"
    abandoned: List[int]         # positions tagged
    restart: List[int]           # the material that stands

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "shape": self.shape, "abandoned": self.abandoned, "restart": self.restart}


@dataclass
class Result:
    events: List[Event] = field(default_factory=list)
    flags: Dict[int, List[str]] = field(default_factory=dict)   # position -> flags to add

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
            out[f"{e.kind}:{e.shape}"] = out.get(f"{e.kind}:{e.shape}", 0) + 1
        return dict(sorted(out.items()))


def tag(tokens: Sequence[Tuple[str, Sequence[str]]]) -> Result:
    """``tokens`` is ``[(text, flags), ...]`` for the spoken tokens in order.
    Returns the events and, per position, the flags to add."""
    n = len(tokens)
    text = [t for t, _ in tokens]
    flags = [tuple(f) for _, f in tokens]
    res = Result()

    def filler(k: int) -> bool:
        return _is_filler(flags[k], text[k])

    def span_ok(a: int, b: int) -> bool:      # [a, b) with no fillers and inside bounds
        return b <= n and all(not filler(k) for k in range(a, b))

    def next_spoken(k: int) -> int:
        while k < n and filler(k):
            k += 1
        return k

    def add(kind: str, shape: str, a: List[int], b: List[int]) -> None:
        res.events.append(Event(kind, shape, a, b))
        for k in a:
            res.flags.setdefault(k, [])
            if kind not in res.flags[k]:
                res.flags[k].append(kind)

    i = 0
    while i < n:
        if filler(i):
            i += 1
            continue
        found = None
        # --- restatement: A == B, up to 3 words each, fillers allowed between ---
        for la in (3, 2, 1):
            if not span_ok(i, i + la) or any(_terminal(text[k]) for k in range(i, i + la - 1)):
                continue
            j = next_spoken(i + la)
            ea = " ".join(_expand(text[k]) for k in range(i, i + la))
            for lb in (3, 2, 1):
                if not span_ok(j, j + lb):
                    continue
                eb = " ".join(_expand(text[k]) for k in range(j, j + lb))
                if ea and ea == eb:
                    found = ("repetition", "restatement", list(range(i, i + la)), list(range(j, j + lb)), la)
                    break
            if found:
                break
        # --- repair: completed partial ---
        if not found and "partial_word" in flags[i]:
            j = next_spoken(i + 1)
            stem = _norm(text[i]).rstrip("-")
            if j < n and stem:
                nxt = _norm(text[j])
                if nxt.startswith(stem) and len(nxt) > len(stem) and not nxt.endswith("-"):
                    found = ("repair", "completed_partial", [i], [j], 1)
        # --- repair: substitution, first word kept, one later word changed ---
        if not found:
            for la in (2, 3):
                if not span_ok(i, i + la) or any(_terminal(text[k]) for k in range(i, i + la)):
                    continue
                j = next_spoken(i + la)
                if not span_ok(j, j + la):
                    continue
                a = [_expand(text[k]) for k in range(i, i + la)]
                b = [_expand(text[k]) for k in range(j, j + la)]
                if a[0] == b[0] and a != b and sum(x != y for x, y in zip(a, b)) == 1 \
                        and not any(_terminal(text[k]) for k in range(i, i + la)):
                    found = ("repair", "substitution", list(range(i, i + la)), list(range(j, j + la)), la)
                    break
        if found:
            kind, shape, a, b, la = found
            add(kind, shape, a, b)
            i += la          # the restart may itself be abandoned again: "as as as"
        else:
            i += 1
    return res
