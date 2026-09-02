"""The fluent view and the three-dissenter conjunction (plan §7b, D18).

Cross-model disagreement localises proper-noun garbles without a glossary:
transcribe with the primary plus three small models from independent lineages,
and flag any token where **all** dissenters disagree. The conjunction is what
supplies precision -- one dissenter flags ~9% of tokens, three flag ~1%.

Comparison runs on a *fluent view* of the primary: a verbatim transcript
carries disfluencies the fluent dissenters cannot match, so raw comparison
would flag every ``[UH]``. An index map carries flags back to the original
tokens, so disfluency tokens cannot be flagged -- correct, since no dissenter
can confirm them.

Two refinements, both measured 2026-09-01 on the spike outputs (residue over
two windows 2.1% -> 0.95%, recall on the name-dense window 5/5):

* **pronoun-'s expansion** -- ``them's`` -> ``them is``, as ``gonna`` already is.
* **adjacency mask** -- a candidate beside a dropped marker or partial is not
  flagged; the dissenters reshape the word next to a disfluency as often as
  the disfluency itself. Exempt: a capitalised, non-sentence-initial token,
  because hesitation precedes retrieval of a hard name -- ``Eric Korshane [UH]``
  is the canonical shape -- and the mask would otherwise hide exactly the
  tokens the detector exists for.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BRACKET = re.compile(r"^[\[\(<].*[\]\)>]$")
PUNCT = re.compile(r"[\"“”‘’.,;:!?()\[\]{}…]")
_NUM = re.compile(r"^\d+([.,]\d+)?(st|nd|rd|th|s)?$")
_NUMWORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy "
    "eighty ninety hundred thousand million billion".split()
)
CONTRACTIONS = {
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of",
    "sorta": "sort of", "lemme": "let me", "dunno": "don't know", "'cause": "because",
    "cause": "because", "y'all": "you all", "'em": "them", "outta": "out of", "lotta": "lot of",
}
PRONOUN_S = {
    "them's": "them is", "there's": "there is", "that's": "that is", "what's": "what is",
    "he's": "he is", "she's": "she is", "who's": "who is", "here's": "here is",
    "one's": "one is", "where's": "where is", "how's": "how is", "it's": "it is",
}
TERMINAL = (".", "?", "!")


@dataclass
class FluentView:
    tokens: List[str]                 # normalised tokens
    index: List[int]                  # fluent position -> original token index
    dropped: Set[int] = field(default_factory=set)   # original indices of markers / partials


def is_disfluency_token(text: str) -> bool:
    t = text.strip()
    return bool(t) and (bool(BRACKET.match(t)) or (t.endswith("-") and len(t) > 1))


def fluent_view(tokens: Sequence[str], *, expand_pronoun_s: bool = True) -> FluentView:
    out: List[str] = []
    idx: List[int] = []
    dropped: Set[int] = set()
    prev: Optional[str] = None
    for i, raw in enumerate(tokens):
        t = raw.strip()
        if not t or is_disfluency_token(t):
            dropped.add(i)
            continue
        t = t.lower()
        core = t.strip(PUNCT.pattern)
        t = CONTRACTIONS.get(core, t)
        if expand_pronoun_s:
            t = PRONOUN_S.get(core, t)
        for piece in re.split(r"[-–—/\s]+", PUNCT.sub("", t)):
            piece = piece.replace("'", "").replace("’", "")
            if not piece:
                continue
            if _NUM.match(piece) or piece in _NUMWORDS:
                piece = "<num>"
            if piece == prev:              # adjacent repetition: "the the"
                continue
            out.append(piece)
            idx.append(i)
            prev = piece
    return FluentView(tokens=out, index=idx, dropped=dropped)


def unmatched(primary: FluentView, other: FluentView) -> Set[int]:
    """Fluent positions of ``primary`` that no ``equal`` opcode against ``other`` covers."""
    sm = difflib.SequenceMatcher(a=primary.tokens, b=other.tokens, autojunk=False)
    matched: Set[int] = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            matched.update(range(i1, i2))
    return set(range(len(primary.tokens))) - matched


def sentence_initial(tokens: Sequence[str], i: int) -> bool:
    j = i - 1
    while j >= 0 and is_disfluency_token(tokens[j]):
        j -= 1
    return j < 0 or tokens[j].rstrip("\"')]”’").endswith(TERMINAL)


def name_like(tokens: Sequence[str], i: int) -> bool:
    """Capitalised and not sentence-initial: the shape of a proper noun in a
    verbatim transcript, and the shape the adjacency mask must never hide."""
    core = tokens[i].strip(PUNCT.pattern)
    return core[:1].isupper() and not sentence_initial(tokens, i)


@dataclass
class Candidate:
    index: int
    token: str
    dissenters_disagreeing: int
    adjacent_disfluency: bool
    name_like: bool
    masked: bool
    allowlisted: bool = False

    @property
    def flagged(self) -> bool:
        return not self.masked and not self.allowlisted

    def as_dict(self) -> Dict[str, Any]:
        return {"i": self.index, "token": self.token, "dissenters": self.dissenters_disagreeing,
                "adjacent_disfluency": self.adjacent_disfluency, "name_like": self.name_like,
                "masked": self.masked, "allowlisted": self.allowlisted}


def normalise_terms(terms: Sequence[str]) -> Set[str]:
    """The fluent form of each word of each term, so "Bob Knight" allows both
    ``bob`` and ``knight`` and matches however the primary punctuated them."""
    out: Set[str] = set()
    for term in terms or ():
        out.update(fluent_view(str(term).split()).tokens)
    out.discard("<num>")
    return out


def conjunction(
    primary_tokens: Sequence[str],
    dissenter_tokens: Sequence[Sequence[str]],
    *,
    expand_pronoun_s: bool = True,
    mask_adjacent: bool = True,
    exempt_name_like: bool = True,
    min_dissenters: Optional[int] = None,
    allow_terms: Sequence[str] = (),
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """Every candidate, masked or not, plus counts. Nothing is dropped: a
    masked or allowlisted candidate is recorded as such, so a re-run with
    different rules is a pure re-annotation (D1).

    ``allow_terms`` are words the user deliberately biased the primary toward
    (its hotwords). The dissenters cannot spell a rare name any better than
    they can confirm it, so a correctly biased ``Courchesne`` would otherwise be
    flagged four times as suspect. It is recorded as a candidate, marked
    allowlisted, and not flagged.
    """
    allow = normalise_terms(allow_terms)
    if not dissenter_tokens:
        raise ValueError("at least one dissenter is required")
    need = len(dissenter_tokens) if min_dissenters is None else min_dissenters
    pv = fluent_view(primary_tokens, expand_pronoun_s=expand_pronoun_s)
    per = [unmatched(pv, fluent_view(d, expand_pronoun_s=expand_pronoun_s)) for d in dissenter_tokens]

    votes: Dict[int, int] = {}                  # original index -> dissenters disagreeing
    for u in per:
        for pos in u:
            oi = pv.index[pos]
            votes[oi] = votes.get(oi, 0) + 1
    # A split token ("self-report" -> two fluent tokens) can collect a vote
    # from each half; cap at the number of dissenters.
    cands: List[Candidate] = []
    for oi in sorted(votes):
        n = min(votes[oi], len(dissenter_tokens))
        if n < need:
            continue
        adjacent = (oi - 1) in pv.dropped or (oi + 1) in pv.dropped
        nl = name_like(primary_tokens, oi)
        masked = bool(mask_adjacent and adjacent and not (exempt_name_like and nl))
        allowed = bool(allow) and any(t in allow for t in fluent_view([primary_tokens[oi]]).tokens)
        cands.append(Candidate(oi, primary_tokens[oi], n, adjacent, nl, masked, allowed))

    stats = {
        "primary_tokens": len(primary_tokens),
        "fluent_tokens": len(pv.tokens),
        "dropped_disfluency_tokens": len(pv.dropped),
        "dissenters": len(dissenter_tokens),
        "unmatched_per_dissenter": [len(u) for u in per],
        "candidates": len(cands),
        "masked": sum(1 for c in cands if c.masked),
        "allowlisted": sum(1 for c in cands if c.allowlisted and not c.masked),
        "flagged": sum(1 for c in cands if c.flagged),
        "flagged_pct_of_fluent": round(100 * sum(1 for c in cands if c.flagged) / max(1, len(pv.tokens)), 2),
        "rules": {"expand_pronoun_s": expand_pronoun_s, "mask_adjacent": mask_adjacent,
                  "exempt_name_like": exempt_name_like, "min_dissenters": need,
                  "allow_terms": sorted(allow)},
    }
    return cands, stats
