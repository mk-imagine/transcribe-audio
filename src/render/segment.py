"""Sentences, turns, and within-turn anchors (plan §6).

Sentence boundary = terminal punctuation ∨ pause > threshold ∨ speaker change.
Punctuation is the primary signal -- measured, it alone yields a sentence every
5 s -- and the pause rule supplements it.

**Pause threshold, re-derived 2026-09-01 on the fixture.** §6's provisional
0.3 s was measured through `_adjust_pauses` (bug 8), which collapsed small
gaps. On raw CrisperWhisper timestamps 72% of inter-word gaps are exactly
0.000 -- the Viterbi aligner partitions the timeline, so that is its nature,
not a bug -- and the rest are real. At 0.3 s a pause boundary lands every
2.3 s, fragmenting; at **0.5 s** every 4.8 s, matching punctuation's cadence.
0.5 is the default. It is a flag, and re-rendering is instant.
"""

from __future__ import annotations

from typing import List, Optional

from render import RWord, Sentence, Turn

TERMINAL = (".", "?", "!")
_CLOSERS = "\"')]”’»"


def is_terminal(word: RWord) -> bool:
    """Ends a sentence: terminal punctuation, and not a bracketed marker."""
    if word.is_marker:
        return False
    return word.text.rstrip(_CLOSERS).endswith(TERMINAL)


def gap(a: RWord, b: RWord) -> Optional[float]:
    if a.end is None or b.start is None:
        return None
    return b.start - a.end


def sentences(words: List[RWord], pause_threshold: float) -> List[Sentence]:
    """Split into sentences. Every word lands in exactly one, in order."""
    out: List[Sentence] = []
    cur: List[RWord] = []
    for k, w in enumerate(words):
        cur.append(w)
        nxt = words[k + 1] if k + 1 < len(words) else None
        boundary = None
        if nxt is None:
            boundary = "end"
        elif nxt.speaker != w.speaker:
            boundary = "speaker"
        elif is_terminal(w):
            boundary = "punct"
        else:
            g = gap(w, nxt)
            if g is not None and g > pause_threshold:
                boundary = "pause"
        if boundary:
            out.append(Sentence(words=cur, boundary=boundary))
            cur = []
    return out


def turns(sents: List[Sentence]) -> List[Turn]:
    """Group consecutive same-speaker sentences. The diarizer splits one
    speaker's stretch at every pause (59 turns, one main speaker, in the
    fixture); a turn here is the contiguous stretch, not the diarizer's segment."""
    out: List[Turn] = []
    for s in sents:
        if out and out[-1].speaker == s.speaker:
            out[-1].sentences.append(s)
        else:
            out.append(Turn(speaker=s.speaker, sentences=[s]))
    return out


def place_anchors(turn: Turn, interval: float, snap_window: Optional[float] = None) -> None:
    """Within-turn anchors every ``interval`` seconds, snapped to a sentence start.

    A long turn stays citable (D9) without line numbers, which are stable only
    while the render is (§6). Each target time takes the nearest unused
    sentence start within ``snap_window`` (default interval/2); a run-on with
    no boundary near the target falls back to the nearest word start, so a
    forty-second sentence still gets a timestamp inside it.
    """
    turn.anchors = {}
    start, end = turn.start, turn.end
    if interval <= 0 or start is None or end is None or end - start <= interval:
        return
    window = interval / 2 if snap_window is None else snap_window

    # Candidates: sentence starts after the turn's own first sentence.
    sent_starts = [(s.start, s.words[0].i) for s in turn.sentences[1:] if s.start is not None]
    word_starts = [(w.start, w.i) for w in turn.words if w.start is not None]
    used_words = set()
    last_t = start
    target = start + interval
    while target < end:
        pick = None
        near = [(abs(t - target), t, i) for t, i in sent_starts
                if t > last_t and i not in used_words and abs(t - target) <= window]
        if near:
            _, t, i = min(near)
            pick, kind = (t, i), "sentence"
        else:
            near = [(abs(t - target), t, i) for t, i in word_starts
                    if t > last_t and i not in used_words]
            if near:
                _, t, i = min(near)
                pick, kind = (t, i), "word"
        if pick is None:
            break
        t, i = pick
        turn.anchors[i] = kind
        used_words.add(i)
        last_t = t
        target = max(target + interval, t + interval / 2)


def segment(words: List[RWord], pause_threshold: float, anchor_interval: float) -> List[Turn]:
    ts = turns(sentences(words, pause_threshold))
    for t in ts:
        place_anchors(t, anchor_interval)
    return ts


def stats(ts: List[Turn]) -> dict:
    """Counts for the header stamp and the checks."""
    sents = [s for t in ts for s in t.sentences]
    by_boundary: dict = {}
    for s in sents:
        by_boundary[s.boundary] = by_boundary.get(s.boundary, 0) + 1
    return {
        "turns": len(ts),
        "sentences": len(sents),
        "boundaries": dict(sorted(by_boundary.items())),
        "anchors": sum(len(t.anchors) for t in ts),
        "speakers": sorted({t.speaker for t in ts if t.speaker}, key=str),
    }
