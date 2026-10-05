#!/usr/bin/env python3
"""Diarization repair candidates: the places a model pass would judge.

    python3 scripts/diarization_candidates.py transcripts/lectures/X_raw.json
        -> X_speaker_candidates.json beside the input

Build step 4 of docs/diarization_repair_plan.md (§1.5). The classes, their
parameters, the metrics and the bar were pre-registered, before this file was
written, in docs/diarization_candidates_plan.md; that file is the spec, and
every rule below cites it. In short: the render the lecture profile shows
(``assign`` plus ``smooth`` at their defaults) is regenerated from the raw
JSON, and each class proposes a *window* -- a span of words, not a point --
where a speaker error may lie:

* ``shift``: every rendered boundary, with the N words either side where the
  true change may be;
* split candidates (is this boundary spurious?): ``long_island``,
  ``edge_island``, ``no_punct_lowercase``, ``zero_gap``, ``inside_sentence``;
* ``dropped_raw_change``: words a raw diarizer turn labeled X that the render
  labels otherwise, with no rendered X boundary close enough to contain it;
* merge candidates (did someone else speak inside this run?):
  ``question_answer``, ``backchannel``, ``address_reply``, ``register_change``.

Overlapping windows are merged into *sites*, one model judgment each in step 5.

It proposes and decides nothing. Stage 1 stays untouched (D3): this reads the
raw JSON and writes a separate file. Deterministic, no model, stdlib plus
stage 2 itself. The output holds word indices, times, labels and lexicon
matches, never free transcript text; it lives beside the raw JSON in
transcripts/ (gitignored) all the same.

Exit codes: 0 ok; 2 refused (missing or invalid input).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import schema  # noqa: E402
from render import PROFILES, RenderParams, RWord, select_stream, to_rwords  # noqa: E402
from render import segment, speakers  # noqa: E402
from render.formats import fmt_time  # noqa: E402

FORMAT = 1
PLAN = "docs/diarization_candidates_plan.md"

# In output order at one window: the class list of the plan's §2.
CLASSES = ("shift", "long_island", "edge_island", "no_punct_lowercase", "zero_gap",
           "inside_sentence", "dropped_raw_change", "question_answer", "backchannel",
           "address_reply", "register_change")
FAMILY = {"shift": "shift", "long_island": "split", "edge_island": "split",
          "no_punct_lowercase": "split", "zero_gap": "split", "inside_sentence": "split",
          "dropped_raw_change": "raw", "question_answer": "merge", "backchannel": "merge",
          "address_reply": "merge", "register_change": "merge"}


def default_params() -> Dict[str, Any]:
    """The pre-registered parameters (plan §3). Changing one is a post-hoc change."""
    return {
        "near_words": 3,
        "zero_gap_s": 0.005,
        "island_max_words": 15,
        "backchannel_max_words": 3,
        "address_max_words": 10,
        "register_window_words": 20,
        "register_min_words": 5,
        "register_onset": 2,
        "answer_openers": sorted(
            "yes yeah yep yup no nope nah right exactly correct sure okay ok well um uh hmm mm "
            "mhm mm-hmm uh-huh i i'm i've i'd i'll maybe probably because good great true".split()),
        "backchannel": sorted(
            "yeah yes yep yup okay ok right sure exactly correct true mm mhm mm-hmm uh-huh hmm "
            "uh um cool gotcha".split()),
        "second_person": sorted(
            "you your you're you've you'd you'll yours yourself yourselves y'all".split()),
        "solicit": sorted("anyone anybody someone somebody question questions".split()),
        "solicit_bigrams": [["go", "ahead"]],
        "first_person": sorted("i i'm i've i'd i'll me my mine myself".split()),
    }


def params_sha256(params: Dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_EDGE = re.compile(r"^[^\w'-]+|[^\w'-]+$")


def token(text: str) -> str:
    """Lowercased, curly apostrophes made straight, edge punctuation stripped (plan §1)."""
    return _EDGE.sub("", text.lower().replace("’", "'").replace("‘", "'"))


def first_letter_lower(text: str) -> bool:
    c = next((ch for ch in text if ch.isalpha()), None)
    return c is not None and c.islower()


def ends_question(w: RWord) -> bool:
    return segment.is_terminal(w) and w.text.rstrip(segment._CLOSERS).endswith("?")


# ------------------------------------------------------------ the render ----

@dataclass
class Render:
    """The stream as the lecture profile renders it, plus what the classes read."""
    words: List[RWord]
    labels: List[Optional[str]]          # rendered: assign + smooth
    hits: List[Set[str]]                 # labels of raw turns that hit each word
    hit_turns: List[Set[int]]            # indices into ``turns`` (sorted) that hit each word
    turns: List[Dict[str, Any]]          # raw turns, sorted by (start, end)
    sentences: List[Tuple[int, int]]     # (first, last) stream positions
    stretch: List[int]                   # label stretch id per position
    stretches: List[Tuple[int, int, Optional[str]]]
    boundaries: List[int]                # positions p where the label changes entering p
    mode: str
    warning: Optional[str]
    flipped: int
    pause: float
    max_island: int
    profile: str

    @property
    def n(self) -> int:
        return len(self.words)

    def gap(self, p: int) -> Optional[float]:
        """The gap entering word p (between p-1 and p)."""
        return segment.gap(self.words[p - 1], self.words[p]) if p > 0 else None


def turn_hits(words: List[RWord], turns: List[Dict[str, Any]]) -> List[Set[int]]:
    """Per word, the raw turns that hit it: overlap above zero, or the midpoint inside.

    The labeling sheet's raw-turn accounting uses the same rule
    (``diarization_labels.raw_accounting``)."""
    out: List[Set[int]] = [set() for _ in words]
    if not turns:
        return out
    starts = [t["start"] for t in turns]
    max_len = max(t["end"] - t["start"] for t in turns)
    for k, w in enumerate(words):
        if w.start is None or w.end is None:
            continue
        mid = (w.start + w.end) / 2
        lo = bisect.bisect_left(starts, w.start - max_len)
        hi = bisect.bisect_right(starts, w.end)
        for j in range(lo, hi):
            t = turns[j]
            if min(w.end, t["end"]) - max(w.start, t["start"]) > 0 or t["start"] <= mid <= t["end"]:
                out[k].add(j)
    return out


def render(doc: Dict[str, Any], profile: str = "lecture") -> Render:
    params = RenderParams(profile=PROFILES[profile])
    raw_words, mode, warning = select_stream(doc, params.profile.stream)
    words = to_rwords(raw_words)
    turns = sorted(doc.get("speaker_turns") or [], key=lambda t: (t["start"], t["end"]))
    speakers.assign(words, turns)
    flipped = speakers.smooth(words, params.pause_threshold, params.max_island) if params.smoothing else 0
    labels = [w.speaker for w in words]
    ht = turn_hits(words, turns)
    hits = [{turns[j]["speaker"] for j in js} for js in ht]

    sents: List[Tuple[int, int]] = []
    k = 0
    for s in segment.sentences(words, params.pause_threshold):
        sents.append((k, k + len(s.words) - 1))
        k += len(s.words)

    stretches: List[Tuple[int, int, Optional[str]]] = []
    stretch: List[int] = []
    for p, lab in enumerate(labels):
        if p == 0 or lab != labels[p - 1]:
            stretches.append((p, p, lab))
        else:
            s0, _, l0 = stretches[-1]
            stretches[-1] = (s0, p, l0)
        stretch.append(len(stretches) - 1)
    boundaries = [s for s, _, _ in stretches[1:]]
    return Render(words=words, labels=labels, hits=hits, hit_turns=ht, turns=turns,
                  sentences=sents, stretch=stretch, stretches=stretches, boundaries=boundaries,
                  mode=mode, warning=warning, flipped=flipped, pause=params.pause_threshold,
                  max_island=params.max_island, profile=profile)


# ------------------------------------------------------------- the classes ---

class Out:
    """Collects candidates; windows are clamped to the stream (plan §1)."""

    def __init__(self, r: Render):
        self.r = r
        self.items: List[Dict[str, Any]] = []

    def add(self, cls: str, a: int, b: int, at: int, evidence: Dict[str, Any]) -> None:
        a, b = max(0, a), min(self.r.n - 1, b)
        if a >= b:          # a window must hold at least one transition
            return
        self.items.append({"class": cls, "a": a, "b": b, "at": at, "evidence": evidence})


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 3)


def shift(r: Render, out: Out, P: Dict[str, Any]) -> None:
    """Every boundary p: window [p-N-1, p+N], transitions p-N .. p+N (plan §2, shift)."""
    n_ = P["near_words"]
    for p in r.boundaries:
        out.add("shift", p - n_ - 1, p + n_, p, {
            "from": r.labels[p - 1], "to": r.labels[p], "gap_s": _round(r.gap(p)),
            "prev_terminal": segment.is_terminal(r.words[p - 1])})


def islands(r: Render, out: Out, P: Dict[str, Any]) -> None:
    """Short stretches between two of one other label: long_island inside a
    pause-bounded run, edge_island at its edge (plan §2, split)."""
    st = r.stretches
    for k in range(1, len(st) - 1):
        s, e, lab = st[k]
        if not (st[k - 1][2] == st[k + 1][2] != lab) or e - s + 1 > P["island_max_words"]:
            continue
        gb, ga = r.gap(s), r.gap(e + 1)
        edge = (gb is not None and gb > r.pause) or (ga is not None and ga > r.pause)
        out.add("edge_island" if edge else "long_island", s - 1, e + 1, s, {
            "label": lab, "neighbors": st[k - 1][2], "words": e - s + 1,
            "gap_before_s": _round(gb), "gap_after_s": _round(ga),
            "assigned_words": sum(1 for q in range(s, e + 1) if r.words[q].speaker_raw == lab),
            "max_island": r.max_island})


def boundary_cues(r: Render, out: Out, P: Dict[str, Any]) -> None:
    """no_punct_lowercase, zero_gap, inside_sentence at each boundary (plan §2, split)."""
    for p in r.boundaries:
        prev_term = segment.is_terminal(r.words[p - 1])
        g = r.gap(p)
        ev = {"from": r.labels[p - 1], "to": r.labels[p], "gap_s": _round(g), "prev_terminal": prev_term}
        if not prev_term and first_letter_lower(r.words[p].text):
            out.add("no_punct_lowercase", p - 1, p, p, dict(ev))
        if g is not None and g <= P["zero_gap_s"]:
            out.add("zero_gap", p - 1, p, p, dict(ev))
        if not prev_term and (g is None or g <= r.pause):
            out.add("inside_sentence", p - 1, p, p, dict(ev))


def dropped_raw(r: Render, out: Out, P: Dict[str, Any]) -> None:
    """Raw-turn labels the render dropped (plan §2, dropped raw change)."""
    n_, n = P["near_words"], r.n
    raw = [r.hits[p] | ({r.words[p].speaker_raw} if r.words[p].speaker_raw is not None else set())
           for p in range(n)]
    contested = [raw[p] - {r.labels[p]} for p in range(n)]
    for x in sorted(set().union(*contested) if contested else set(), key=str):
        p = 0
        while p < n:
            if x not in contested[p]:
                p += 1
                continue
            a = p
            while p + 1 < n and x in contested[p + 1]:
                p += 1
            b = p
            p += 1
            borders = (a > 0 and r.labels[a - 1] == x) or (b < n - 1 and r.labels[b + 1] == x)
            length = b - a + 1
            if borders and length <= n_:
                continue                # within N of a rendered X boundary: shift holds it
            assigned = sum(1 for q in range(a, b + 1) if r.words[q].speaker_raw == x)
            kind = "displaced" if borders else "smoothed" if assigned else "outvoted"
            out.add("dropped_raw_change", a - n_ - 1, b + n_ + 1, a, {
                "raw_label": x, "rendered": sorted({r.labels[q] for q in range(a, b + 1)}, key=str),
                "words": length, "borders_raw_label": borders, "assigned_words": assigned,
                "kind": kind})

    # Raw label runs that hit no word at all ("empty" in the sheet's accounting).
    runs: List[Tuple[Optional[str], List[int]]] = []
    for j, t in enumerate(r.turns):
        if runs and runs[-1][0] == t["speaker"]:
            runs[-1][1].append(j)
        else:
            runs.append((t["speaker"], [j]))
    hit_any = set().union(*r.hit_turns) if r.hit_turns else set()
    for x, js in runs:
        if hit_any & set(js):
            continue
        t0 = min(r.turns[j]["start"] for j in js)
        t1 = max(r.turns[j]["end"] for j in js)
        g = next((q for q in range(n) if r.words[q].start is not None and r.words[q].start >= t0), None)
        if g is None or g == 0:
            continue
        if any(r.labels[q] == x for q in range(max(0, g - n_), min(n, g + n_))):
            continue
        out.add("dropped_raw_change", g - n_ - 1, g + n_, g, {
            "raw_label": x, "rendered": sorted({r.labels[g - 1], r.labels[g]}, key=str),
            "words": 0, "borders_raw_label": False, "assigned_words": 0, "kind": "empty",
            "run_turns": len(js), "run_s": [_round(t0), _round(t1)]})


def merge_cues(r: Render, out: Out, P: Dict[str, Any]) -> None:
    """Textual turn-taking cues inside one label stretch (plan §2, merge)."""
    openers, back = set(P["answer_openers"]), set(P["backchannel"])
    second, solicit, fps = set(P["second_person"]), set(P["solicit"]), set(P["first_person"])
    bigrams = {tuple(b) for b in P["solicit_bigrams"]}
    sents, toks = r.sentences, [token(w.text) for w in r.words]
    for j, (f, l) in enumerate(sents):
        same_prev = j > 0 and r.stretch[sents[j - 1][1]] == r.stretch[f]
        same_next = j + 1 < len(sents) and r.stretch[sents[j + 1][0]] == r.stretch[l]
        nw = l - f + 1
        st = toks[f:l + 1]

        if same_next:
            rr = sents[j + 1][0]
            op = toks[rr]
            if ends_question(r.words[l]) and op in openers:
                out.add("question_answer", f - 1, rr, rr, {"opener": op, "question_words": nw})
            if nw <= P["address_max_words"] and segment.is_terminal(r.words[l]) and op in openers:
                cue = next((t for t in st if t in second or t in solicit), None)
                if cue is None:
                    cue = next((" ".join(st[k:k + 2]) for k in range(nw - 1)
                                if tuple(st[k:k + 2]) in bigrams), None)
                if cue is not None:
                    out.add("address_reply", l, rr, rr, {"cue": cue, "opener": op, "address_words": nw})

        if same_prev and same_next and 1 <= nw <= P["backchannel_max_words"] and all(t in back for t in st):
            out.add("backchannel", f - 1, l + 1, f, {"tokens": st, "words": nw})

        if same_prev:
            ss, se, _ = r.stretches[r.stretch[f]]
            w_ = P["register_window_words"]
            before = toks[max(ss, f - w_):f]
            after = toks[f:min(se, f + w_ - 1) + 1]
            if len(before) >= P["register_min_words"] and len(after) >= P["register_min_words"]:
                fb = sum(1 for t in before if t in fps)
                fa = sum(1 for t in after if t in fps)
                if fb == 0 and fa >= P["register_onset"]:
                    out.add("register_change", f - 1, f, f, {
                        "first_person_before": fb, "first_person_after": fa,
                        "before_words": len(before), "after_words": len(after)})


# --------------------------------------------------------------- assemble ---

def sites_of(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Connected components of windows sharing at least one word (plan §4)."""
    out: List[Dict[str, Any]] = []
    for c in sorted(cands, key=lambda c: (c["a"], c["b"])):
        if out and c["a"] <= out[-1]["b"]:
            out[-1]["b"] = max(out[-1]["b"], c["b"])
            out[-1]["members"].append(c)
        else:
            out.append({"a": c["a"], "b": c["b"], "members": [c]})
    return out


def generate(doc: Dict[str, Any], profile: str = "lecture",
             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Candidates and sites for a record. Deterministic: no clock, no randomness."""
    P = params or default_params()
    r = render(doc, profile)
    out = Out(r)
    shift(r, out, P)
    islands(r, out, P)
    boundary_cues(r, out, P)
    dropped_raw(r, out, P)
    merge_cues(r, out, P)

    src = doc.get("source") or {}
    offset = float((src.get("excerpt") or {}).get("offset_s") or 0.0)
    dur = src.get("duration_s")
    hours = dur / 3600 if dur else None
    w = r.words

    items = sorted(out.items, key=lambda c: (c["a"], c["b"], CLASSES.index(c["class"]), c["at"]))
    cands = []
    for k, c in enumerate(items, 1):
        c["id"] = f"C-{k:04d}"
        cands.append({
            "id": c["id"], "class": c["class"], "family": FAMILY[c["class"]],
            "first_word_i": w[c["a"]].i, "last_word_i": w[c["b"]].i, "at_word_i": w[c["at"]].i,
            "words": c["b"] - c["a"] + 1, "start_s": w[c["a"]].start, "end_s": w[c["b"]].end,
            "time": fmt_time(w[c["a"]].start or 0.0, offset), "evidence": c["evidence"]})
    sites = []
    for k, s in enumerate(sites_of(items), 1):
        sites.append({
            "id": f"J-{k:03d}", "first_word_i": w[s["a"]].i, "last_word_i": w[s["b"]].i,
            "words": s["b"] - s["a"] + 1, "time": fmt_time(w[s["a"]].start or 0.0, offset),
            "candidates": [c["id"] for c in sorted(s["members"], key=lambda c: c["id"])],
            "classes": [x for x in CLASSES if any(c["class"] == x for c in s["members"])]})

    covered = sum(s["words"] for s in sites)
    per_hour = (lambda x: round(x / hours, 1) if hours else None)   # noqa: E731
    by_class = {x: sum(1 for c in cands if c["class"] == x) for x in CLASSES}
    return {
        "format": FORMAT, "generator": "scripts/diarization_candidates.py", "plan": PLAN,
        "audio_duration_s": dur, "stream": r.mode, "stream_warning": r.warning, "offset_s": offset,
        "render": {"profile": profile, "pause_threshold_s": r.pause, "smoothing": True,
                   "max_island": r.max_island, "speaker_map": None, "words_reassigned": r.flipped},
        "params": P, "params_sha256": params_sha256(P),
        "word_index_refers_to": f"the {r.mode!r} stream's words[].i",
        "window_rule": "a window [first_word_i, last_word_i] holds the transitions between "
                       "consecutive words that both lie inside it",
        "counts": {
            "stream_words": r.n, "boundaries": len(r.boundaries),
            "candidates": len(cands), "candidates_per_hour": per_hour(len(cands)),
            "by_class": by_class,
            "sites": len(sites), "sites_per_hour": per_hour(len(sites)),
            "words_covered": covered, "words_covered_share": round(covered / r.n, 4) if r.n else None,
        },
        "boundaries": [w[p].i for p in r.boundaries],
        "candidates": cands,
        "sites": sites,
    }


def summary(res: Dict[str, Any]) -> List[str]:
    c = res["counts"]
    out = [f"stream: {res['stream']}, {c['stream_words']} words, {c['boundaries']} rendered boundaries",
           f"candidates: {c['candidates']} ({c['candidates_per_hour']} per audio hour)"]
    for x in CLASSES:
        out.append(f"  {x:<20} {c['by_class'][x]}")
    out.append(f"sites: {c['sites']} ({c['sites_per_hour']} per audio hour); words covered "
               f"{c['words_covered']} ({100 * (c['words_covered_share'] or 0):.1f}%)")
    return out


# ------------------------------------------------------------------ CLI -----

def stem_of(path: Path) -> str:
    return path.name[:-len("_raw.json")] if path.name.endswith("_raw.json") else path.stem


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="a *_raw.json")
    p.add_argument("-o", "--output", default=None,
                   help="default: <stem>_speaker_candidates.json beside the input")
    p.add_argument("--profile", default="lecture", choices=sorted(PROFILES),
                   help="the render profile whose stream is judged (default lecture)")
    a = p.parse_args(argv)
    inp = Path(a.input)
    try:
        doc = schema.read(inp)
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    out = Path(a.output) if a.output else inp.parent / f"{stem_of(inp)}_speaker_candidates.json"
    res = generate(doc, a.profile)
    label = os.path.relpath(inp.resolve(), out.parent.resolve())
    res = {"source": label, "source_sha256": sha256_file(inp), **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    for ln in summary(res):
        print(ln)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
