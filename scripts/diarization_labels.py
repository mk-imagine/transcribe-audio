#!/usr/bin/env python3
"""Diarization labeling sheet: generate one, and check a filled one.

    python scripts/diarization_labels.py transcripts/lectures/X_raw.json
        -> X_labels.md and X_label-times.txt beside the input
    python scripts/diarization_labels.py --check transcripts/lectures/X_labels.md
        -> progress, parse errors with line numbers, and X_labels.json

Build step 1 of docs/diarization_repair_plan.md (§1.5). The operator reads the
transcript once, as stage 2 renders it today -- the lecture profile's stream,
``assign`` plus ``smooth`` at their defaults, no speaker map -- and labels two
things in place:

* every place the rendered speaker label changes (a *boundary block*): ``real``,
  ``spurious`` or ``unsure``, optionally with ``who:`` speaks next;
* every place a different person speaks with no boundary (a *missed change*):
  a ``>> change: WHO`` marker line where they start, ``>> back: WHO`` where the
  earlier speaker resumes.

Those labels are the gold standard the plan's §1.4 scores against, so the
checker is strict: it regenerates the sheet from the raw JSON named in its
header (refusing if the file's sha256 differs) and walks the filled sheet
against it. Any line that is not a generated line, a filled field or a
well-formed marker is reported with its line number, never skipped.

Pure Python, no model, no dependencies beyond stage 2 itself. The sheet and
its labels describe a class session with student voices: they live in
transcripts/ (gitignored) and are never committed.

Exit codes: 0 ok; 1 the filled sheet has errors (no JSON is written);
2 the input was refused (missing, wrong hash, or an existing sheet without --force).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import schema  # noqa: E402
from render import PROFILES, RenderParams, RWord, select_stream, to_rwords  # noqa: E402
from render import segment, speakers  # noqa: E402
from render.formats import fmt_time  # noqa: E402

SHEET_FORMAT = 1
TRANSCRIPT_RULE = "==== TRANSCRIPT ===="
WHO_RE = re.compile(r"^(L|S[2-9]?|\?)$", re.IGNORECASE)
VERDICTS = ("real", "spurious", "unsure")
# `>> change: S` or `>> back: L "and so the"`. Straight or curly double quotes.
MARKER_RE = re.compile(
    r'^>>\s*(change|back)\s*:\s*(\S+)(?:\s+["“”]([^"“”]*)["“”])?\s*$', re.IGNORECASE)
MARKERISH_RE = re.compile(r"^(>|(change|back)\s*:)", re.IGNORECASE)
TEXT_ID_RE = re.compile(r"^(T\d{4,})\b")


# ------------------------------------------------------------- the model ----

@dataclass
class TextLine:
    lid: str                    # "T0001"
    words: List[RWord]
    speaker: Optional[str]

    @property
    def start(self) -> float:
        return next((w.start for w in self.words if w.start is not None), 0.0)


@dataclass
class Boundary:
    bid: str                    # "B-001"
    start: float                # the first word of the new label, record time
    frm: Optional[str]
    to: Optional[str]
    line_before: str
    line_after: str
    word_i: int


@dataclass
class Sheet:
    lines: List[Tuple[str, Any]]        # (kind, payload); kind in header/start/bhead/bfield/wfield/text/blank
    boundaries: List[Boundary]
    text: Dict[str, TextLine]
    offset: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return "\n".join(_line_text(k, p, self.offset) for k, p in self.lines) + "\n"


def _line_text(kind: str, payload: Any, offset: float) -> str:
    if kind in ("header", "start", "blank"):
        return payload
    if kind == "bhead":
        b = payload
        return f"#### {b.bid}  {fmt_time(b.start, offset)}  {b.frm} -> {b.to}"
    if kind == "bfield":
        return "boundary:"
    if kind == "wfield":
        return "who:"
    if kind == "text":
        t = payload
        return f"{t.lid} [{fmt_time(t.start, offset)}] " + " ".join(w.text for w in t.words)
    raise ValueError(kind)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_changes(seq: List[Optional[str]]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def raw_accounting(words: List[RWord], turns: List[Dict[str, Any]]) -> Dict[str, int]:
    """Why the raw turns' label changes outnumber the rendered ones.

    The raw turns are grouped into label runs (consecutive turns with one
    label); a run is "empty" if no word overlaps it at all and "outvoted" if
    every word it overlaps was assigned to the other label by max-overlap.
    Neither kind of run can surface as a rendered speaker change. A word no
    turn covers goes to the nearest turn (assign's fallback), so the run holding
    that turn counts as winning it.
    """
    ts = sorted(turns, key=lambda t: (t["start"], t["end"]))
    if not ts:
        return {"raw_changes": 0, "runs": 0, "empty": 0, "outvoted": 0, "kept_changes": 0}
    runs: List[Dict[str, Any]] = []
    for t in ts:
        if runs and runs[-1]["speaker"] == t["speaker"]:
            runs[-1]["turns"].append(t)
        else:
            runs.append({"speaker": t["speaker"], "turns": [t]})
    timed = [w for w in words if w.start is not None and w.end is not None]

    def hits(w: RWord, t: Dict[str, Any]) -> bool:
        return (min(w.end, t["end"]) - max(w.start, t["start"]) > 0
                or t["start"] <= (w.start + w.end) / 2 <= t["end"])

    # A word no turn overlaps or contains goes to the nearest turn (assign's
    # fallback); the run holding that turn wins it.
    run_of = {id(t): k for k, r in enumerate(runs) for t in r["turns"]}
    starts = [t["start"] for t in ts]
    max_len = max((t["end"] - t["start"] for t in ts), default=0.0)
    gap_wins = set()
    for w in timed:
        lo, hi = bisect.bisect_left(starts, w.start - max_len), bisect.bisect_right(starts, w.end)
        if not any(hits(w, t) for t in ts[lo:hi]):
            mid = (w.start + w.end) / 2
            near = min(ts, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
            gap_wins.add(run_of[id(near)])
    empty = outvoted = 0
    kept: List[str] = []
    for k, r in enumerate(runs):
        lo = min(t["start"] for t in r["turns"]); hi = max(t["end"] for t in r["turns"])
        touching = [w for w in timed if w.end >= lo and w.start <= hi
                    and any(hits(w, t) for t in r["turns"])]
        if k in gap_wins or any(w.speaker_raw == r["speaker"] for w in touching):
            kept.append(r["speaker"])
        elif not touching:
            empty += 1
        else:
            outvoted += 1
    return {"raw_changes": len(runs) - 1, "runs": len(runs), "empty": empty,
            "outvoted": outvoted, "kept_changes": label_changes(kept)}


def instructions() -> List[str]:
    return [
        "Read the transcript top to bottom. Each `####` boundary block is a place where the",
        "rendered speaker label changes: after `boundary:` type `real` (a different person starts",
        "speaking there), `spurious` (the same person continues) or `unsure`; after `who:` you may",
        "record who speaks next. Where a different person starts speaking with NO boundary block,",
        "insert a line `>> change: WHO` directly above the line where they start, and",
        "`>> back: WHO` above the line where the earlier speaker resumes, if they do. WHO is `L`",
        "(the lecturer), `S` (a student), `S2`-`S9` (further distinguishable students) or `?`.",
        "If the change falls mid-line, add the new speaker's first words in double quotes:",
        "`>> change: S \"is that the\"`. A change stays open until its `>> back:` or the next",
        "boundary labeled `real`. Open the audio only when the text is ambiguous; mark `unsure`",
        "rather than guess. Do not edit transcript lines or this header. Check as you go with",
        "`python scripts/diarization_labels.py --check <this file>`.",
    ]


def build(doc: Dict[str, Any], source_label: str, source_sha: str) -> Sheet:
    """The sheet for a record, deterministic: no clock, no randomness."""
    params = RenderParams(profile=PROFILES["lecture"])       # defaults: pause 0.5, max_island 2
    raw_words, mode_used, warning = select_stream(doc, params.profile.stream)
    words = to_rwords(raw_words)
    turns_raw = doc.get("speaker_turns") or []
    speakers.assign(words, turns_raw)
    assigned = label_changes([w.speaker for w in words])
    acct = raw_accounting(words, turns_raw)
    flipped = speakers.smooth(words, params.pause_threshold, params.max_island)
    sents = segment.sentences(words, params.pause_threshold)
    turns = segment.turns(sents)

    src = doc.get("source") or {}
    excerpt = src.get("excerpt") or {}
    offset = float(excerpt.get("offset_s") or 0.0)

    text: Dict[str, TextLine] = {}
    body: List[Tuple[str, Any]] = []
    boundaries: List[Boundary] = []
    n = 0
    prev_lid: Optional[str] = None
    for k, t in enumerate(turns):
        if k == 0:
            body.append(("start", f"#### START  {t.speaker}"))
        else:
            first = t.sentences[0].words[0]
            b = Boundary(bid=f"B-{len(boundaries) + 1:03d}", start=first.start or 0.0,
                         frm=turns[k - 1].speaker, to=t.speaker, line_before=prev_lid or "",
                         line_after=f"T{n + 1:04d}", word_i=first.i)
            boundaries.append(b)
            body += [("blank", ""), ("bhead", b), ("bfield", None), ("wfield", None)]
        for s in t.sentences:
            n += 1
            tl = TextLine(lid=f"T{n:04d}", words=s.words, speaker=t.speaker)
            text[tl.lid] = tl
            body.append(("text", tl))
            prev_lid = tl.lid

    stem = Path(source_label).name
    stem = stem[:-len("_raw.json")] if stem.endswith("_raw.json") else Path(stem).stem
    head = [
        f"# Diarization labeling sheet: {stem}",
        "",
        *instructions(),
        "",
        f"- source: {source_label}",
        f"- source sha256: {source_sha}",
        f"- audio: {src.get('audio_path')}  sha256 {src.get('audio_sha256')}",
        f"- stream: {mode_used} (the lecture profile's)" + (f"; WARNING: {warning}" if warning else ""),
        f"- render: assign + smooth, pause_threshold {params.pause_threshold} s, "
        f"max_island {params.max_island}, no speaker map; smoothing reassigned {flipped} word(s)",
        f"- times: recording time hh:mm:ss.s"
        + (f" (excerpt offset +{offset:.0f} s applied)" if offset else ""),
        f"- boundary blocks: {len(boundaries)}; text lines: {n}",
        f"- vs. the raw turns: {acct['raw_changes']} label changes there, {len(boundaries)} here. "
        f"Of the raw turns' {acct['runs']} label runs (consecutive turns with one label), "
        f"{acct['empty']} hold no word at all and {acct['outvoted']} lose every word they overlap "
        f"to a longer-overlapping turn of the other label, so max-overlap assignment gives each "
        f"no word; {acct['kept_changes']} changes remain (word-level: {assigned}). Smoothing "
        f"then removes {assigned - len(boundaries)}. A dropped run is not checked here as a "
        f"boundary; if someone else really spoke there, it is a missed change: mark it.",
        f"- sheet format: {SHEET_FORMAT}",
        "",
        TRANSCRIPT_RULE,
        "",
    ]
    lines: List[Tuple[str, Any]] = [("header", h) for h in head] + body
    meta = {
        "stem": stem, "source": source_label, "source_sha256": source_sha,
        "audio_path": src.get("audio_path"), "audio_sha256": src.get("audio_sha256"),
        "stream": mode_used, "offset_s": offset,
        "render": {"profile": "lecture", "pause_threshold_s": params.pause_threshold,
                   "smoothing": True, "max_island": params.max_island, "speaker_map": None,
                   "words_reassigned": flipped},
        "counts": {"boundaries": len(boundaries), "text_lines": n,
                   "assigned_changes": assigned, **acct},
    }
    return Sheet(lines=lines, boundaries=boundaries, text=text, offset=offset, meta=meta)


def times_file(sheet: Sheet) -> str:
    """hh:mm:ss per boundary, rounded down so playback starts just before it."""
    out = [f"# {sheet.meta['stem']}: diarization boundaries, for checking the audio",
           "# hh:mm:ss (rounded down)  id  from -> to"]
    for b in sheet.boundaries:
        t = int(max(0.0, b.start + sheet.offset))
        out.append(f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}  {b.bid}  {b.frm} -> {b.to}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- check -----

def _norm(tok: str) -> str:
    return re.sub(r"[^\w']", "", tok.lower()).strip("'")


@dataclass
class Marker:
    kind: str           # change | back
    who: str
    quote: Optional[str]
    sheet_line: int
    lid: str = ""
    pos: int = 0        # word offset within the text line
    word: Optional[RWord] = None


def locate(quote: Optional[str], tl: TextLine) -> Tuple[Optional[int], Optional[str]]:
    """Word offset of a quoted run of whole tokens in a text line, or an error."""
    if quote is None:
        return 0, None
    q = [_norm(x) for x in quote.split()]
    q = [x for x in q if x]
    if not q:
        return None, "the quote is empty"
    toks = [_norm(w.text) for w in tl.words]
    hits = [k for k in range(len(toks) - len(q) + 1) if toks[k:k + len(q)] == q]
    if not hits:
        return None, f'"{quote}" is not a run of whole words in {tl.lid}'
    if len(hits) > 1:
        return None, f'"{quote}" occurs {len(hits)} times in {tl.lid}; quote more words'
    return hits[0], None


def check(sheet_path: Path, source_override: Optional[Path] = None) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Parse a filled sheet. Returns (labels, errors, notes); raises SystemExit(2) on refusal."""
    filled = sheet_path.read_text().splitlines()
    label = next((ln[len("- source: "):] for ln in filled if ln.startswith("- source: ")), None)
    sha = next((ln[len("- source sha256: "):].strip() for ln in filled
                if ln.startswith("- source sha256: ")), None)
    if label is None or sha is None:
        raise SystemExit(f"{sheet_path}: no '- source:' / '- source sha256:' header lines; "
                         "is this a labeling sheet?")
    src = source_override or (sheet_path.parent / label if not os.path.isabs(label) else Path(label))
    if not src.is_file():
        raise SystemExit(f"{sheet_path}: source {src} not found; --check regenerates the sheet "
                         "from it (pass --source to point elsewhere)")
    if sha256_file(src) != sha:
        raise SystemExit(f"{src}: sha256 differs from the sheet's header; this sheet was made "
                         "from a different file")
    sheet = build(schema.read(src), label, sha)

    errors: List[str] = []
    notes: List[str] = []
    exp = [(k, p, _line_text(k, p, sheet.offset)) for k, p in sheet.lines]
    exp = [e for e in exp if e[2].strip()]          # blank lines are free to add or drop
    rule_at = next(i for i, (k, p, s) in enumerate(exp) if s == TRANSCRIPT_RULE)
    verdict: Dict[str, Optional[str]] = {b.bid: None for b in sheet.boundaries}
    who: Dict[str, Optional[str]] = {b.bid: None for b in sheet.boundaries}
    markers: List[Marker] = []
    pending: List[Marker] = []
    events: List[Tuple[str, Any]] = []       # ("marker", Marker) | ("boundary", bid), in order
    cur_block: Optional[Boundary] = None
    k = 0

    def matches(e: Tuple[str, Any, str], s: str) -> bool:
        kind = e[0]
        if kind == "bfield":
            return re.match(r"^boundary\s*:", s, re.IGNORECASE) is not None
        if kind == "wfield":
            return re.match(r"^who\s*:", s, re.IGNORECASE) is not None
        return s == e[2]

    def accept(n: int, e: Tuple[str, Any, str], s: str) -> None:
        nonlocal cur_block
        kind, p, _ = e
        if kind == "bhead":
            for m in pending:
                errors.append(f"line {m.sheet_line}: marker directly above boundary {p.bid}; a change "
                              f"there is the boundary's own label (`boundary: real`, `who: {m.who}`)")
            pending.clear()
            cur_block = p
            events.append(("boundary", p.bid))
        elif kind == "bfield":
            v = s.split(":", 1)[1].strip()
            if v and v.lower() not in VERDICTS:
                errors.append(f"line {n}: {cur_block.bid}: unknown verdict {v!r}; use real, spurious or unsure")
            elif v:
                verdict[cur_block.bid] = v.lower()
        elif kind == "wfield":
            v = s.split(":", 1)[1].strip()
            if v and not WHO_RE.match(v):
                errors.append(f"line {n}: {cur_block.bid}: unknown who {v!r}; use L, S, S2-S9 or ?")
            elif v:
                who[cur_block.bid] = v.upper()
                if verdict[cur_block.bid] == "spurious":
                    errors.append(f"line {n}: {cur_block.bid}: `who:` on a spurious boundary; nobody "
                                  "new speaks there")
            cur_block = None
        elif kind == "text":
            cur_block = None
            seen = set()
            for m in pending:
                pos, err = locate(m.quote, p)
                if err:
                    errors.append(f"line {m.sheet_line}: {err}")
                    continue
                if pos in seen:
                    errors.append(f"line {m.sheet_line}: two markers at the same word of {p.lid}")
                    continue
                if seen and pos < max(seen):
                    errors.append(f"line {m.sheet_line}: markers above {p.lid} are out of order")
                    continue
                seen.add(pos)
                m.lid, m.pos, m.word = p.lid, pos, p.words[pos]
                markers.append(m)
                events.append(("marker", m))
            pending.clear()

    for n, line in enumerate(filled, 1):
        s = line.strip()
        if not s:
            continue
        mm = MARKER_RE.match(s)
        if mm:
            kind, w, quote = mm.group(1).lower(), mm.group(2), mm.group(3)
            if not WHO_RE.match(w):
                errors.append(f"line {n}: unknown who {w!r} in marker; use L, S, S2-S9 or ?")
            elif k <= rule_at:
                errors.append(f"line {n}: marker above the transcript")
            elif cur_block is not None:
                errors.append(f"line {n}: marker inside boundary block {cur_block.bid}; put it above "
                              "the transcript line where the speaker changes")
            else:
                pending.append(Marker(kind=kind, who=w.upper(), quote=quote, sheet_line=n))
            continue
        if MARKERISH_RE.match(s):
            errors.append(f"line {n}: malformed marker {s!r}; the form is `>> change: WHO` or "
                          "`>> back: WHO`, optionally followed by the first words in double quotes")
            continue
        if k < len(exp) and matches(exp[k], s):
            accept(n, exp[k], s); k += 1
            continue
        tid = TEXT_ID_RE.match(s)
        if k < len(exp) and tid and exp[k][0] == "text" and tid.group(1) == exp[k][1].lid:
            errors.append(f"line {n}: transcript line {tid.group(1)} was edited; restore it")
            accept(n, exp[k], exp[k][2]); k += 1
            continue
        # A deleted line? Look ahead for this one among the expected lines.
        j = next((j for j in range(k + 1, min(len(exp), k + 200)) if matches(exp[j], s)), None)
        if j is not None:
            missing = [exp[i][2] for i in range(k, j)]
            errors.append(f"line {n}: {len(missing)} expected line(s) missing above it, "
                          f"starting with {missing[0]!r}")
            k = j
            accept(n, exp[k], s); k += 1
            continue
        errors.append(f"line {n}: unrecognized line {s[:60]!r}; not a sheet line, a field or a marker")
    if k < len(exp):
        errors.append(f"end of sheet: {len(exp) - k} expected line(s) missing, starting with {exp[k][2]!r}")
    for m in pending:
        errors.append(f"line {m.sheet_line}: marker with no transcript line below it")

    # Balance: a change stays open until its back, or a boundary labeled real.
    open_m: Optional[Marker] = None
    crossed_unlabeled = False
    for kind, ev in events:
        if kind == "boundary":
            v = verdict[ev]
            if v == "real":
                open_m, crossed_unlabeled = None, False
            elif v is None and open_m is not None:
                crossed_unlabeled = True
            continue
        m: Marker = ev
        if m.kind == "change":
            if open_m is not None and open_m.who == m.who and m.who != "?":
                errors.append(f"line {m.sheet_line}: change to {m.who} while the change to {m.who} "
                              f"at line {open_m.sheet_line} is still open")
            open_m, crossed_unlabeled = m, False
        else:
            if open_m is None:
                errors.append(f"line {m.sheet_line}: back with no open change")
            elif open_m.who == m.who and m.who != "?":
                errors.append(f"line {m.sheet_line}: back to {m.who}, who is the one already speaking "
                              f"since line {open_m.sheet_line}")
            open_m, crossed_unlabeled = None, False
    if open_m is not None:
        if crossed_unlabeled:
            notes.append(f"change at line {open_m.sheet_line} is still open past an unlabeled boundary "
                         "(fine while you are still labeling)")
        else:
            errors.append(f"line {open_m.sheet_line}: change never returns; add `>> back: WHO` where "
                          "the earlier speaker resumes, or label the boundary where it ends `real`")

    off = sheet.offset
    labels = {
        "sheet": str(sheet_path), "sheet_format": SHEET_FORMAT,
        **{k_: sheet.meta[k_] for k_ in ("source", "source_sha256", "audio_path", "audio_sha256",
                                          "stream", "offset_s", "render", "counts")},
        "word_index_refers_to": f"the {sheet.meta['stream']!r} stream's words[].i",
        "complete": all(verdict.values()) and not notes,
        "boundaries": [{"id": b.bid, "start_s": b.start, "time": fmt_time(b.start, off),
                        "from": b.frm, "to": b.to, "word_i": b.word_i,
                        "line_before": b.line_before, "line_after": b.line_after,
                        "verdict": verdict[b.bid], "who": who[b.bid]} for b in sheet.boundaries],
        "markers": [{"kind": m.kind, "who": m.who, "line_id": m.lid, "sheet_line": m.sheet_line,
                     "quote": m.quote, "word_i": m.word.i, "start_s": m.word.start,
                     "time": fmt_time(m.word.start or 0.0, off)} for m in markers],
    }
    return labels, errors, notes


def progress(labels: Dict[str, Any]) -> List[str]:
    bs = labels["boundaries"]
    done = [b for b in bs if b["verdict"]]
    by = {v: sum(1 for b in done if b["verdict"] == v) for v in VERDICTS}
    ms = labels["markers"]
    last = max((b["id"] for b in done), default=None)
    return [
        f"boundaries labeled: {len(done)} / {len(bs)}  "
        + "  ".join(f"{v} {n}" for v, n in by.items())
        + (f"  (last labeled: {last})" if last else ""),
        f"missed-change markers: {sum(m['kind'] == 'change' for m in ms)} change, "
        f"{sum(m['kind'] == 'back' for m in ms)} back",
    ]


# ------------------------------------------------------------------ CLI -----

def stem_of(path: Path) -> str:
    return path.name[:-len("_raw.json")] if path.name.endswith("_raw.json") else path.stem


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="a *_raw.json to make a sheet from, or with --check a filled sheet")
    p.add_argument("--check", action="store_true", help="parse a filled sheet; write <sheet>.json if clean")
    p.add_argument("--source", default=None, help="with --check: the raw JSON, if not where the header says")
    p.add_argument("-o", "--output_dir", default=None, help="default: beside the input")
    p.add_argument("--force", action="store_true", help="overwrite an existing sheet (and its labels)")
    a = p.parse_args(argv)
    inp = Path(a.input)

    if a.check:
        try:
            labels, errors, notes = check(inp, Path(a.source) if a.source else None)
        except SystemExit as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        for ln in progress(labels):
            print(ln)
        for ln in notes:
            print(f"note: {ln}")
        out = inp.with_suffix(".json")
        if errors:
            print(f"\n{len(errors)} error(s); {out.name} not written:")
            for e in errors:
                print(f"  {e}")
            return 1
        out.write_text(json.dumps(labels, indent=1) + "\n")
        print(f"no errors; wrote {out}")
        return 0

    try:
        doc = schema.read(inp)
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(a.output_dir) if a.output_dir else inp.parent
    stem = stem_of(inp)
    sheet_path = out_dir / f"{stem}_labels.md"
    times_path = out_dir / f"{stem}_label-times.txt"
    if sheet_path.exists() and not a.force:
        print(f"refused: {sheet_path} exists and may hold labels; --force to overwrite", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    label = os.path.relpath(inp.resolve(), out_dir.resolve())
    sheet = build(doc, label, sha256_file(inp))
    sheet_path.write_text(sheet.render())
    times_path.write_text(times_file(sheet))
    c = sheet.meta["counts"]
    print(f"wrote {sheet_path}: {c['boundaries']} boundary blocks, {c['text_lines']} lines "
          f"(raw turns: {c['raw_changes']} label changes)")
    print(f"wrote {times_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
