#!/usr/bin/env python3
"""Score the rendered speaker labels against a checked labeling sheet.

    python3 scripts/score_diarization.py transcripts/lectures/X_labels.json \\
        --speaker-map SPEAKER_01=L,SPEAKER_00=S
        -> the baseline numbers; --json FILE also writes them

Build step 3 of docs/diarization_repair_plan.md (§1.5): the baseline, §1.4
step 1. The input is the labels JSON that ``diarization_labels.py --check``
writes from a filled sheet. The rendered labels are regenerated from the raw
JSON exactly as the sheet showed them (``assign`` plus ``smooth``, lecture
profile defaults), refusing a source whose sha256 differs or a render whose
boundaries are not the ones that were labeled.

The sheet records verdicts and markers, not a speaker per word, and it has no
place to say which diarizer label is whom, so ``--speaker-map`` says it, or,
as ``--speaker-map auto``, derives it from the labels by a fixed rule
(``derive_speaker_map``). The per-word truth is read off the sheet like this:

* the first word's speaker is its rendered label's identity under the map;
* a boundary marked ``real`` starts its ``who``, or, with no ``who``, the new
  label's identity under the map; ``spurious`` changes nothing; ``unsure`` (or
  unlabeled) makes the speaker unknown;
* a marker, ``change`` or ``back`` alike, starts its WHO from its word (after
  a boundary at the same word);
* an unknown speaker stays unknown across ``spurious`` and ``unsure``
  boundaries until a ``real`` boundary or a marker names one. Unknown words,
  and words whose speaker is ``?``, are excluded from every per-word number.

That is the grammar's own rule -- a change stays open until its ``back`` or
the next ``real`` boundary -- read as a speaker per word.

Error words are classified two ways. By direction: a student's words printed
under the instructor's label (the high-harm direction of §1.1), the
instructor's under a student's, or one student's under another's. By cause,
from what changes where the error run starts: the label changed and the
speaker did not (a *spurious split*), the speaker changed and the label did not
(a *missed change*), or both changed but the new label names the wrong person
(*wrong label*).

A *missed change* is a word where the true speaker changes and the rendered
label does not. One that leaves no word misattributed (a return to the speaker
the label already names) is counted, and reported apart.

Boundary precision is also given *within N words* (N = 1 up to ``--near``): a
spurious boundary whose true change lies within N words counts as a *near
miss*, a boundary placed a word or two off rather than one with no change
behind it. The pairing is one to one: a near miss is matched to a missed
change (a change already at a boundary has been found), and no two boundaries
share one change. Precision within N is (real + near misses) / (real +
spurious).

Pure Python, no model, no dependencies beyond stage 2 itself. The output holds
counts, word indices, line ids and times, never transcript text.

Exit codes: 0 ok; 2 refused (missing or changed source, a render that no
longer matches the labels, an incomplete sheet without --allow-incomplete, or
a speaker map that does not cover the rendered labels).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diarization_labels as dl  # noqa: E402
from pipeline import schema  # noqa: E402

INSTRUCTOR = "L"
UNKNOWN_WHO = "?"
STUDENT_RE = re.compile(r"^S[2-9]?$")
MAP_WHO_RE = re.compile(r"^(L|S[2-9]?)$")
NEAR_WORDS = 3          # a boundary this many words from a true change is "displaced"
DIRECTIONS = ("student_as_instructor", "instructor_as_student", "student_as_other_student", "other")
CAUSES = ("spurious_split", "missed_change", "wrong_label", "after_unknown")


def is_student(who: Optional[str]) -> bool:
    return who is not None and STUDENT_RE.match(who) is not None


def parse_speaker_map(spec: str) -> Dict[str, str]:
    """``SPEAKER_01=L,SPEAKER_00=S`` -> {label: who}. WHO is L, S or S2-S9."""
    out: Dict[str, str] = {}
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{part!r}: expected LABEL=WHO")
        label, who = (x.strip() for x in part.split("=", 1))
        who = who.upper()
        if not label or not MAP_WHO_RE.match(who):
            raise ValueError(f"{part!r}: WHO must be L, S or S2-S9")
        if out.get(label, who) != who:
            raise ValueError(f"{label} is mapped twice")
        out[label] = who
    if not out:
        raise ValueError("the speaker map is empty")
    return out


# ------------------------------------------------------------- the stream ---

@dataclass
class Stream:
    """The words as the labeled sheet showed them, in order."""
    index: List[int]                 # words[].i, the labels' word_i
    labels: List[Optional[str]]      # rendered label after assign + smooth
    line_of: List[str]               # sheet line id
    start: List[Optional[float]]
    offset: float
    boundary_at: Dict[int, str]      # position -> boundary id


def rendered_stream(labels: Dict[str, Any], labels_path: Path,
                    source_override: Optional[Path] = None) -> Stream:
    """Regenerate the sheet the labels came from; refuse if it is not the same one."""
    label = labels["source"]
    src = source_override or (Path(label) if os.path.isabs(label) else labels_path.parent / label)
    if not src.is_file():
        raise SystemExit(f"source {src} not found (pass --source to point elsewhere)")
    if dl.sha256_file(src) != labels["source_sha256"]:
        raise SystemExit(f"{src}: sha256 differs from the labels' source_sha256")
    sheet = dl.build(schema.read(src), label, labels["source_sha256"])
    got = [(b.bid, b.word_i, b.frm, b.to) for b in sheet.boundaries]
    want = [(b["id"], b["word_i"], b["from"], b["to"]) for b in labels["boundaries"]]
    if got != want:
        raise SystemExit("the render no longer matches the labeled sheet (boundaries differ); "
                         "the labels describe a different rendering")
    if labels.get("render") != sheet.meta["render"]:
        raise SystemExit("the render parameters differ from the ones the sheet was labeled with")
    words = [(t.lid, w) for t in sheet.text.values() for w in t.words]
    pos = {w.i: k for k, (_, w) in enumerate(words)}
    return Stream(index=[w.i for _, w in words], labels=[w.speaker for _, w in words],
                  line_of=[lid for lid, _ in words], start=[w.start for _, w in words],
                  offset=sheet.offset, boundary_at={pos[b.word_i]: b.bid for b in sheet.boundaries})


# -------------------------------------------------------------- the truth ---

def derive_truth(labels: Dict[str, Any], stream: Stream,
                 speaker_map: Dict[str, str]) -> List[Optional[str]]:
    """The true speaker per word, or None where the sheet leaves it unknown."""
    pos = {i: k for k, i in enumerate(stream.index)}
    at_boundary = {pos[b["word_i"]]: b for b in labels["boundaries"]}
    at_marker: Dict[int, List[Dict[str, Any]]] = {}
    for m in labels["markers"]:
        at_marker.setdefault(pos[m["word_i"]], []).append(m)
    cur = speaker_map.get(stream.labels[0]) if stream.labels else None
    out: List[Optional[str]] = []
    for p in range(len(stream.index)):
        b = at_boundary.get(p)
        if b is not None:
            if b["verdict"] == "real":
                cur = b["who"] or speaker_map.get(b["to"])
            elif b["verdict"] != "spurious":        # unsure, or not labeled yet
                cur = None
        for m in at_marker.get(p, ()):
            cur = m["who"]
        out.append(cur)
    return out


STUDENTS = ("S",) + tuple(f"S{k}" for k in range(2, 10))
MAP_RULE = "auto: docs/diarization_candidates_v2_plan.md §6.6 (as amended, A2)"


def derive_speaker_map(labels: Dict[str, Any], stream: Stream) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """The speaker map the labels imply: ``--speaker-map auto``.

    The rule is pre-registered in docs/diarization_candidates_v2_plan.md §6.6
    (as amended, A2). It reads only the words whose speaker the labels *state*
    -- the truth derived with an empty map, so a marker or a ``real``
    boundary's ``who:`` sets it, and nothing the map would supply does:

    * nL(X), nS(X): the stated instructor and student words under label X;
      NL, NS: their totals;
    * X is L iff nL(X)/NL >= nS(X)/NS, each share 0 when its total is 0, so a
      label with no stated words, or a lecture with no stated student, is L;
      otherwise X is the student with the most stated words under X (ties: S,
      then S2 .. S9).

    Not plurality: a student's label that spurious splits fill with the
    lecturer's words would map to L. Not one to one: a lecturer split across
    two labels is one person. Not the map's own output: ``derive_truth`` reads
    the map at the first word and at every empty ``who:``, so a map checked
    against the truth it produced can confirm a wrong guess. Raises SystemExit
    where the labels leave a label's identity entirely to the map: no stated
    word under it, and a ``real`` boundary with an empty ``who:`` into it.
    """
    rendered = sorted({lab for lab in stream.labels if lab is not None}, key=str)
    if not rendered:
        raise SystemExit("no rendered speaker labels to map")
    stated = derive_truth(labels, stream, {})
    tally: Dict[str, Dict[str, int]] = {x: {} for x in rendered}
    for lab, who in zip(stream.labels, stated):
        if lab is not None and who is not None and who != UNKNOWN_WHO:
            tally[lab][who] = tally[lab].get(who, 0) + 1
    n_l = {x: tally[x].get(INSTRUCTOR, 0) for x in rendered}
    n_s = {x: sum(v for k, v in tally[x].items() if is_student(k)) for x in rendered}
    tot_l, tot_s = sum(n_l.values()), sum(n_s.values())

    def instructor(x: str) -> bool:
        if tot_s == 0:
            return True                     # the student share is 0
        if tot_l == 0:
            return n_s[x] == 0              # the instructor share is 0
        return n_l[x] * tot_s >= n_s[x] * tot_l

    out = {x: INSTRUCTOR if instructor(x) else
           max(STUDENTS, key=lambda w, x=x: (tally[x].get(w, 0), -STUDENTS.index(w))) for x in rendered}
    empty_who = [b for b in labels["boundaries"] if b["verdict"] == "real" and not b["who"]]
    for b in empty_who:
        if b["to"] in tally and not tally[b["to"]]:
            raise SystemExit(f"{b['to']} has no word whose speaker the labels state, and {b['id']} "
                             "(real) into it leaves who: empty; fill that who: (an operator answer, "
                             "logged) or pass the map with --speaker-map")
    return out, {
        "rule": MAP_RULE,
        "stated_words": sum(1 for w in stated if w is not None and w != UNKNOWN_WHO),
        "unstated_words": sum(1 for w in stated if w is None),
        "real_boundaries_without_who": [b["id"] for b in empty_who],
        "instructor_words": tot_l, "student_words": tot_s,
        "labels": {x: {"rendered_words": sum(1 for lab in stream.labels if lab == x),
                       "stated_instructor_words": n_l[x], "stated_student_words": n_s[x],
                       "stated_by_who": dict(sorted(tally[x].items())), "who": out[x]}
                   for x in rendered}}


def resolve_speaker_map(spec: str, labels: Dict[str, Any],
                        stream: Stream) -> Tuple[Dict[str, str], Any]:
    """``auto``, or an explicit LABEL=WHO list. Returns (map, how it was set)."""
    if spec.strip().lower() == "auto":
        return derive_speaker_map(labels, stream)
    return parse_speaker_map(spec), "given"


# ---------------------------------------------------------------- scoring ---

def _direction(truth: str, shown: Optional[str]) -> str:
    if is_student(truth) and shown == INSTRUCTOR:
        return "student_as_instructor"
    if truth == INSTRUCTOR and is_student(shown):
        return "instructor_as_student"
    if is_student(truth) and is_student(shown):
        return "student_as_other_student"
    return "other"


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def near_misses(spurious: List[int], missed: List[int], n: int) -> List[int]:
    """The spurious boundary positions matched one to one with missed changes within n words.

    In word order, each boundary takes the earliest unmatched change in its
    window. Every window is the same width, so no other pairing matches more.
    """
    taken, out = set(), []
    for p in sorted(spurious):
        q = next((q for q in sorted(missed) if p - n <= q <= p + n and q not in taken), None)
        if q is not None:
            taken.add(q)
            out.append(p)
    return out


def score(labels: Dict[str, Any], stream: Stream, truth: List[Optional[str]],
          speaker_map: Dict[str, str], near: int = NEAR_WORDS) -> Dict[str, Any]:
    n = len(truth)
    shown = [speaker_map.get(lab) for lab in stream.labels]
    known = [t is not None and t != UNKNOWN_WHO for t in truth]
    err = [known[p] and truth[p] != shown[p] for p in range(n)]

    def changed(p: int) -> Optional[bool]:
        """Does the true speaker change entering word p? None if either side is unknown."""
        if p == 0 or truth[p - 1] is None or truth[p] is None:
            return None
        if truth[p - 1] == UNKNOWN_WHO and truth[p] == UNKNOWN_WHO:
            return None
        return truth[p] != truth[p - 1]

    def label_changed(p: int) -> bool:
        return p > 0 and stream.labels[p] != stream.labels[p - 1]

    def when(p: int) -> str:
        return dl.fmt_time(stream.start[p] or 0.0, stream.offset)

    # Boundaries: the verdicts, and where each sits against the per-word truth.
    bs = labels["boundaries"]
    verdicts = {v: sum(1 for b in bs if b["verdict"] == v) for v in dl.VERDICTS}
    unlabeled = sum(1 for b in bs if not b["verdict"])
    decided = verdicts["real"] + verdicts["spurious"]
    changes = [p for p in range(n) if changed(p)]
    pos = {i: k for k, i in enumerate(stream.index)}
    b_pos = sorted(stream.boundary_at)
    cross: Dict[str, Dict[str, int]] = {}
    for b in bs:
        p = pos[b["word_i"]]
        c = changed(p)
        if c is None:
            where = "unknown"
        elif c:
            where = "at_true_change"
        elif any(changed(q) for q in range(max(0, p - near), min(n, p + near + 1))):
            where = f"true_change_within_{near}_words"
        else:
            where = "no_true_change_nearby"
        row = cross.setdefault(b["verdict"] or "unlabeled", {})
        row[where] = row.get(where, 0) + 1

    # Near misses: spurious boundaries a few words off a change the label missed.
    missed_at = [p for p in changes if not label_changed(p)]
    spurious_at = [pos[b["word_i"]] for b in bs if b["verdict"] == "spurious"]
    within = []
    for k in range(1, near + 1):
        hits = near_misses(spurious_at, missed_at, k)
        within.append({"words": k, "near_misses": len(hits),
                       "precision": _ratio(verdicts["real"] + len(hits), decided),
                       "near_miss_boundaries": [stream.boundary_at[p] for p in hits]})

    # Error runs: consecutive error words with one true speaker and one label.
    runs: List[Tuple[int, int, str]] = []
    p = 0
    while p < n:
        if not err[p]:
            p += 1
            continue
        a = p
        while p + 1 < n and err[p + 1] and truth[p + 1] == truth[a] and stream.labels[p + 1] == stream.labels[a]:
            p += 1
        if a == 0 or not known[a - 1]:
            cause = "after_unknown"
        else:
            tc, lc = truth[a] != truth[a - 1], label_changed(a)
            cause = "wrong_label" if tc and lc else "missed_change" if tc else "spurious_split"
        runs.append((a, p, cause))
        p += 1
    run_at = {a: (a, e, c) for a, e, c in runs}
    by_direction = {d: 0 for d in DIRECTIONS}
    by_cause = {c: 0 for c in CAUSES}
    for a, e, c in runs:
        by_direction[_direction(truth[a], shown[a])] += e - a + 1
        by_cause[c] += e - a + 1

    # Missed changes: the true speaker changes, the rendered label does not.
    missed = []
    for p in changes:
        if label_changed(p):
            continue
        run = run_at.get(p)
        missed.append({
            "word_i": stream.index[p], "line_id": stream.line_of[p], "time": when(p),
            "from": truth[p - 1], "to": truth[p], "label": stream.labels[p],
            "misattributed_words": (run[1] - run[0] + 1) if run else 0,
            "direction": _direction(truth[p], shown[p]) if run else None,
            "nearest_boundary_words": min((abs(p - q) for q in b_pos), default=None),
        })
    hi = [m for m in missed if m["direction"] == "student_as_instructor"]

    # Unknown windows: maximal runs of words the sheet leaves unknown.
    windows = []
    p = 0
    while p < n:
        if known[p]:
            p += 1
            continue
        a = p
        while p + 1 < n and not known[p + 1]:
            p += 1
        windows.append({"first_word_i": stream.index[a], "last_word_i": stream.index[p],
                        "first_line": stream.line_of[a], "last_line": stream.line_of[p],
                        "time": when(a), "words": p - a + 1,
                        "opened_by": stream.boundary_at.get(a)})
        p += 1

    scored = sum(known)
    wrong = sum(err)
    student_words = sum(1 for p in range(n) if known[p] and is_student(truth[p]))
    return {
        "labels_complete": labels.get("complete"),
        "speaker_map": speaker_map,
        "near_words": near,
        "boundaries": {
            "total": len(bs), **verdicts, "unlabeled": unlabeled,
            "precision": _ratio(verdicts["real"], decided),
            "spurious_split_rate": _ratio(verdicts["spurious"], decided),
            "precision_if_unsure_all_spurious": _ratio(verdicts["real"], len(bs) - unlabeled),
            "precision_if_unsure_all_real": _ratio(verdicts["real"] + verdicts["unsure"], len(bs) - unlabeled),
            "precision_within_words": within,
            "verdict_vs_truth": cross,
        },
        "true_changes": {
            "total": len(changes),
            "at_a_boundary": len(changes) - len(missed),
            "missed": len(missed),
        },
        "missed_changes": {
            "count": len(missed),
            "misattributing": sum(1 for m in missed if m["misattributed_words"]),
            "words": sum(m["misattributed_words"] for m in missed),
            "student_in_instructor_turn": {"count": len(hi), "words": sum(m["misattributed_words"] for m in hi)},
            f"within_{near}_words_of_a_boundary": sum(
                1 for m in missed if m["nearest_boundary_words"] is not None and m["nearest_boundary_words"] <= near),
            "items": missed,
        },
        "words": {
            "total": n, "scored": scored, "excluded_unknown": n - scored,
            "correct": scored - wrong, "wrong": wrong, "accuracy": _ratio(scored - wrong, scored),
            "student_words": student_words,
            "student_as_instructor": by_direction["student_as_instructor"],
            "student_as_instructor_share_of_student_words": _ratio(by_direction["student_as_instructor"], student_words),
            "by_direction": by_direction,
            "by_cause": by_cause,
        },
        "unknown_windows": windows,
    }


def summary(r: Dict[str, Any]) -> List[str]:
    b, w, m, t = r["boundaries"], r["words"], r["missed_changes"], r["true_changes"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"   # noqa: E731
    near = r["near_words"]
    out = [
        f"boundaries: {b['total']}  real {b['real']}  spurious {b['spurious']}  unsure {b['unsure']}"
        + (f"  unlabeled {b['unlabeled']}" if b["unlabeled"] else ""),
        f"  precision (real / real+spurious): {pct(b['precision'])}; spurious splits {pct(b['spurious_split_rate'])}"
        f"; with unsure counted either way {pct(b['precision_if_unsure_all_spurious'])}"
        f"-{pct(b['precision_if_unsure_all_real'])}",
        "  precision within N words (a spurious boundary within N words of a missed change is a near "
        "miss): " + ", ".join(f"N={x['words']} {pct(x['precision'])} (+{x['near_misses']})"
                               for x in b["precision_within_words"]),
        "  verdict vs. per-word truth: " + "; ".join(
            f"{v}: " + ", ".join(f"{k} {c}" for k, c in sorted(row.items()))
            for v, row in r["boundaries"]["verdict_vs_truth"].items()),
        f"true speaker changes: {t['total']}, {t['at_a_boundary']} at a boundary, {t['missed']} missed",
        f"missed changes: {m['count']}; {m['misattributing']} misattribute {m['words']} word(s); "
        f"{m['student_in_instructor_turn']['count']} put student speech in the instructor's turn "
        f"({m['student_in_instructor_turn']['words']} word(s)); "
        f"{m[f'within_{near}_words_of_a_boundary']} lie within {near} words of a boundary",
        f"words: {w['total']}; scored {w['scored']} ({w['excluded_unknown']} unknown, excluded); "
        f"accuracy {pct(w['accuracy'])} ({w['wrong']} wrong)",
        f"  student words under the instructor's label: {w['student_as_instructor']} of {w['student_words']} "
        f"student words ({pct(w['student_as_instructor_share_of_student_words'])})",
        "  by direction: " + ", ".join(f"{k} {v}" for k, v in w["by_direction"].items()),
        "  by cause: " + ", ".join(f"{k} {v}" for k, v in w["by_cause"].items()),
    ]
    for win in r["unknown_windows"]:
        out.append(f"unknown window: {win['first_line']}-{win['last_line']} from {win['time']}, "
                   f"{win['words']} word(s)" + (f", opened by {win['opened_by']}" if win["opened_by"] else ""))
    return out


# ------------------------------------------------------------------ CLI -----

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", help="the *_labels.json that diarization_labels.py --check wrote")
    p.add_argument("--speaker-map", required=True,
                   help="who each diarizer label is, e.g. SPEAKER_01=L,SPEAKER_00=S; or 'auto', the "
                        "map the labels imply (docs/diarization_candidates_v2_plan.md §6.6)")
    p.add_argument("--source", default=None, help="the raw JSON, if not where the labels say")
    p.add_argument("--near", type=int, default=NEAR_WORDS,
                   help="words either side for a 'displaced' boundary; precision within N words is "
                        f"given for N = 1 to this (default {NEAR_WORDS})")
    p.add_argument("--allow-incomplete", action="store_true", help="score a sheet still being labeled")
    p.add_argument("--json", default=None, help="also write the numbers here")
    a = p.parse_args(argv)
    path = Path(a.labels)
    try:
        if a.speaker_map.strip().lower() != "auto":
            parse_speaker_map(a.speaker_map)            # a malformed map fails before any work
        labels = json.loads(path.read_text())
        if not labels.get("complete") and not a.allow_incomplete:
            raise SystemExit("the labels are not complete (an unlabeled boundary or an open change); "
                             "finish the sheet, or pass --allow-incomplete")
        stream = rendered_stream(labels, path, Path(a.source) if a.source else None)
        speaker_map, map_rule = resolve_speaker_map(a.speaker_map, labels, stream)
        unmapped = sorted({lab for lab in stream.labels if lab is not None} - set(speaker_map))
        if unmapped:
            raise SystemExit(f"--speaker-map does not say who {', '.join(unmapped)} is")
    except (OSError, ValueError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    result = score(labels, stream, derive_truth(labels, stream, speaker_map), speaker_map, a.near)
    result = {"labels": str(path), "labels_sha256": dl.sha256_file(path),
              "source": labels["source"], "source_sha256": labels["source_sha256"],
              **result, "speaker_map_rule": map_rule}
    if map_rule != "given":
        print("speaker map (auto): " + ",".join(f"{k}={v}" for k, v in sorted(speaker_map.items())))
    for ln in summary(result):
        print(ln)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
