#!/usr/bin/env python3
"""Rename recorder files (YYMMDD_XXXX.wav) to course-and-week names.

    python scripts/rename_recordings.py data/            # dry run: shows old -> new
    python scripts/rename_recordings.py data/ --apply    # actually rename

The recorder names a file by date and a running counter, which says nothing
about what was recorded. The semester calendar does: a recording made on a
Tuesday in the third week of Fall 2026 is PSY 498, F26, week 3. Add each new
semester to SEMESTERS below; keep the old ones, so an old recording still
renames correctly.

Naming: PSY<code>-<term>-WK<N>-<Day>.<ext>, e.g. PSY777-F26-WK1-Mon.wav,
PSY498-F26-WK1-Tue.wav. The term keeps a course code that recurs in a later
semester from colliding with its earlier names. The day is always present: a
course that meets twice a week needs it, and one name shape for every file is
easier to read and to glob. Two recordings on the same day get -1, -2 in
recorder order. Nothing is ever overwritten; a clash is reported and the file
is left alone. Files outside every semester, on days with no class, or not in
the recorder's format are skipped and listed.

On a day two courses share, the recording's own clock decides: the BWF `bext`
chunk the recorder writes into the file (set the recorder to BWF). Its
TimeReference is the first sample's time of day, and the audio's length gives
the end; the course whose window overlaps that span most wins. It is inside the
file, so no copy can change it -- unlike the file's modification time, which a
copy can reset. A file without `bext` is renamed as usual on a one-course day
and skipped, with the reason, on a shared one.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- calendar ---

#: One entry per semester; add new ones, keep old ones. Dates must not overlap.
#:   term:    the short code in the file name, e.g. F26, S27
#:   start:   first day; week 1 is the seven days starting here
#:   end:     last day, inclusive; a recording after it is not renamed
#:   courses: course code -> the days it meets, and an optional time window,
#:            e.g. "time": (time(9, 0), time(11, 45)) for 09:00-11:45. Times are
#:            only consulted when two courses share a day, and then every course
#:            on that day needs one; the recording's span comes from its BWF
#:            `bext` chunk. Leave time=None when one course per day is enough.
SEMESTERS: List[dict] = [
    {"name": "Fall 2026", "term": "F26", "start": date(2026, 8, 24), "end": date(2026, 12, 11),
     "courses": {
         "777": {"name": "Multivariate Statistics", "days": ("Mon", "Wed"), "time": None},
         "498": {"name": "Cognitive Neuroscience", "days": ("Tue",), "time": None},
         "896": {"name": "RADLab", "days": ("Fri",), "time": None},
     }},
]

#: Department prefix in the new name.
PREFIX = "PSY"

# ------------------------------------------------------------------ logic ----

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
RECORDER = re.compile(r"^(?P<yymmdd>\d{6})_(?P<seq>\d+)\.(?P<ext>[A-Za-z0-9]+)$")


def semester_problems(semesters: List[dict]) -> List[str]:
    """What is wrong with the SEMESTERS table; empty when it is usable."""
    problems = []
    for s in semesters:
        if s["start"] > s["end"]:
            problems.append(f"{s['name']}: starts {s['start']}, after it ends {s['end']}")
        for code, c in s["courses"].items():
            bad = [d for d in c["days"] if d not in WEEKDAYS]
            if bad:
                problems.append(f"{s['name']}: course {code} meets on unknown day(s) {', '.join(bad)}")
    terms = [s["term"] for s in semesters]
    problems += [f"term {t} is used by more than one semester" for t in sorted(set(terms)) if terms.count(t) > 1]
    ordered = sorted(semesters, key=lambda s: s["start"])
    for a, b in zip(ordered, ordered[1:]):
        if b["start"] <= a["end"]:
            problems.append(f"{a['name']} ({a['start']} to {a['end']}) overlaps "
                            f"{b['name']} ({b['start']} to {b['end']})")
    return problems


def semester_for(day: date, semesters: List[dict]) -> Optional[dict]:
    """The semester whose start..end (inclusive) holds `day`, or None."""
    for s in semesters:
        if s["start"] <= day <= s["end"]:
            return s
    return None


def week_number(day: date, start: date) -> int:
    """1 for the first seven days from `start`, 2 for the next, ..."""
    return (day - start).days // 7 + 1


def courses_on(weekday: str, courses: Dict[str, dict]) -> List[str]:
    return [code for code, c in courses.items() if weekday in c["days"]]


Span = Tuple[float, float]  # (start, end) of the audio, in seconds after midnight


def _seconds(t: time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def _clock(s: float) -> str:
    s = int(s) % 86400
    return f"{s // 3600:02d}:{s // 60 % 60:02d}"


def recording_span(path: Path) -> Optional[Span]:
    """The audio's (start, end) from the BWF `bext` chunk, or None if the file has none.

    TimeReference is the first sample's offset from midnight, in samples; the
    end adds the `data` chunk's length. On the Tascam DR-40X that end falls a
    few seconds before OriginationTime, which is therefore when the file was
    closed, not when it began, and is not used.
    """
    tref = rate = byte_rate = data_len = None
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            return None
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            cid, size = struct.unpack("<4sI", head)
            body_at = f.tell()
            if cid == b"bext" and size >= 346:
                f.seek(body_at + 338)  # past Description, Originator, OriginatorReference, Date, Time
                (tref,) = struct.unpack("<Q", f.read(8))
            elif cid == b"fmt " and size >= 16:
                _, _, rate, byte_rate = struct.unpack("<HHII", f.read(12))
            elif cid == b"data":
                data_len = size
            f.seek(body_at + size + (size & 1))
    if tref is None or not rate or not byte_rate or data_len is None:
        return None
    start = tref / rate
    return start, start + data_len / byte_rate


def course_for(day: date, span: Optional[Span], courses: Dict[str, dict]) -> Tuple[Optional[str], str]:
    """The course code for a recording on `day` spanning `span`, or (None, why)."""
    wd = WEEKDAYS[day.weekday()]
    hits = courses_on(wd, courses)
    if not hits:
        return None, f"no class on {wd}"
    if len(hits) == 1:
        return hits[0], ""
    # More than one course that day: the recording's span against every window.
    shared = f"{len(hits)} courses meet on {wd}"
    untimed = [c for c in hits if not courses[c].get("time")]
    if untimed:
        return None, f"{shared} and {', '.join(untimed)} has no time window"
    if span is None:
        return None, f"{shared} and the file has no BWF time (bext chunk) to choose between them"
    start, end = span
    overlap = {}
    for code in hits:
        t0, t1 = (_seconds(t) for t in courses[code]["time"])
        overlap[code] = min(end, t1) - max(start, t0)
    best = max(overlap.values())
    winners = [c for c, o in overlap.items() if o == best]
    heard = f"recorded {_clock(start)}-{_clock(end)}"
    if best <= 0:
        return None, f"{shared}; {heard} overlaps no course's window"
    if len(winners) > 1:
        return None, f"{shared}; {heard} overlaps {' and '.join(winners)} equally"
    return winners[0], ""


def parse_recorder_name(name: str) -> Optional[Tuple[date, int, str]]:
    m = RECORDER.match(name)
    if not m:
        return None
    try:
        d = datetime.strptime(m.group("yymmdd"), "%y%m%d").date()
    except ValueError:
        return None
    return d, int(m.group("seq")), m.group("ext")


def target_name(code: str, term: str, week: int, day: date, ext: str, *,
                same_day_index: Optional[int], prefix: str = PREFIX) -> str:
    name = f"{prefix}{code}-{term}-WK{week}-{WEEKDAYS[day.weekday()]}"
    if same_day_index is not None:
        name += f"-{same_day_index}"
    return f"{name}.{ext.lower()}"


def plan(directory: Path, *, semesters: List[dict] = SEMESTERS,
         prefix: str = PREFIX) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, str]]]:
    """(renames, skipped). Pure: touches nothing."""
    parsed = []
    skipped: List[Tuple[Path, str]] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        r = parse_recorder_name(p.name)
        if r is None:
            skipped.append((p, "not in YYMMDD_XXXX form"))
            continue
        parsed.append((p, *r))

    # Same-day recordings, in recorder order, get -1, -2, ... only when there is more than one.
    per_day: Dict[date, List[Tuple[int, Path]]] = {}
    for p, d, seq, _ in parsed:
        per_day.setdefault(d, []).append((seq, p))
    same_day_index: Dict[Path, Optional[int]] = {}
    for d, items in per_day.items():
        items.sort()
        for n, (_, p) in enumerate(items, 1):
            same_day_index[p] = n if len(items) > 1 else None

    renames: List[Tuple[Path, Path]] = []
    targets: Dict[Path, Path] = {}
    for p, d, seq, ext in parsed:
        sem = semester_for(d, semesters)
        if sem is None:
            skipped.append((p, f"{d} is in no semester"))
            continue
        code, why = course_for(d, recording_span(p), sem["courses"])
        if code is None:
            skipped.append((p, f"{d} ({WEEKDAYS[d.weekday()]}, {sem['term']}): {why}"))
            continue
        new = p.with_name(target_name(code, sem["term"], week_number(d, sem["start"]), d, ext,
                                      same_day_index=same_day_index[p], prefix=prefix))
        if new == p:
            continue
        if new.exists():
            skipped.append((p, f"target {new.name} already exists; not overwriting"))
            continue
        if new in targets:
            skipped.append((p, f"target {new.name} is also claimed by {targets[new].name}"))
            continue
        targets[new] = p
        renames.append((p, new))
    return renames, skipped


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default="data", help="folder of recordings (default: data)")
    ap.add_argument("--apply", action="store_true", help="rename; without it, only show the plan")
    ap.add_argument("--prefix", default=PREFIX, help=f"department prefix (default {PREFIX})")
    args = ap.parse_args(argv)

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2
    problems = semester_problems(SEMESTERS)
    if problems:
        print("SEMESTERS is not usable:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    for sem in SEMESTERS:
        print(f"{sem['term']} {sem['start']} ({WEEKDAYS[sem['start'].weekday()]}) to {sem['end']}: "
              + "; ".join(f"{args.prefix} {c} on {'/'.join(v['days'])}"
                          + (f" {v['time'][0]:%H:%M}-{v['time'][1]:%H:%M}" if v.get("time") else "")
                          for c, v in sem["courses"].items()))
    renames, skipped = plan(directory, semesters=SEMESTERS, prefix=args.prefix)
    for old, new in renames:
        print(f"  {old.name:24s} -> {new.name}")
    for p, why in skipped:
        print(f"  {p.name:24s}    skipped: {why}")
    if not renames:
        print("nothing to rename")
        return 0
    if not args.apply:
        print(f"{len(renames)} rename(s) planned; re-run with --apply to do them")
        return 0
    for old, new in renames:
        old.rename(new)
    print(f"renamed {len(renames)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
