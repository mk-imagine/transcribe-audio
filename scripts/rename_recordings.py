#!/usr/bin/env python3
"""Rename recorder files (YYMMDD_XXXX.wav) to course-and-week names.

    python scripts/rename_recordings.py data/            # dry run: shows old -> new
    python scripts/rename_recordings.py data/ --apply    # actually rename

The recorder names a file by date and a running counter, which says nothing
about what was recorded. The semester calendar does: a recording made on a
Tuesday in the third week of term is PSY 498, week 3. Edit the two blocks below
for a new semester.

Naming: PSY<code>-week<N>-<Day>.<ext>, e.g. PSY777-week1-Mon.wav,
PSY498-week1-Tue.wav. The day is always present: a course that meets twice a
week needs it, and one name shape for every file is easier to read and to
glob. Two recordings on the same day get -1, -2 in recorder order. Nothing is
ever overwritten; a clash is reported and the file is left alone. Files on days
with no class, before the semester, or not in the recorder's format are skipped
and listed.

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

#: First day of the semester. Week 1 is the seven days starting here.
#: Fall 2026: Monday 24 August. (31 August is the Monday of week 2.)
SEMESTER_START = date(2026, 8, 24)

#: Course code -> the days it meets, and an optional time window,
#: e.g. "time": (time(9, 0), time(11, 45)) for 09:00-11:45.
#: Times are only consulted when two courses share a day, and then every course
#: on that day needs one; the recording's span comes from its BWF `bext` chunk.
#: Leave time=None when one course per day is enough, as it is this semester.
COURSES: Dict[str, dict] = {
    "777": {"name": "Multivariate Statistics", "days": ("Mon", "Wed"), "time": None},
    "498": {"name": "Cognitive Neuroscience", "days": ("Tue",), "time": None},
    "896": {"name": "Lab", "days": ("Fri",), "time": None},
}

#: Department prefix in the new name.
PREFIX = "PSY"

# ------------------------------------------------------------------ logic ----

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
RECORDER = re.compile(r"^(?P<yymmdd>\d{6})_(?P<seq>\d+)\.(?P<ext>[A-Za-z0-9]+)$")


def week_number(day: date, start: date = SEMESTER_START) -> Optional[int]:
    """1 for the first seven days from `start`, 2 for the next, ... None before it."""
    delta = (day - start).days
    return None if delta < 0 else delta // 7 + 1


def courses_on(weekday: str, courses: Dict[str, dict] = COURSES) -> List[str]:
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


def course_for(day: date, span: Optional[Span], courses: Dict[str, dict] = COURSES) -> Tuple[Optional[str], str]:
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


def target_name(code: str, week: int, day: date, ext: str, *,
                same_day_index: Optional[int], prefix: str = PREFIX) -> str:
    name = f"{prefix}{code}-week{week}-{WEEKDAYS[day.weekday()]}"
    if same_day_index is not None:
        name += f"-{same_day_index}"
    return f"{name}.{ext.lower()}"


def plan(directory: Path, *, start: date = SEMESTER_START, courses: Dict[str, dict] = COURSES,
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
        week = week_number(d, start)
        if week is None:
            skipped.append((p, f"{d} is before the semester start {start}"))
            continue
        code, why = course_for(d, recording_span(p), courses)
        if code is None:
            skipped.append((p, f"{d} ({WEEKDAYS[d.weekday()]}): {why}"))
            continue
        new = p.with_name(target_name(code, week, d, ext, same_day_index=same_day_index[p], prefix=prefix))
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
    ap.add_argument("--start", type=lambda s: date.fromisoformat(s), default=SEMESTER_START,
                    help=f"first day of the semester, YYYY-MM-DD (default {SEMESTER_START})")
    ap.add_argument("--prefix", default=PREFIX, help=f"department prefix (default {PREFIX})")
    args = ap.parse_args(argv)

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2
    print(f"semester start {args.start} ({WEEKDAYS[args.start.weekday()]}); "
          + "; ".join(f"{args.prefix} {c} on {'/'.join(v['days'])}"
                      + (f" {v['time'][0]:%H:%M}-{v['time'][1]:%H:%M}" if v.get("time") else "")
                      for c, v in COURSES.items()))
    renames, skipped = plan(directory, start=args.start, prefix=args.prefix)
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
