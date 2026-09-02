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
glob. Two recordings on the same day get -1, -2 in recorder order. Nothing is ever overwritten; a clash is reported and the
file is left alone. Files on days with no class, before the semester, or not in
the recorder's format are skipped and listed.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- calendar ---

#: First day of the semester. Week 1 is the seven days starting here.
SEMESTER_START = date(2026, 8, 31)

#: Course code -> the days it meets, and an optional time window.
#: Times are only consulted when two courses share a day; the recorder's
#: filename carries no time, so the file's modification time is used, which is
#: only as reliable as the copy that put the file where it is. Leave time=None
#: when one course per day is enough, as it is this semester.
COURSES: Dict[str, dict] = {
    "777": {"name": "Multivariate Statistics", "days": ("Mon", "Wed"), "time": None},
    "498": {"name": "Cognitive Neuroscience", "days": ("Tue",), "time": None},
    "896": {"name": "Lab", "days": ("Fri",), "time": None},
}

#: Files the calendar cannot place, or places wrongly: current name -> (code, week, day).
#: A file listed here is renamed from this table and nothing is inferred from its
#: name. This is where a recorder whose date was wrong, or a file that arrived
#: under some other name, gets pinned by the person who knows what it is.
OVERRIDES: Dict[str, Tuple[str, int, str]] = {
    "tate_1.m4a": ("777", 1, "Mon"),
    "260831_0015.wav": ("777", 1, "Wed"),      # the recorder's date says Monday; the lecture was Wednesday
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


def course_for(day: date, when: Optional[time], courses: Dict[str, dict] = COURSES) -> Tuple[Optional[str], str]:
    """The course code for a recording on `day`, or (None, why)."""
    wd = WEEKDAYS[day.weekday()]
    hits = courses_on(wd, courses)
    if not hits:
        return None, f"no class on {wd}"
    if len(hits) == 1:
        return hits[0], ""
    # More than one course that day: a time window has to decide.
    timed = [c for c in hits if courses[c].get("time")]
    if when is None or not timed:
        return None, f"{len(hits)} courses meet on {wd} and no time window resolves it"
    for code in timed:
        t0, t1 = courses[code]["time"]
        if t0 <= when <= t1:
            return code, ""
    return None, f"{len(hits)} courses meet on {wd}; {when:%H:%M} falls in no course's window"


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
         prefix: str = PREFIX, overrides: Dict[str, Tuple[str, int, str]] = OVERRIDES,
         ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, str]]]:
    """(renames, skipped). Pure: touches nothing."""
    parsed = []
    skipped: List[Tuple[Path, str]] = []
    renames: List[Tuple[Path, Path]] = []
    targets: Dict[Path, Path] = {}

    def claim(p: Path, new: Path) -> None:
        if new == p:
            return
        if new.exists():
            skipped.append((p, f"target {new.name} already exists; not overwriting"))
        elif new in targets:
            skipped.append((p, f"target {new.name} is also claimed by {targets[new].name}"))
        else:
            targets[new] = p
            renames.append((p, new))

    # Overrides first: pinned by hand, inferred from nothing.
    for name, (code, week, day) in overrides.items():
        p = directory / name
        if not p.is_file():
            skipped.append((p, "override names a file that is not here"))
            continue
        if code not in courses or day not in WEEKDAYS or week < 1:
            skipped.append((p, f"override ({code}, {week}, {day}) names an unknown course or day"))
            continue
        ext = p.suffix.lstrip(".").lower() or "wav"
        claim(p, p.with_name(f"{prefix}{code}-week{week}-{day}.{ext}"))

    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.name in overrides:
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

    for p, d, seq, ext in parsed:
        week = week_number(d, start)
        if week is None:
            skipped.append((p, f"{d} is before the semester start {start}"))
            continue
        when = datetime.fromtimestamp(p.stat().st_mtime).time()
        code, why = course_for(d, when, courses)
        if code is None:
            skipped.append((p, f"{d} ({WEEKDAYS[d.weekday()]}): {why}"))
            continue
        claim(p, p.with_name(target_name(code, week, d, ext, same_day_index=same_day_index[p], prefix=prefix)))
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
          + "; ".join(f"{args.prefix} {c} on {'/'.join(v['days'])}" for c, v in COURSES.items())
          + (f"; {len(OVERRIDES)} override(s)" if OVERRIDES else ""))
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
