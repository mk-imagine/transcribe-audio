#!/usr/bin/env python3
"""Checks for scripts/rename_recordings.py. Zero dependencies.

    python3 tests/check_rename.py
"""

import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rename_recordings as rr  # noqa: E402

PASSED, FAILED = [], []


def check(name):
    def wrap(fn):
        try:
            fn(); PASSED.append(name)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        return fn
    return wrap


@check("the semester starts on a Monday, and weeks are seven-day blocks from it")
def _():
    assert rr.SEMESTER_START.weekday() == 0, "2026-08-31 should be a Monday"
    assert rr.week_number(date(2026, 8, 31)) == 1
    assert rr.week_number(date(2026, 9, 6)) == 1          # the following Sunday
    assert rr.week_number(date(2026, 9, 7)) == 2
    assert rr.week_number(date(2026, 12, 7)) == 15
    assert rr.week_number(date(2026, 8, 30)) is None      # the day before


@check("courses map to weekdays; a day with no class is refused with a reason")
def _():
    assert rr.course_for(date(2026, 8, 31), None) == ("777", "")   # Mon
    assert rr.course_for(date(2026, 9, 1), None) == ("498", "")    # Tue
    assert rr.course_for(date(2026, 9, 2), None) == ("777", "")    # Wed
    assert rr.course_for(date(2026, 9, 4), None) == ("896", "")    # Fri: lab
    code, why = rr.course_for(date(2026, 9, 3), None)               # Thu
    assert code is None and "no class on Thu" in why
    assert rr.course_for(date(2026, 9, 5), None)[0] is None         # Sat


@check("two courses on one day need a time window; the file's time decides")
def _():
    from datetime import time
    courses = {"777": {"days": ("Mon",), "time": (time(9, 0), time(11, 0))},
               "498": {"days": ("Mon",), "time": (time(13, 0), time(15, 0))}}
    assert rr.course_for(date(2026, 8, 31), time(10, 0), courses) == ("777", "")
    assert rr.course_for(date(2026, 8, 31), time(14, 0), courses) == ("498", "")
    assert rr.course_for(date(2026, 8, 31), time(12, 0), courses)[0] is None
    assert rr.course_for(date(2026, 8, 31), None, courses)[0] is None


@check("recorder names parse; anything else is left alone")
def _():
    assert rr.parse_recorder_name("260901_0004.wav") == (date(2026, 9, 1), 4, "wav")
    assert rr.parse_recorder_name("260901_0004.WAV")[2] == "WAV"
    assert rr.parse_recorder_name("251211_0009.wav") == (date(2025, 12, 11), 9, "wav")
    for bad in ("geisler.wav", "PSY498-week1.wav", "26090_0004.wav", "261301_0001.wav", "260901.wav"):
        assert rr.parse_recorder_name(bad) is None, bad


@check("names: PSY<code>-week<N>-<Day>, the day always present; extension lowercased; same-day index last")
def _():
    assert rr.target_name("498", 1, date(2026, 9, 1), "WAV", same_day_index=None) == "PSY498-week1-Tue.wav"
    assert rr.target_name("777", 1, date(2026, 8, 31), "wav", same_day_index=None) == "PSY777-week1-Mon.wav"
    assert rr.target_name("777", 1, date(2026, 9, 2), "wav", same_day_index=None) == "PSY777-week1-Wed.wav"
    assert rr.target_name("896", 1, date(2026, 9, 4), "wav", same_day_index=None) == "PSY896-week1-Fri.wav"
    assert rr.target_name("498", 3, date(2026, 9, 15), "m4a", same_day_index=2) == "PSY498-week3-Tue-2.m4a"


def _seed(td, names):
    for n in names:
        (td / n).write_bytes(b"RIFF")


@check("a plan over a folder: right names, skips explained, nothing touched")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _seed(td, ["260831_0001.wav", "260901_0002.wav", "260902_0003.wav", "260903_0004.wav", "260904_0007.wav",
                   "260908_0005.wav", "260908_0006.wav", "260829_0001.wav", "geisler.wav", "notes.txt"])
        renames, skipped = rr.plan(td)
        got = {o.name: n.name for o, n in renames}
        assert got == {"260831_0001.wav": "PSY777-week1-Mon.wav", "260901_0002.wav": "PSY498-week1-Tue.wav",
                       "260902_0003.wav": "PSY777-week1-Wed.wav", "260904_0007.wav": "PSY896-week1-Fri.wav",
                       "260908_0005.wav": "PSY498-week2-Tue-1.wav", "260908_0006.wav": "PSY498-week2-Tue-2.wav"}, got
        why = {p.name: w for p, w in skipped}
        assert "no class on Thu" in why["260903_0004.wav"]
        assert "before the semester" in why["260829_0001.wav"]
        assert "not in YYMMDD" in why["geisler.wav"] and "not in YYMMDD" in why["notes.txt"]
        assert sorted(p.name for p in td.iterdir()) == sorted(
            ["260831_0001.wav", "260901_0002.wav", "260902_0003.wav", "260903_0004.wav", "260904_0007.wav",
             "260908_0005.wav", "260908_0006.wav", "260829_0001.wav", "geisler.wav", "notes.txt"]), "plan() must not touch files"


@check("never overwrite: an existing target or a duplicate claim is skipped, not clobbered")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _seed(td, ["260901_0002.wav", "PSY498-week1-Tue.wav"])
        renames, skipped = rr.plan(td)
        why = {p.name: w for p, w in skipped}
        assert renames == [] and "already exists" in why["260901_0002.wav"], why
        assert (td / "PSY498-week1-Tue.wav").read_bytes() == b"RIFF", "the existing file is untouched"


@check("the CLI: dry run by default, --apply renames, exit codes")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d); _seed(td, ["260901_0002.wav", "260902_0003.wav"])
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td)],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "--apply" in r.stdout and (td / "260901_0002.wav").exists(), r.stdout
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td), "--apply"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "renamed 2" in r.stdout, r.stdout
        assert sorted(p.name for p in td.iterdir()) == ["PSY498-week1-Tue.wav", "PSY777-week1-Wed.wav"]
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td), "--apply"],
                           capture_output=True, text=True)
        assert "nothing to rename" in r.stdout, "a second run must be a no-op"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td / "nope")],
                           capture_output=True, text=True)
        assert r.returncode == 2


@check("--start overrides the semester start")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d); _seed(td, ["260901_0002.wav"])
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td), "--start", "2026-08-24"],
                           capture_output=True, text=True)
        assert "PSY498-week2-Tue.wav" in r.stdout, r.stdout


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
