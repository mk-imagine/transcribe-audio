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


F26 = rr.SEMESTERS[0]


@check("Fall 2026: Mon 24 Aug to Fri 11 Dec inclusive; weeks are seven-day blocks from the start")
def _():
    assert (F26["term"], F26["start"], F26["end"]) == ("F26", date(2026, 8, 24), date(2026, 12, 11))
    assert F26["start"].weekday() == 0 and F26["end"].weekday() == 4
    from datetime import time
    assert {c: v["time"] for c, v in F26["courses"].items()} == {
        "777": (time(14, 0), time(15, 40)), "498": (time(9, 0), time(11, 45)), "896": (time(13, 0), time(14, 30))}
    assert rr.semester_problems(rr.SEMESTERS) == []
    start = F26["start"]
    assert rr.week_number(date(2026, 8, 24), start) == 1
    assert rr.week_number(date(2026, 8, 26), start) == 1         # the first recorded lecture, a Wednesday
    assert rr.week_number(date(2026, 8, 30), start) == 1         # the following Sunday
    assert rr.week_number(date(2026, 8, 31), start) == 2         # Monday of week 2
    assert rr.week_number(date(2026, 12, 11), start) == 16       # the last day
    for d, want in ((date(2026, 8, 23), None), (date(2026, 8, 24), "F26"),
                    (date(2026, 12, 11), "F26"), (date(2026, 12, 12), None)):
        got = rr.semester_for(d, rr.SEMESTERS)
        assert (got and got["term"]) == want, (d, got)


@check("courses map to weekdays; a day with no class is refused with a reason")
def _():
    c = F26["courses"]
    assert rr.course_for(date(2026, 8, 31), None, c) == ("777", "")   # Mon
    assert rr.course_for(date(2026, 9, 1), None, c) == ("498", "")    # Tue
    assert rr.course_for(date(2026, 9, 2), None, c) == ("777", "")    # Wed
    assert rr.course_for(date(2026, 9, 4), None, c) == ("896", "")    # Fri: lab
    code, why = rr.course_for(date(2026, 9, 3), None, c)               # Thu
    assert code is None and "no class on Thu" in why
    assert rr.course_for(date(2026, 9, 5), None, c)[0] is None         # Sat


def _sem(name, term, start, end, courses):
    return {"name": name, "term": term, "start": start, "end": end, "courses": courses}


@check("a bad SEMESTERS table is refused: overlapping dates, a reversed range, a repeated term, an unknown day, a reversed window")
def _():
    one = {"777": {"days": ("Mon",), "time": None}}
    fall = _sem("Fall 2026", "F26", date(2026, 8, 24), date(2026, 12, 11), one)
    spring = _sem("Spring 2027", "S27", date(2027, 1, 25), date(2027, 5, 21), one)
    assert rr.semester_problems([spring, fall]) == []
    winter = _sem("Winter 2026", "W26", date(2026, 12, 11), date(2027, 1, 15), one)   # shares 11 Dec with fall
    probs = rr.semester_problems([fall, winter])
    assert len(probs) == 1 and "Fall 2026" in probs[0] and "overlaps" in probs[0], probs
    assert "after it ends" in rr.semester_problems([_sem("X", "X", date(2027, 5, 1), date(2027, 1, 1), one)])[0]
    probs = rr.semester_problems([fall, dict(spring, term="F26")])
    assert probs == ["term F26 is used by more than one semester"], probs
    probs = rr.semester_problems([_sem("X", "X", date(2027, 1, 1), date(2027, 5, 1), {"1": {"days": ("Tues",)}})])
    assert "unknown day(s) Tues" in probs[0], probs
    from datetime import time
    probs = rr.semester_problems([_sem("X", "X", date(2027, 1, 1), date(2027, 5, 1),
                                       {"1": {"days": ("Mon",), "time": (time(15, 40), time(14, 0))}})])
    assert "15:40-14:00 does not start before it ends" in probs[0], probs


def _bwf(path, start_s, seconds, *, rate=8, bext_size=1116):
    """A mono 16-bit WAV whose bext chunk says the audio starts `start_s` after midnight.

    The layout is the Tascam DR-40X's: bext (1116 bytes) first, then fmt, then data.
    """
    import struct
    data = bytes(2 * round(seconds * rate))
    bext = bytearray(bext_size)
    bext[320:338] = b"2026-09-2218:20:58"                        # OriginationDate/Time: file close, unused
    bext[338:346] = struct.pack("<Q", round(start_s * rate))     # TimeReference, in samples
    fmt = struct.pack("<HHIIHH", 1, 1, rate, 2 * rate, 2, 16)
    body = b"WAVE" + b"bext" + struct.pack("<I", bext_size) + bext \
        + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _hm(h, m):
    return h * 3600 + m * 60


@check("the recording's span comes from BWF bext: TimeReference to TimeReference + length")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # The values decoded from the DR-40X test recording 260922_0001.wav, 48 kHz.
        _bwf(td / "a.wav", 3164830848 / 48000, 11554560 / 96000, rate=48000)
        start, end = rr.recording_span(td / "a.wav")
        assert abs(start - 65933.976) < 1e-6 and abs(end - 66054.336) < 1e-6, (start, end)   # 18:18:53.976-18:20:54.336
        (td / "plain.wav").write_bytes(b"RIFF\x04\x00\x00\x00WAVE")                        # a WAV with no bext
        (td / "short.wav").write_bytes(b"RIFF")
        (td / "notes.txt").write_text("hello")
        for n in ("plain.wav", "short.wav", "notes.txt"):
            assert rr.recording_span(td / n) is None, n
        try:
            rr.recording_span(td / "missing.wav")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing file is an error, not 'no bext'")


@check("two courses on one day: the recording's span against each window, most overlap wins")
def _():
    from datetime import time
    courses = {"498": {"days": ("Mon",), "time": (time(9, 0), time(11, 45))},
               "777": {"days": ("Mon",), "time": (time(13, 0), time(15, 0))}}
    mon = date(2026, 8, 31)
    assert rr.course_for(mon, (_hm(9, 0), _hm(10, 53)), courses) == ("498", "")    # stopped early
    assert rr.course_for(mon, (_hm(9, 2), _hm(11, 55)), courses) == ("498", "")    # ran ten minutes over
    assert rr.course_for(mon, (_hm(11, 30), _hm(13, 10)), courses) == ("498", "")  # 15 min vs 10 min
    assert rr.course_for(mon, (_hm(12, 50), _hm(15, 5)), courses) == ("777", "")
    code, why = rr.course_for(mon, (_hm(11, 50), _hm(12, 50)), courses)
    assert code is None and "12:50 overlaps no course" in why, why
    code, why = rr.course_for(mon, None, courses)
    assert code is None and "no BWF time" in why, why
    tied = {"498": {"days": ("Mon",), "time": (time(9, 0), time(10, 0))},
            "777": {"days": ("Mon",), "time": (time(10, 0), time(11, 0))}}
    code, why = rr.course_for(mon, (_hm(9, 30), _hm(10, 30)), tied)
    assert code is None and "equally" in why, why
    half = {"498": {"days": ("Mon",), "time": (time(9, 0), time(11, 45))}, "777": {"days": ("Mon",), "time": None}}
    code, why = rr.course_for(mon, (_hm(9, 0), _hm(10, 0)), half)
    assert code is None and "777 has no time window" in why, why


@check("recorder names parse; anything else is left alone")
def _():
    assert rr.parse_recorder_name("260901_0004.wav") == (date(2026, 9, 1), 4, "wav")
    assert rr.parse_recorder_name("260901_0004.WAV")[2] == "WAV"
    assert rr.parse_recorder_name("251211_0009.wav") == (date(2025, 12, 11), 9, "wav")
    for bad in ("geisler.wav", "PSY498-week1.wav", "26090_0004.wav", "261301_0001.wav", "260901.wav"):
        assert rr.parse_recorder_name(bad) is None, bad


@check("names: PSY<code>-<term>-WK<N>-<Day>, the day always present; extension lowercased; same-day index last")
def _():
    assert rr.target_name("498", "F26", 1, date(2026, 9, 1), "WAV", same_day_index=None) == "PSY498-F26-WK1-Tue.wav"
    assert rr.target_name("777", "F26", 1, date(2026, 8, 31), "wav", same_day_index=None) == "PSY777-F26-WK1-Mon.wav"
    assert rr.target_name("777", "F26", 1, date(2026, 9, 2), "wav", same_day_index=None) == "PSY777-F26-WK1-Wed.wav"
    assert rr.target_name("896", "F26", 1, date(2026, 9, 4), "wav", same_day_index=None) == "PSY896-F26-WK1-Fri.wav"
    assert rr.target_name("498", "F26", 3, date(2026, 9, 15), "m4a", same_day_index=2) == "PSY498-F26-WK3-Tue-2.m4a"


def _seed(td, names):
    for n in names:
        (td / n).write_bytes(b"RIFF")


@check("a plan over a folder: right names, skips explained, nothing touched")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        names = ["260826_0001.wav", "260831_0001.wav", "260901_0002.wav", "260902_0003.wav", "260903_0004.wav",
                 "260904_0007.wav", "260908_0005.wav", "260908_0006.wav", "261211_0008.wav",
                 "260820_0001.wav", "261214_0009.wav", "260402_0010.wav", "geisler.wav", "notes.txt"]
        _seed(td, names)
        renames, skipped = rr.plan(td)
        got = {o.name: n.name for o, n in renames}
        assert got == {"260826_0001.wav": "PSY777-F26-WK1-Wed.wav",
                       "260831_0001.wav": "PSY777-F26-WK2-Mon.wav", "260901_0002.wav": "PSY498-F26-WK2-Tue.wav",
                       "260902_0003.wav": "PSY777-F26-WK2-Wed.wav", "260904_0007.wav": "PSY896-F26-WK2-Fri.wav",
                       "260908_0005.wav": "PSY498-F26-WK3-Tue-1.wav", "260908_0006.wav": "PSY498-F26-WK3-Tue-2.wav",
                       "261211_0008.wav": "PSY896-F26-WK16-Fri.wav"}, got                  # the last day
        why = {p.name: w for p, w in skipped}
        assert "no class on Thu" in why["260903_0004.wav"]
        for n in ("260820_0001.wav", "261214_0009.wav", "260402_0010.wav"):         # before, after, a past term
            assert "in no semester" in why[n], why[n]
        assert "not in YYMMDD" in why["geisler.wav"] and "not in YYMMDD" in why["notes.txt"]
        assert sorted(p.name for p in td.iterdir()) == sorted(names), "plan() must not touch files"


@check("several semesters: each file takes its own semester's courses, term and weeks; the gap between is skipped")
def _():
    fall = _sem("Fall 2026", "F26", date(2026, 8, 24), date(2026, 12, 11),
                {"498": {"days": ("Tue",), "time": None}})
    spring = _sem("Spring 2027", "S27", date(2027, 1, 25), date(2027, 5, 21),
                  {"498": {"days": ("Tue",), "time": None}, "600": {"days": ("Thu",), "time": None}})
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _seed(td, ["260901_0001.wav", "261215_0002.wav", "270126_0003.wav", "270128_0004.wav", "260903_0005.wav"])
        renames, skipped = rr.plan(td, semesters=[fall, spring])
        got = {o.name: n.name for o, n in renames}
        assert got == {"260901_0001.wav": "PSY498-F26-WK2-Tue.wav",     # same code, two terms, no clash
                       "270126_0003.wav": "PSY498-S27-WK1-Tue.wav",
                       "270128_0004.wav": "PSY600-S27-WK1-Thu.wav"}, got
        why = {p.name: w for p, w in skipped}
        assert "in no semester" in why["261215_0002.wav"], why              # winter break
        assert "F26" in why["260903_0005.wav"] and "no class on Thu" in why["260903_0005.wav"], why


@check("never overwrite: an existing target or a duplicate claim is skipped, not clobbered")
def _():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _seed(td, ["260901_0002.wav", "PSY498-F26-WK2-Tue.wav"])
        renames, skipped = rr.plan(td)
        why = {p.name: w for p, w in skipped}
        assert renames == [] and "already exists" in why["260901_0002.wav"], why
        assert (td / "PSY498-F26-WK2-Tue.wav").read_bytes() == b"RIFF", "the existing file is untouched"


@check("bext is only needed on a shared day: without it a one-course day still renames, a shared day skips")
def _():
    from datetime import time
    courses = {"498": {"days": ("Mon", "Tue"), "time": (time(9, 0), time(11, 45))},
               "777": {"days": ("Mon",), "time": (time(13, 0), time(15, 0))}}
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _seed(td, ["260901_0001.wav", "260831_0002.wav"])     # Tue (one course), Mon (shared); no bext
        _bwf(td / "260831_0003.wav", _hm(13, 2), 60)           # Mon afternoon, with bext
        renames, skipped = rr.plan(td, semesters=[_sem("Fall 2026", "F26", date(2026, 8, 24), date(2026, 12, 11), courses)])
        got = {o.name: n.name for o, n in renames}
        assert got == {"260901_0001.wav": "PSY498-F26-WK2-Tue.wav", "260831_0003.wav": "PSY777-F26-WK2-Mon-2.wav"}, got
        why = {p.name: w for p, w in skipped}
        assert "no BWF time" in why["260831_0002.wav"], why


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
        assert sorted(p.name for p in td.iterdir()) == ["PSY498-F26-WK2-Tue.wav", "PSY777-F26-WK2-Wed.wav"]
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td), "--apply"],
                           capture_output=True, text=True)
        assert "nothing to rename" in r.stdout, "a second run must be a no-op"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_recordings.py"), str(td / "nope")],
                           capture_output=True, text=True)
        assert r.returncode == 2


@check("the CLI refuses an unusable SEMESTERS table with exit 2, before touching anything")
def _():
    saved = rr.SEMESTERS
    one = {"498": {"days": ("Tue",), "time": None}}
    rr.SEMESTERS = [_sem("Fall 2026", "F26", date(2026, 8, 24), date(2026, 12, 11), one),
                    _sem("Winter 2026", "W26", date(2026, 12, 1), date(2027, 1, 15), one)]
    try:
        with tempfile.TemporaryDirectory() as d:
            td = Path(d); _seed(td, ["260901_0002.wav"])
            import contextlib, io
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                assert rr.main([str(td), "--apply"]) == 2
            assert "overlaps Winter 2026" in err.getvalue(), err.getvalue()
            assert [p.name for p in td.iterdir()] == ["260901_0002.wav"]
    finally:
        rr.SEMESTERS = saved


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
