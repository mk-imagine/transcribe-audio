#!/usr/bin/env python3
"""Checks for scripts/diarization_labels.py. Zero dependencies.

    python3 tests/check_labels.py

A small synthetic record exercises the sheet end to end: deterministic
generation, a filled sheet read back through the checker, and every parse
error the checker promises to report. The real sheet is never committed (it
holds student speech), so the synthetic record is built here, in code.
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import diarization_labels as dl  # noqa: E402

SCRIPT = ROOT / "scripts" / "diarization_labels.py"
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


# ----------------------------------------------------------- the record -----
#
# A lecturer (SPEAKER_00), a student question the diarizer catches
# (SPEAKER_01), and a student question it misses inside the lecturer's turn.
# The raw turns also carry three label runs that never become rendered
# boundaries: a one-word island that smoothing flips back, a turn over
# silence, and a turn every word of which a longer A turn outvotes.

A, B = "SPEAKER_00", "SPEAKER_01"
INTENDED = [  # (text, start, end); a gap over 0.5 s ends a line
    ("So", 0.5, 0.8), ("today", 0.8, 1.2), ("we", 1.2, 1.4), ("cover", 1.4, 1.8),
    ("memory.", 1.8, 2.4), ("Any", 3.0, 3.3), ("questions?", 3.3, 3.9),
    ("Is", 10.5, 10.7), ("this", 10.7, 10.9), ("on", 10.9, 11.0), ("the", 11.0, 11.1),
    ("exam?", 11.1, 11.6),
    ("Yes", 13.5, 13.8), ("it", 13.8, 13.9), ("is.", 13.9, 14.2),
    ("Now", 15.0, 15.2), ("the", 15.2, 15.3), ("hippocampus", 15.3, 16.0), ("matters.", 16.0, 16.5),
    ("Okay,", 26.2, 26.4), ("okay.", 26.4, 26.6),
    ("Can", 27.2, 27.4), ("you", 27.4, 27.5), ("repeat", 27.5, 27.8), ("that?", 27.8, 28.1),
    ("Sure,", 28.7, 29.0), ("the", 29.0, 29.1), ("hippocampus", 29.1, 29.8), ("again.", 29.8, 30.3),
]
TURNS = [
    (0.0, 10.0, A), (10.4, 13.2, B), (13.4, 15.2, A),
    (15.2, 15.3, B),                     # takes "the": an island smoothing flips back
    (15.3, 20.0, A),
    (25.0, 25.5, B),                     # over silence: no word at all
    (26.0, 31.0, A),
    (28.8, 29.2, B),                     # every word it overlaps goes to the A turn
]


def word(i, text, s, e):
    return {"i": i, "text": text, "start": s, "end": e, "timing_source": "native",
            "speaker": None, "speaker_source": None, "conf": None, "flags": []}


def make_doc():
    intended = [word(i, *w) for i, w in enumerate(INTENDED)]
    verbatim = [word(0, "[UM]", 0.1, 0.4)] + [word(i + 1, *w) for i, w in enumerate(INTENDED)]
    verbatim[0]["flags"] = ["filled_pause"]
    return {
        "schema_version": "1.0",
        "source": {"audio_path": "data/synthetic.wav", "audio_sha256": "ab" * 32, "duration_s": 31.0},
        "run": {"created_utc": "2026-10-02T00:00:00Z", "device": "cpu",
                "pipeline_version": {"commit": "0" * 40, "branch": "main", "dirty": False}},
        "asr": {"model_id": "mock", "revision": "0", "capabilities": {}, "granularity": "word",
                "params": {"mode": "verbatim"}, "performance": {}},
        "diarization": {"model_id": "mock-diarizer", "revision": "0", "params": {}},
        "words": verbatim,
        "text": " ".join(w["text"] for w in verbatim),
        "speaker_turns": [{"start": s, "end": e, "speaker": spk} for s, e, spk in TURNS],
        "secondary_stream": {"mode": "intended", "text": " ".join(t for t, _, _ in INTENDED),
                             "words": intended},
        "errors": [], "warnings": [],
    }


def workdir():
    """A temp dir holding rec_raw.json and its freshly generated sheet."""
    td = tempfile.TemporaryDirectory()
    d = Path(td.name)
    (d / "rec_raw.json").write_text(json.dumps(make_doc(), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        assert dl.main([str(d / "rec_raw.json")]) == 0
    return td, d, d / "rec_labels.md"


def run_check(sheet):
    labels, errors, notes = dl.check(sheet)
    return labels, errors, notes


# --------------------------------------------------- editing helpers -------

def lines_of(sheet):
    return sheet.read_text().splitlines()


def insert_above(sheet, needle, new):
    """Insert ``new`` above the first transcript line containing ``needle``."""
    ls = lines_of(sheet)
    k = next(n for n, ln in enumerate(ls) if ln.startswith("T0") and needle in ln)
    ls.insert(k, new)
    sheet.write_text("\n".join(ls) + "\n")
    return k + 1                                # 1-based line number of the insert


def set_field(sheet, bid, field, value):
    ls = lines_of(sheet)
    k = next(n for n, ln in enumerate(ls) if ln.startswith(f"#### {bid} "))
    j = next(n for n in range(k, len(ls)) if ls[n].startswith(f"{field}:"))
    ls[j] = f"{field}: {value}"
    sheet.write_text("\n".join(ls) + "\n")
    return j + 1


def expect_error(sheet, *fragments):
    _, errors, _ = run_check(sheet)
    joined = "\n".join(errors)
    for f in fragments:
        assert f in joined, f"expected {f!r} in errors:\n{joined}"
    return errors


# ----------------------------------------------------------- generation -----

@check("the sheet renders assign + smooth on the intended stream: 2 boundary blocks from 7 raw changes")
def _():
    td, d, sheet = workdir()
    with td:
        text = sheet.read_text()
        assert text.count("\n#### B-") == 2, text
        assert "#### B-001  00:00:10.5  SPEAKER_00 -> SPEAKER_01\nboundary:\nwho:\n" \
               "T0003 [00:00:10.5] Is this on the exam?\n" in text
        assert "#### B-002  00:00:13.5  SPEAKER_01 -> SPEAKER_00\n" in text
        assert "#### START  SPEAKER_00\nT0001 [00:00:00.5] So today we cover memory.\n" in text
        assert "[UM]" not in text, "the lecture profile's stream is the intended one"
        assert "- stream: intended" in text
        assert "smoothing reassigned 1 word(s)" in text
        # the island ("the" at 15.2 s) was flipped back, so its line is unbroken
        assert "T0005 [00:00:15.0] Now the hippocampus matters.\n" in text
        assert "- boundary blocks: 2; text lines: 8" in text
        assert "7 label changes there, 2 here" in text
        assert "1 hold no word at all and 1 lose every word" in text
        assert "4 changes remain (word-level: 4). Smoothing then removes 2." in text


@check("every transcript line has a stable id and a timestamp; the header carries source, sha256, params")
def _():
    td, d, sheet = workdir()
    with td:
        body = sheet.read_text().split(dl.TRANSCRIPT_RULE, 1)[1]
        tl = [ln for ln in body.splitlines() if ln.startswith("T")]
        assert [ln[:5] for ln in tl] == [f"T{n:04d}" for n in range(1, 9)]
        import re
        assert all(re.match(r"^T\d{4} \[\d\d:\d\d:\d\d\.\d\] \S", ln) for ln in tl), tl
        text = sheet.read_text()
        assert "- source: rec_raw.json\n" in text
        assert f"- source sha256: {dl.sha256_file(d / 'rec_raw.json')}\n" in text
        assert "pause_threshold 0.5 s, max_island 2, no speaker map" in text
        flat = " ".join(text.split())
        assert "Open the audio only when the text is ambiguous; mark `unsure` rather than guess." in flat


@check("generation is deterministic: two runs give identical bytes")
def _():
    td, d, sheet = workdir()
    with td:
        first = sheet.read_text(); times = (d / "rec_label-times.txt").read_text()
        with contextlib.redirect_stdout(io.StringIO()):
            assert dl.main([str(d / "rec_raw.json"), "--force"]) == 0
        assert sheet.read_text() == first and (d / "rec_label-times.txt").read_text() == times
        a = dl.build(make_doc(), "x_raw.json", "0").render()
        b = dl.build(make_doc(), "x_raw.json", "0").render()
        assert a == b


@check("the timestamps file lists hh:mm:ss per boundary, rounded down")
def _():
    td, d, sheet = workdir()
    with td:
        t = (d / "rec_label-times.txt").read_text().splitlines()
        assert t[2:] == ["00:00:10  B-001  SPEAKER_00 -> SPEAKER_01",
                         "00:00:13  B-002  SPEAKER_01 -> SPEAKER_00"], t


@check("an existing sheet is never overwritten without --force (it may hold labels)")
def _():
    td, d, sheet = workdir()
    with td:
        set_field(sheet, "B-001", "boundary", "real")
        before = sheet.read_text()
        r = subprocess.run([sys.executable, str(SCRIPT), str(d / "rec_raw.json")],
                           capture_output=True, text=True)
        assert r.returncode == 2 and "--force" in r.stderr
        assert sheet.read_text() == before


# ------------------------------------------------------------ round trip ----

@check("round trip: a filled sheet parses back to the labels typed, and the CLI writes the JSON")
def _():
    td, d, sheet = workdir()
    with td:
        set_field(sheet, "B-001", "boundary", "real"); set_field(sheet, "B-001", "who", "S")
        set_field(sheet, "B-002", "boundary", "Real"); set_field(sheet, "B-002", "who", "l")
        n1 = insert_above(sheet, "Can you repeat that?", ">> change: S")
        n2 = insert_above(sheet, "Sure, the hippocampus again.", '>>back:L  "the hippocampus"')
        r = subprocess.run([sys.executable, str(SCRIPT), "--check", str(sheet)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "boundaries labeled: 2 / 2" in r.stdout and "1 change, 1 back" in r.stdout
        lab = json.loads((d / "rec_labels.json").read_text())
        assert lab["complete"] is True and lab["stream"] == "intended"
        assert [(b["id"], b["verdict"], b["who"], b["word_i"]) for b in lab["boundaries"]] == \
               [("B-001", "real", "S", 7), ("B-002", "real", "L", 12)]
        assert [(m["kind"], m["who"], m["line_id"], m["word_i"], m["sheet_line"], m["quote"])
                for m in lab["markers"]] == [("change", "S", "T0007", 21, n1, None),
                                             ("back", "L", "T0008", 26, n2, "the hippocampus")]
        assert lab["markers"][1]["time"] == "00:00:29.0"
        assert lab["counts"]["raw_changes"] == 7 and lab["source_sha256"] == dl.sha256_file(d / "rec_raw.json")


@check("progress while labeling: partial verdicts count, blank lines are free, an open change is a note")
def _():
    td, d, sheet = workdir()
    with td:
        set_field(sheet, "B-001", "boundary", "unsure")
        insert_above(sheet, "Okay, okay.", "")
        insert_above(sheet, "Any questions?", ">> change: S")   # student asks; B-002 not labeled yet
        labels, errors, notes = run_check(sheet)
        assert not errors, errors
        assert dl.progress(labels)[0].startswith("boundaries labeled: 1 / 2  real 0  spurious 0  unsure 1")
        assert notes and "still open past an unlabeled boundary" in notes[0]
        assert labels["complete"] is False


@check("a change to a second student while one is open is fine; a spurious boundary keeps it open")
def _():
    td, d, sheet = workdir()
    with td:
        set_field(sheet, "B-001", "boundary", "spurious"); set_field(sheet, "B-002", "boundary", "real")
        insert_above(sheet, "Any questions?", ">> change: S")
        insert_above(sheet, "Is this on the exam?", ">> change: S2")
        _, errors, _ = run_check(sheet)
        assert not errors, errors


# ---------------------------------------------------------- parse errors ----

def fresh(edit):
    """Apply ``edit(sheet)`` to a fresh sheet whose boundaries are all labeled."""
    td, d, sheet = workdir()
    set_field(sheet, "B-001", "boundary", "real"); set_field(sheet, "B-002", "boundary", "real")
    out = edit(sheet)
    return td, sheet, out


ERROR_CASES = [
    ("a marker with one >",
     lambda s: insert_above(s, "Can you", "> change: S"), "malformed marker"),
    ("a misspelled marker keyword",
     lambda s: insert_above(s, "Can you", ">> chnage: S"), "malformed marker"),
    ("a marker without >>",
     lambda s: insert_above(s, "Can you", "change: S"), "malformed marker"),
    ("a marker with no who",
     lambda s: insert_above(s, "Can you", ">> change:"), "malformed marker"),
    ("an unknown who in a marker",
     lambda s: insert_above(s, "Can you", ">> change: X"), "unknown who 'X' in marker"),
    ("an unknown verdict",
     lambda s: set_field(s, "B-001", "boundary", "yes"), "B-001: unknown verdict 'yes'"),
    ("an unknown who on a boundary",
     lambda s: set_field(s, "B-001", "who", "Bob"), "B-001: unknown who 'Bob'"),
    ("who on a spurious boundary",
     lambda s: (set_field(s, "B-001", "boundary", "spurious"), set_field(s, "B-001", "who", "S"))[1],
     "`who:` on a spurious boundary"),
    ("back with no open change",
     lambda s: insert_above(s, "Sure, the", ">> back: L"), "back with no open change"),
    ("a change that never returns",
     lambda s: insert_above(s, "Can you", ">> change: S"), "change never returns"),
    ("a change to the speaker already open",
     lambda s: (insert_above(s, "Okay, okay.", ">> change: S"), insert_above(s, "Can you", ">> change: S"))[1],
     "while the change to S at line"),
    ("back to the one already speaking",
     lambda s: (insert_above(s, "Can you", ">> change: S"), insert_above(s, "Sure, the", ">> back: S"))[1],
     "back to S, who is the one already speaking"),
    ("a marker directly above a boundary block",
     lambda s: insert_above(s, "Any questions?", "") and _above_block(s, "B-001", ">> change: S"),
     "marker directly above boundary B-001"),
    ("a marker inside a boundary block",
     lambda s: _inside_block(s, "B-001", ">> change: S"), "marker inside boundary block B-001"),
    ("a marker above the transcript",
     lambda s: _above_rule(s, ">> change: S"), "marker above the transcript"),
    ("a quote that is not in the line",
     lambda s: insert_above(s, "Can you", '>> change: S "the exam"'), "is not a run of whole words in T0007"),
    ("a quote that matches part of a word only",
     lambda s: insert_above(s, "Can you", '>> change: S "repea"'), "is not a run of whole words"),
    ("an ambiguous quote",
     lambda s: _two_okays(s), "occurs 2 times in T0006"),
    ("an empty quote",
     lambda s: insert_above(s, "Can you", '>> change: S ""'), "the quote is empty"),
    ("two markers at one word",
     lambda s: (insert_above(s, "Can you", ">> change: S"), insert_above(s, "Can you", ">> change: S2"))[1],
     "two markers at the same word of T0007"),
    ("markers out of order within a line",
     lambda s: (insert_above(s, "Sure, the", '>> change: S "again"'),
                insert_above(s, "Sure, the", '>> back: L "hippocampus again"'))[1],
     "out of order"),
    ("a stray typed line",
     lambda s: insert_above(s, "Can you", "student asks here I think"), "unrecognized line"),
    ("an edited transcript line",
     lambda s: _replace(s, "T0007 [00:00:27.2] Can you repeat that?", "T0007 [00:00:27.2] Can you repeat this?"),
     "transcript line T0007 was edited"),
    ("a deleted transcript line",
     lambda s: _replace(s, "T0006 [00:00:26.2] Okay, okay.", None), "1 expected line(s) missing above it"),
    ("a deleted who: field",
     lambda s: _replace_in_block(s, "B-002", "who:", None), "missing above it, starting with 'who:'"),
    ("a deleted tail",
     lambda s: _replace(s, "T0008 [00:00:28.7] Sure, the hippocampus again.", None),
     "end of sheet: 1 expected line(s) missing"),
    ("a marker with nothing below it",
     lambda s: _append(s, ">> back: L"), "marker with no transcript line below it"),
    ("an edited header line",
     lambda s: _replace(s, "- sheet format: 1", "- sheet format: 2"), "unrecognized line '- sheet format: 2'"),
]


def _above_block(s, bid, new):
    ls = lines_of(s); k = next(n for n, ln in enumerate(ls) if ln.startswith(f"#### {bid} "))
    ls.insert(k, new); s.write_text("\n".join(ls) + "\n"); return k + 1


def _inside_block(s, bid, new):
    ls = lines_of(s); k = next(n for n, ln in enumerate(ls) if ln.startswith(f"#### {bid} "))
    ls.insert(k + 2, new); s.write_text("\n".join(ls) + "\n"); return k + 3


def _above_rule(s, new):
    ls = lines_of(s); k = ls.index(dl.TRANSCRIPT_RULE)
    ls.insert(k, new); s.write_text("\n".join(ls) + "\n"); return k + 1


def _two_okays(s):
    return insert_above(s, "Okay, okay.", '>> change: S "okay"')   # "Okay," and "okay." both match


def _replace(s, old, new):
    ls = lines_of(s)
    if old not in ls:
        return None
    k = ls.index(old)
    if new is None:
        del ls[k]
    else:
        ls[k] = new
    s.write_text("\n".join(ls) + "\n"); return k + 1


def _replace_in_block(s, bid, old, new):
    ls = lines_of(s); k = next(n for n, ln in enumerate(ls) if ln.startswith(f"#### {bid} "))
    j = next(n for n in range(k, len(ls)) if ls[n] == old)
    del ls[j]; s.write_text("\n".join(ls) + "\n"); return j + 1


def _append(s, new):
    ls = lines_of(s) + [new]; s.write_text("\n".join(ls) + "\n"); return len(ls)


for _name, _edit, _frag in ERROR_CASES:
    def _case(edit=_edit, frag=_frag):
        td, sheet, _ = fresh(edit)
        with td:
            errors = expect_error(sheet, frag)
            # Every error carries a line number, or says it is at the end of the sheet.
            assert all(e.startswith("line ") or e.startswith("end of sheet") for e in errors), errors
            r = subprocess.run([sys.executable, str(SCRIPT), "--check", str(sheet)],
                               capture_output=True, text=True)
            assert r.returncode == 1 and "not written" in r.stdout, r.stdout
            assert not (sheet.parent / "rec_labels.json").exists(), "no JSON from a sheet with errors"
    check(f"parse error reported: {_name}")(_case)


@check("errors name the line the operator typed")
def _():
    td, sheet, line = fresh(lambda s: insert_above(s, "Can you", "> change: S"))
    with td:
        errors = expect_error(sheet, "malformed marker")
        assert errors[0].startswith(f"line {line}: "), (line, errors)


@check("refused with exit 2: a source whose sha256 differs, and a missing source")
def _():
    td, d, sheet = workdir()
    with td:
        doc = make_doc(); doc["words"][1]["text"] = "Soo"
        (d / "rec_raw.json").write_text(json.dumps(doc))
        r = subprocess.run([sys.executable, str(SCRIPT), "--check", str(sheet)], capture_output=True, text=True)
        assert r.returncode == 2 and "sha256 differs" in r.stderr, r.stderr
        (d / "rec_raw.json").unlink()
        r = subprocess.run([sys.executable, str(SCRIPT), "--check", str(sheet)], capture_output=True, text=True)
        assert r.returncode == 2 and "not found" in r.stderr, r.stderr


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
