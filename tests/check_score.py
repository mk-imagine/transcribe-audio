#!/usr/bin/env python3
"""Checks for scripts/score_diarization.py. Zero dependencies.

    python3 tests/check_score.py

A synthetic lecture, built here in code, goes the whole way: sheet generated,
filled, checked, scored. Every per-word truth and every number below is worked
out by hand from the table in RECORD, so a check failing means the scorer
reads the sheet differently from the way the table says it should.
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
import score_diarization as sd  # noqa: E402

SCRIPT = ROOT / "scripts" / "score_diarization.py"
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
# A = SPEAKER_00 is the instructor's label, B = SPEAKER_01 a student's. One
# line per sentence, a 1 s pause between lines (so smoothing flips nothing).
# Each row: diarizer label, text, the true speaker of each word (None:
# unknown), and how the row is labeled on the sheet.

A, B = "SPEAKER_00", "SPEAKER_01"
MAP = {A: "L", B: "S"}
RECORD = [
    # label, text, truth per word, sheet labeling
    (A, "Today we study memory.", "L L L L", None),
    (B, "And its limits.", "L L L", ("spurious", "")),            # B-001: an island
    (A, "First the hippocampus.", "L L L", ("spurious", "")),     # B-002
    (A, "Right, is it on the exam?", "L S S S S S", '>> change: S "is it"'),
    (A, "Yes it is.", "L L L", ">> back: L"),
    (B, "What about sleep?", "S S S", ("real", "S")),              # B-003
    (B, "Sleep helps consolidation.", "L L L", ">> change: L"),
    (A, "So remember that.", "L L L", ("spurious", "")),           # B-004: 3 words late
    (B, "I have another question.", "S2 S2 S2 S2", ("real", "S2")),  # B-005: the wrong student's label
    (A, "Go ahead.", "L L", ("real", "")),                         # B-006: who from the map
    (B, "Maybe later.", "U U", ("unsure", "")),                    # B-007
    (A, "Okay then.", "U U", ("spurious", "")),                    # B-008: still unknown
    (B, "Can I ask?", "S S S", ("real", "S")),                     # B-009
    (A, "that makes sense.", "S S S", ("spurious", "")),           # B-010: the student continues
    (A, "Moving on now.", "L L L", ">> change: L"),                # a miss the label already gets right
    (B, "Wait.", "S", ("real", "S")),                              # B-011
    (A, "Yes?", "L", ("real", "")),                                # B-012
]
TRUTH = [None if t == "U" else t for _, _, ts, _ in RECORD for t in ts.split()]

# A second record, for the near-miss view: spurious boundaries 1, 2 and 3 words
# off a change the label missed; one beside a change a real boundary already
# found; one with no change nearby; and two either side of a single missed
# change. Word positions in the comments: "@" a boundary, "^" a missed change.
NEAR_RECORD = [
    (A, "We begin with attention.", "L L L L", None),                  # 0-3
    (A, "Okay so yes.", "L L S", '>> change: S "yes."'),               # ^6
    (B, "Is attention limited?", "S S S", ("spurious", "")),          # B-001 @7: 1 word late
    (A, "Yes.", "L", ("real", "")),                                    # B-002 @10, at its change
    (B, "It is limited.", "L L L", ("spurious", "")),                 # B-003 @11: beside a found change
    (A, "Capacity is finite.", "L L L", ("spurious", "")),            # B-004 @14: none within 3 words
    (A, "Any questions now?", "L L L", None),                          # 17-19
    (B, "Take time.", "L L", ("spurious", "")),                       # B-005 @20: 2 words early
    (B, "Why though?", "S S", ">> change: S"),                         # ^22
    (A, "Because capacity is shared.", "L L L L", ("real", "")),      # B-006 @24, at its change
    (A, "Let us go on now.", "L L L L L", None),                       # 28-32
    (A, "Hold on, is that true?", "L L S S S", '>> change: S "is that"'),  # ^35
    (B, "I am not sure.", "S S S S", ("spurious", "")),               # B-007 @38: 3 words late
    (A, "It is true.", "L L L", ("real", "")),                         # B-008 @42, at its change
    (A, "Now the last point.", "L L L L", None),                       # 45-48
    (B, "Now.", "L", ("spurious", "")),                               # B-009 @49: 1 word early
    (B, "Wait.", "S", ">> change: S"),                                 # ^50
    (A, "Is that shared too?", "S S S S", ("spurious", "")),          # B-010 @51: ^50 is taken
    (A, "It is shared.", "L L L", ">> back: L"),                       # ^55, 4 words from B-010
    (A, "That is all.", "L L L", None),                                # 58-60
]
NEAR_TRUTH = [t for _, _, ts, _ in NEAR_RECORD for t in ts.split()]


def make_doc(record=RECORD):
    words, turns, t, i = [], [], 0.5, 0
    for label, text, _, _ in record:
        s0 = t
        for tok in text.split():
            words.append({"i": i, "text": tok, "start": round(t, 2), "end": round(t + 0.3, 2),
                          "timing_source": "native", "speaker": None, "speaker_source": None,
                          "conf": None, "flags": []})
            i += 1; t += 0.3
        turns.append({"start": round(s0 - 0.05, 2), "end": round(t + 0.05, 2), "speaker": label})
        t += 1.0
    return {
        "schema_version": "1.0",
        "source": {"audio_path": "data/synthetic.wav", "audio_sha256": "cd" * 32, "duration_s": t},
        "run": {"created_utc": "2026-10-02T00:00:00Z", "device": "cpu",
                "pipeline_version": {"commit": "0" * 40, "branch": "main", "dirty": False}},
        "asr": {"model_id": "mock", "revision": "0", "capabilities": {}, "granularity": "word",
                "params": {"mode": "intended"}, "performance": {}},
        "diarization": {"model_id": "mock-diarizer", "revision": "0", "params": {}},
        "words": words, "text": " ".join(w["text"] for w in words),
        "speaker_turns": turns, "errors": [], "warnings": [],
    }


def fill(sheet, skip=(), record=RECORD):
    """Label the generated sheet the way the record says; boundaries in ``skip`` stay blank.

    One sentence per row, so line Tnnnn is record[n - 1], and a boundary block
    belongs to the row of the line below it."""
    lines = sheet.read_text().splitlines()
    out = []
    for n, ln in enumerate(lines):
        if ln.startswith("T0"):
            row = record[int(ln[1:5]) - 1]
            if isinstance(row[3], str):
                out.append(row[3])
        elif ln.startswith(("boundary:", "who:")):
            bid = next(x for x in reversed(out) if x.startswith("#### B-")).split()[1]
            below = next(x for x in lines[n:] if x.startswith("T0"))
            verdict, who = record[int(below[1:5]) - 1][3]
            if ln.startswith("boundary:"):
                ln = "boundary:" if bid in skip else f"boundary: {verdict}"
            else:
                ln = f"who: {who}".rstrip()
        out.append(ln)
    sheet.write_text("\n".join(out) + "\n")


def workdir(skip=(), record=RECORD):
    """A temp dir with rec_raw.json, its filled sheet, and the checked labels."""
    td = tempfile.TemporaryDirectory()
    d = Path(td.name)
    (d / "rec_raw.json").write_text(json.dumps(make_doc(record), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        assert dl.main([str(d / "rec_raw.json")]) == 0
        fill(d / "rec_labels.md", skip, record)
        rc = dl.main(["--check", str(d / "rec_labels.md")])
    assert rc == 0, "the filled synthetic sheet does not check clean"
    return td, d, d / "rec_labels.json"


def scored(lab_path, speaker_map=MAP):
    labels = json.loads(lab_path.read_text())
    stream = sd.rendered_stream(labels, lab_path)
    truth = sd.derive_truth(labels, stream, speaker_map)
    return labels, stream, truth, sd.score(labels, stream, truth, speaker_map)


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


# ---------------------------------------------------------------- truth -----

@check("the synthetic sheet renders as designed: 12 boundaries, 17 lines, 49 words")
def _():
    td, d, lab = workdir()
    with td:
        labels, stream, _, _ = scored(lab)
        assert len(labels["boundaries"]) == 12 and labels["counts"]["text_lines"] == 17
        assert len(stream.index) == len(TRUTH) == 49
        assert [m["kind"] for m in labels["markers"]] == ["change", "back", "change", "change"]


@check("per-word truth matches the hand-worked table, word for word")
def _():
    td, d, lab = workdir()
    with td:
        _, _, truth, _ = scored(lab)
        assert truth == TRUTH, [(p, a, b) for p, (a, b) in enumerate(zip(truth, TRUTH)) if a != b]


@check("an unsure window stays open across a spurious boundary and closes at a marker")
def _():
    td, d, lab = workdir()
    with td:
        labels = json.loads(lab.read_text())
        stream = sd.rendered_stream(labels, lab)
        twelve = stream.index[stream.line_of.index("T0012")]
        labels["markers"].append({"kind": "change", "who": "L", "word_i": twelve})
        truth = sd.derive_truth(labels, stream, MAP)
        assert [truth[p] for p in range(34, 38)] == [None, None, "L", "L"]


@check("a real boundary with who ? is a speaker of unknown identity: excluded, not wrong")
def _():
    td, d, lab = workdir()
    with td:
        labels = json.loads(lab.read_text())
        labels["boundaries"][4]["who"] = "?"            # B-005
        stream = sd.rendered_stream(labels, lab)
        truth = sd.derive_truth(labels, stream, MAP)
        r = sd.score(labels, stream, truth, MAP)
        assert truth[28:32] == ["?"] * 4
        assert r["words"]["scored"] == 41 and r["words"]["wrong"] == 14


# --------------------------------------------------------------- numbers ----

@check("boundary verdicts and precision: 6 real, 5 spurious, 1 unsure")
def _():
    td, d, lab = workdir()
    with td:
        b = scored(lab)[3]["boundaries"]
        assert (b["total"], b["real"], b["spurious"], b["unsure"], b["unlabeled"]) == (12, 6, 5, 1, 0)
        assert b["precision"] == round(6 / 11, 4) and b["spurious_split_rate"] == round(5 / 11, 4)
        assert b["precision_if_unsure_all_spurious"] == 0.5
        assert b["precision_if_unsure_all_real"] == round(7 / 12, 4)


@check("verdict vs. truth: a boundary 3 words from the change it belongs to is 'within 3 words'")
def _():
    td, d, lab = workdir()
    with td:
        x = scored(lab)[3]["boundaries"]["verdict_vs_truth"]
        assert x == {"spurious": {"no_true_change_nearby": 2, "true_change_within_3_words": 2, "unknown": 1},
                     "real": {"at_true_change": 5, "unknown": 1},
                     "unsure": {"unknown": 1}}, x


@check("missed changes: 4, of which 2 misattribute 8 words and 1 puts 5 student words in the instructor's turn")
def _():
    td, d, lab = workdir()
    with td:
        r = scored(lab)[3]
        t, m = r["true_changes"], r["missed_changes"]
        assert (t["total"], t["at_a_boundary"], t["missed"]) == (9, 5, 4)
        assert (m["count"], m["misattributing"], m["words"]) == (4, 2, 8)
        assert m["student_in_instructor_turn"] == {"count": 1, "words": 5}
        assert m["within_3_words_of_a_boundary"] == 3
        assert [(i["line_id"], i["from"], i["to"], i["misattributed_words"], i["direction"])
                for i in m["items"]] == [
            ("T0004", "L", "S", 5, "student_as_instructor"),
            ("T0005", "S", "L", 0, None),
            ("T0007", "S", "L", 3, "instructor_as_student"),
            ("T0015", "S", "L", 0, None)]


@check("per-word: 45 scored, 27 right; 8 student words under the instructor's label")
def _():
    td, d, lab = workdir()
    with td:
        w = scored(lab)[3]["words"]
        assert (w["total"], w["scored"], w["excluded_unknown"], w["correct"], w["wrong"]) == (49, 45, 4, 27, 18)
        assert w["accuracy"] == 0.6 and w["student_words"] == 19
        assert w["student_as_instructor"] == 8
        assert w["by_direction"] == {"student_as_instructor": 8, "instructor_as_student": 6,
                                     "student_as_other_student": 4, "other": 0}
        assert w["by_cause"] == {"spurious_split": 6, "missed_change": 8, "wrong_label": 4,
                                 "after_unknown": 0}


@check("the unknown window is reported by line, with the boundary that opened it")
def _():
    td, d, lab = workdir()
    with td:
        win = scored(lab)[3]["unknown_windows"]
        assert [(x["first_line"], x["last_line"], x["words"], x["opened_by"]) for x in win] == \
               [("T0011", "T0012", 4, "B-007")], win


# ----------------------------------------------------------- near misses ----

@check("near misses: the near-miss record renders as designed and its truth matches the table")
def _():
    td, d, lab = workdir(record=NEAR_RECORD)
    with td:
        labels, stream, truth, _ = scored(lab)
        assert len(labels["boundaries"]) == 10 and len(stream.index) == len(NEAR_TRUTH) == 61
        assert {b["id"]: stream.index.index(b["word_i"]) for b in labels["boundaries"]} == {
            "B-001": 7, "B-002": 10, "B-003": 11, "B-004": 14, "B-005": 20,
            "B-006": 24, "B-007": 38, "B-008": 42, "B-009": 49, "B-010": 51}
        assert truth == NEAR_TRUTH, [(p, a, b) for p, (a, b) in enumerate(zip(truth, NEAR_TRUTH)) if a != b]


@check("near misses: precision within 1, 2, 3 words is 3+2, 3+3, 3+4 of 10")
def _():
    td, d, lab = workdir(record=NEAR_RECORD)
    with td:
        b = scored(lab)[3]["boundaries"]
        assert (b["real"], b["spurious"], b["precision"]) == (3, 7, 0.3)
        assert [(x["words"], x["near_misses"], x["precision"], x["near_miss_boundaries"])
                for x in b["precision_within_words"]] == [
            (1, 2, 0.5, ["B-001", "B-009"]),
            (2, 3, 0.6, ["B-001", "B-005", "B-009"]),
            (3, 4, 0.7, ["B-001", "B-005", "B-007", "B-009"])]


@check("near misses are one to one with missed changes: none beside a found change, none sharing one")
def _():
    td, d, lab = workdir(record=NEAR_RECORD)
    with td:
        b = scored(lab)[3]["boundaries"]
        # Six spurious boundaries have a true change within 3 words; four are near misses.
        # B-003's is B-002's change, already found; B-010's is B-009's.
        assert b["verdict_vs_truth"]["spurious"] == {"true_change_within_3_words": 6,
                                                     "no_true_change_nearby": 1}
        assert not {"B-003", "B-004", "B-010"} & set(b["precision_within_words"][-1]["near_miss_boundaries"])
        # The pairing is the largest there is: matching B-009 to ^50 leaves nothing for B-010.
        assert sd.near_misses([49, 51], [50], 1) == [49]
        assert sd.near_misses([51, 49], [50, 52], 1) == [49, 51]
        assert sd.near_misses([5], [2, 8], 2) == []


@check("near misses on the first record: 0, 0, 2 (B-004 3 words late, B-010 3 words early)")
def _():
    td, d, lab = workdir()
    with td:
        w = scored(lab)[3]["boundaries"]["precision_within_words"]
        assert [(x["near_misses"], x["precision"], x["near_miss_boundaries"]) for x in w] == [
            (0, round(6 / 11, 4), []), (0, round(6 / 11, 4), []),
            (2, round(8 / 11, 4), ["B-004", "B-010"])]


# ------------------------------------------------------------------ CLI -----

@check("CLI: exit 0, a summary, and a JSON with no transcript text in it")
def _():
    td, d, lab = workdir()
    with td:
        out = d / "score.json"
        r = run(lab, "--speaker-map", f"{A}=L,{B}=S", "--json", out)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "boundaries: 12  real 6  spurious 5  unsure 1" in r.stdout
        assert "missed changes: 4; 2 misattribute 8 word(s); 1 put student speech" in r.stdout
        assert "N=1 54.5% (+0), N=2 54.5% (+0), N=3 72.7% (+2)" in r.stdout, r.stdout
        r2 = run(lab, "--speaker-map", f"{A}=L,{B}=S", "--near", "2")
        assert r2.returncode == 0 and "N=2 54.5% (+0)\n" in r2.stdout, r2.stdout
        res = json.loads(out.read_text())
        assert res["words"]["accuracy"] == 0.6 and res["source_sha256"] == dl.sha256_file(d / "rec_raw.json")
        blob = out.read_text().lower()
        for tok in ("hippocampus", "consolidation", "exam", "sleep"):
            assert tok not in blob, f"transcript text {tok!r} leaked into the score"


@check("CLI: an incomplete sheet is refused unless --allow-incomplete")
def _():
    td, d, lab = workdir(skip=("B-002",))
    with td:
        r = run(lab, "--speaker-map", f"{A}=L,{B}=S")
        assert r.returncode == 2 and "not complete" in r.stderr, r.stderr
        r = run(lab, "--speaker-map", f"{A}=L,{B}=S", "--allow-incomplete")
        assert r.returncode == 0 and "unlabeled 1" in r.stdout, r.stdout + r.stderr


@check("CLI refuses: a speaker map missing a label, a malformed map")
def _():
    td, d, lab = workdir()
    with td:
        r = run(lab, "--speaker-map", f"{A}=L")
        assert r.returncode == 2 and f"does not say who {B} is" in r.stderr, r.stderr
        for bad in (f"{A}L", f"{A}=X", f"{A}=L,{A}=S", ""):
            r = run(lab, "--speaker-map", bad)
            assert r.returncode == 2 and "refused" in r.stderr, (bad, r.stderr)


@check("CLI refuses: a source whose sha256 differs, and labels from a different render")
def _():
    td, d, lab = workdir()
    with td:
        good = (d / "rec_raw.json").read_text()
        doc = make_doc(); doc["words"][0]["text"] = "Yesterday"
        (d / "rec_raw.json").write_text(json.dumps(doc))
        r = run(lab, "--speaker-map", f"{A}=L,{B}=S")
        assert r.returncode == 2 and "sha256 differs" in r.stderr, r.stderr
        (d / "rec_raw.json").write_text(good)
        labels = json.loads(lab.read_text())
        labels["boundaries"][0]["word_i"] += 1
        lab.write_text(json.dumps(labels))
        r = run(lab, "--speaker-map", f"{A}=L,{B}=S")
        assert r.returncode == 2 and "no longer matches" in r.stderr, r.stderr


# ---------------------------------------------------------- speaker map -----
#
# --speaker-map auto: the rule pre-registered in
# docs/diarization_candidates_v2_plan.md §6.6, as amended (A2). It counts only
# the words whose speaker the labels state. Each record below is built so that
# one part of the rule decides it.

C = "SPEAKER_02"

# B is a student's label that a spurious split fills with the lecturer's
# words: 6 stated as the lecturer's against 5 stated as the student's, so
# plurality would say L. Its share of the student's words is the larger.
SPLIT_RECORD = [
    (A, "Today we study memory and its limits.", "L L L L L L L", None),
    (B, "Is that on the exam?", "S S S S S", ("real", "S")),
    (A, "Yes it is.", "L L L", ("real", "L")),
    (B, "And then the cortex stores them.", "L L L L L L", ("spurious", "")),
    (A, "Over weeks of sleep and more.", "L L L L L L", ("spurious", "")),
]

# The lecturer under two labels, the smaller one first (A, then C), and one
# student word under C. The rule first registered here guessed the larger
# label as the lecturer, took the first word's speaker from that guess, and
# settled on A as a student; amendment A2 exists because of this record.
TWO_LECTURER_RECORD = [
    (A, "Today we study memory.", "L L L L", None),
    (B, "Is it on the exam?", "S S S S S", ("real", "S")),
    (A, "Yes it is.", "L L L", ("real", "L")),
    (C, "Now the second half begins.", "L L L L L", ("spurious", "")),
    (C, "Sleep helps okay thanks.", "L L S S", '>> change: S "okay"'),
    (C, "So that is all.", "L L L L", ">> back: L"),
]

# Nobody but the lecturer speaks; B is all spurious. Nothing is stated.
NO_STUDENT_RECORD = [
    (A, "Today we study memory.", "L L L L", None),
    (B, "And its limits.", "L L L", ("spurious", "")),
    (A, "Then sleep.", "L L", ("spurious", "")),
]

# A student's label holds the most words; every speaker but the first is stated.
BIG_STUDENT_RECORD = [
    (A, "Today we start.", "L L L", None),
    (B, "I will present my project in some detail today for you all.", " ".join(["S"] * 12), ("real", "S")),
    (A, "Thanks that was great.", "L L L L", ("real", "L")),
    (B, "One more point from me on the last slide here please.", " ".join(["S"] * 11), ("real", "S")),
    (A, "Good we move on.", "L L L L", ("real", "L")),
]

# The lecturer is never stated (the return is stated as unknown, ?). Only the
# student is, so B is S and A, with no stated student word, is L.
NO_LECTURER_STATED_RECORD = [
    (A, "Today we study memory.", "L L L L", None),
    (B, "Is it on the exam?", "S S S S S", ("real", "S")),
    (A, "Yes it is.", "? ? ?", ("real", "?")),
]

# B's only turn is a real boundary with an empty who: nothing says who B is.
UNSTATED_RECORD = [
    (A, "Today we study memory.", "L L L L", None),
    (B, "Is it on the exam?", "S S S S S", ("real", "")),
    (A, "Yes it is.", "L L L", ("real", "L")),
]


def auto_map(record):
    td, d, lab = workdir(record=record)
    with td:
        labels = json.loads(lab.read_text())
        stream = sd.rendered_stream(labels, lab)
        m, rule = sd.derive_speaker_map(labels, stream)
        truth = sd.derive_truth(labels, stream, m)
    want = [None if t == "U" else t for _, _, ts, _ in record for t in ts.split()]
    assert truth == want, "the truth under the derived map is not the record's"
    return m, rule


@check("auto map: the record above gives the map its checks use, from the words the labels state")
def _():
    m, rule = auto_map(RECORD)
    assert m == MAP and rule["real_boundaries_without_who"] == ["B-006", "B-012"]
    la, lb = rule["labels"][A], rule["labels"][B]
    assert (la["stated_instructor_words"], la["stated_student_words"]) == (9, 8)
    assert lb["stated_by_who"] == {"L": 3, "S": 7, "S2": 4}
    assert (rule["stated_words"], rule["unstated_words"]) == (31, 18)


@check("auto map: a student's label full of the lecturer's words is still the student's (share, not plurality)")
def _():
    m, rule = auto_map(SPLIT_RECORD)
    assert m == {A: "L", B: "S"}
    assert rule["labels"][B]["stated_by_who"] == {"L": 6, "S": 5}


@check("auto map: a lecturer split across two labels, the smaller first, maps to L twice")
def _():
    m, _ = auto_map(TWO_LECTURER_RECORD)
    assert m == {A: "L", B: "S", C: "L"}


@check("auto map: a student's label holding the most words; no student at all; no lecturer stated")
def _():
    assert auto_map(BIG_STUDENT_RECORD)[0] == {A: "L", B: "S"}
    m, rule = auto_map(NO_STUDENT_RECORD)
    assert m == {A: "L", B: "L"} and rule["stated_words"] == 0
    m, rule = auto_map(NO_LECTURER_STATED_RECORD)
    assert m == {A: "L", B: "S"} and rule["instructor_words"] == 0


@check("auto map refuses a label the labels leave entirely to the map")
def _():
    td, d, lab = workdir(record=UNSTATED_RECORD)
    with td:
        labels = json.loads(lab.read_text())
        try:
            sd.derive_speaker_map(labels, sd.rendered_stream(labels, lab))
            raise AssertionError("a label with no stated word was mapped")
        except SystemExit as exc:
            assert B in str(exc) and "B-001" in str(exc)
        r = run(lab, "--speaker-map", "auto")
        assert r.returncode == 2 and "who: empty" in r.stderr, r.stderr


@check("CLI: --speaker-map auto prints the map, records the rule, and scores as the given map does")
def _():
    td, d, lab = workdir()
    with td:
        out_a, out_g = d / "auto.json", d / "given.json"
        ra = run(lab, "--speaker-map", "auto", "--json", out_a)
        rg = run(lab, "--speaker-map", f"{A}=L,{B}=S", "--json", out_g)
        assert ra.returncode == 0 and rg.returncode == 0, ra.stderr + rg.stderr
        assert f"speaker map (auto): {A}=L,{B}=S" in ra.stdout
        a, g = json.loads(out_a.read_text()), json.loads(out_g.read_text())
        assert a["speaker_map_rule"]["stated_words"] == 31 and g["speaker_map_rule"] == "given"
        drop = ("speaker_map_rule",)
        assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in g.items() if k not in drop}


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
