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


def make_doc():
    words, turns, t, i = [], [], 0.5, 0
    for label, text, _, _ in RECORD:
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


def fill(sheet, skip=()):
    """Label the generated sheet the way RECORD says; boundaries in ``skip`` stay blank.

    One sentence per row, so line Tnnnn is RECORD[n - 1], and a boundary block
    belongs to the row of the line below it."""
    lines = sheet.read_text().splitlines()
    out = []
    for n, ln in enumerate(lines):
        if ln.startswith("T0"):
            row = RECORD[int(ln[1:5]) - 1]
            if isinstance(row[3], str):
                out.append(row[3])
        elif ln.startswith(("boundary:", "who:")):
            bid = next(x for x in reversed(out) if x.startswith("#### B-")).split()[1]
            below = next(x for x in lines[n:] if x.startswith("T0"))
            verdict, who = RECORD[int(below[1:5]) - 1][3]
            if ln.startswith("boundary:"):
                ln = "boundary:" if bid in skip else f"boundary: {verdict}"
            else:
                ln = f"who: {who}".rstrip()
        out.append(ln)
    sheet.write_text("\n".join(out) + "\n")


def workdir(skip=()):
    """A temp dir with rec_raw.json, its filled sheet, and the checked labels."""
    td = tempfile.TemporaryDirectory()
    d = Path(td.name)
    (d / "rec_raw.json").write_text(json.dumps(make_doc(), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        assert dl.main([str(d / "rec_raw.json")]) == 0
        fill(d / "rec_labels.md", skip)
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


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
