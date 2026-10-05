#!/usr/bin/env python3
"""Checks for the diarization bake-off tooling. Zero dependencies.

    python3 tests/check_bakeoff.py

Covers scripts/diarization_bakeoff.py (the converters, the speaker-map rule,
the swap into a copy of the record, the identity check, the scoring and the
bar) and the stdlib side of hpc/bakeoff/run_arm.py (it imports with no ML
package installed, and what it writes is what the converters read). A
synthetic lecture is built here in code, labeled, checked and scored; every
expected number below is worked out by hand from the table in RECORD.
"""

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import diarization_bakeoff as bo  # noqa: E402
import diarization_labels as dl  # noqa: E402
import score_diarization as sd  # noqa: E402

spec = importlib.util.spec_from_file_location("run_arm", ROOT / "hpc" / "bakeoff" / "run_arm.py")
run_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_arm)

PLAN = (ROOT / "docs" / "diarization_bakeoff_plan.md").read_text()
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


def raises(fn, *args, match=""):
    try:
        fn(*args)
    except bo.Refused as exc:
        assert match in str(exc), f"refused, but with {exc!r}"
        return
    raise AssertionError("expected a refusal")


# ----------------------------------------------------------- the record -----
#
# One line per sentence, a 1 s pause between lines, so smoothing flips nothing.
# A = SPEAKER_00 is the instructor's label in the stored (community-1) turns,
# B = SPEAKER_01 a student's. Each row: stored label, text, true speaker of
# each word, and how the sheet is labeled.

A, B = "SPEAKER_00", "SPEAKER_01"
GOLD_MAP = f"{A}=L,{B}=S"
RECORD = [
    (A, "Today we study memory.", "L L L L", None),                    # words 0-3
    (A, "Is that on the exam?", "S S S S S", ">> change: S"),          # 4-8: missed, high harm
    (A, "Yes it is.", "L L L", ">> back: L"),                          # 9-11
    (B, "What about sleep?", "S S S", ("real", "S")),                  # 12-14: B-001
    (A, "Sleep helps memory.", "L L L", ("real", "")),                 # 15-17: B-002
    (B, "And dreams too.", "L L L", ("spurious", "")),                 # 18-20: B-003
    (A, "Any more questions?", "L L L", ("spurious", "")),             # 21-23: B-004
    (A, "I have one.", "S2 S2 S2", ">> change: S2"),                   # 24-26: missed, high harm
    (A, "Go ahead.", "L L", ">> back: L"),                             # 27-28
]
TRUTH = [t for _, _, ts, _ in RECORD for t in ts.split()]

# Challenger arms, one label per row.
GOOD = ["speaker_0", "speaker_1", "speaker_0", "speaker_1", "speaker_0",    # every change found,
        "speaker_0", "speaker_0", "speaker_2", "speaker_0"]                     # nothing split
OVERSPLIT = [A, "V", A, B, A, "W", A, A, A]                    # finds row 2, splits row 6, misses row 8


def row_times():
    out, t = [], 0.5
    for _, text, _, _ in RECORD:
        s0 = t
        t += 0.3 * len(text.split())
        out.append((round(s0 - 0.05, 3), round(t + 0.05, 3)))
        t += 1.0
    return out


def turns_for(labels):
    return [{"start": s, "end": e, "speaker": lab} for (s, e), lab in zip(row_times(), labels)]


def make_doc():
    words, t, i = [], 0.5, 0
    for _, text, _, _ in RECORD:
        for tok in text.split():
            words.append({"i": i, "text": tok, "start": round(t, 2), "end": round(t + 0.3, 2),
                          "timing_source": "native", "speaker": None, "speaker_source": None,
                          "conf": None, "flags": []})
            i += 1; t += 0.3
        t += 1.0
    return {
        "schema_version": "1.0",
        "source": {"audio_path": "data/synthetic.wav", "audio_sha256": "cd" * 32, "duration_s": t},
        "run": {"created_utc": "2026-10-05T00:00:00Z", "device": "cpu",
                "pipeline_version": {"commit": "0" * 40, "branch": "main", "dirty": False},
                "packages": {"pyannote.audio": "4.0.3"}},
        "asr": {"model_id": "mock", "revision": "0", "capabilities": {}, "granularity": "word",
                "params": {"mode": "intended"}, "performance": {}},
        "diarization": {"model_id": "pyannote/speaker-diarization-community-1", "revision": "0", "params": {}},
        "words": words, "text": " ".join(w["text"] for w in words),
        "speaker_turns": turns_for([r[0] for r in RECORD]), "errors": [], "warnings": [],
    }


def fill(sheet):
    """Label the generated sheet as RECORD says. Line Tnnnn is RECORD[n - 1]."""
    lines = sheet.read_text().splitlines()
    out = []
    for n, ln in enumerate(lines):
        if ln.startswith("T0"):
            row = RECORD[int(ln[1:5]) - 1]
            if isinstance(row[3], str):
                out.append(row[3])
        elif ln.startswith(("boundary:", "who:")):
            below = next(x for x in lines[n:] if x.startswith("T0"))
            verdict, who = RECORD[int(below[1:5]) - 1][3]
            ln = f"boundary: {verdict}" if ln.startswith("boundary:") else f"who: {who}".rstrip()
        out.append(ln)
    sheet.write_text("\n".join(out) + "\n")


def workdir():
    td = tempfile.TemporaryDirectory()
    d = Path(td.name)
    (d / "rec_raw.json").write_text(json.dumps(make_doc(), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        assert dl.main([str(d / "rec_raw.json")]) == 0
        fill(d / "rec_labels.md")
        rc = dl.main(["--check", str(d / "rec_labels.md")])
    assert rc == 0, "the filled synthetic sheet does not check clean"
    return td, d, d / "rec_labels.json"


def write_arm(d, arm, run, fmt, labels, audio_sha="cd" * 32):
    """An arm run as hpc/bakeoff/run_arm.py lays it out: native output + provenance.json."""
    o = d / "bakeoff" / arm / run
    o.mkdir(parents=True)
    turns = turns_for(labels)
    if fmt == "rttm":
        name, text = "out.rttm", "".join(
            f"SPEAKER rec 1 {t['start']:.3f} {t['end'] - t['start']:.3f} <NA> <NA> {t['speaker']} <NA> <NA>\n"
            for t in turns)
    elif fmt == "nemo-segments":
        name, text = "segments.txt", "".join(f"{t['start']:.3f} {t['end']:.3f} {t['speaker']}\n" for t in turns)
    else:
        name, text = "turns.json", json.dumps(turns)
    (o / name).write_text(text)
    prov = {"bakeoff_format": 1, "arm": arm, "run": run, "audio": {"sha256": audio_sha},
            "model": {"id": f"mock/{arm}", "revision": "0"}, "package": {"name": "mock", "version": "0"},
            "call": {"params": {}}, "native_output": {"path": name, "format": fmt,
                                                      "sha256": bo.sha256_file(o / name)},
            "performance": {"diarize_s": 1.0}}
    (o / "provenance.json").write_text(json.dumps(prov))
    return o / "provenance.json"


def score_cli(*args):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = bo.main(["score", *map(str, args)])
    return rc, buf.getvalue(), err.getvalue()


# ----------------------------------------------------------- converters -----

@check("rttm: pyannote's ten-field lines become turns; onset + duration is the end")
def _():
    t = bo.convert("# a comment\n\nSPEAKER rec 1 0.500 1.250 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
                   "SPEAKER rec 1 0.100 0.200 <NA> <NA> 3 <NA> <NA>\n", "rttm")
    assert t == [{"start": 0.1, "end": 0.3, "speaker": "3"},
                 {"start": 0.5, "end": 1.75, "speaker": "SPEAKER_00"}], t


@check("rttm refuses: a short line, another record type, two files, a negative duration")
def _():
    raises(bo.parse_rttm, "SPEAKER rec 1 0.5 1.0 <NA> <NA> S0 <NA>\n", match="10 fields")
    raises(bo.parse_rttm, "SPKR-INFO rec 1 <NA> <NA> <NA> unknown S0 <NA> <NA>\n", match="SPEAKER")
    raises(bo.parse_rttm, "SPEAKER a 1 0 1 <NA> <NA> S0 <NA> <NA>\nSPEAKER b 1 1 1 <NA> <NA> S0 <NA> <NA>\n",
           match="2 files")
    raises(bo.parse_rttm, "SPEAKER a 1 1.0 -0.5 <NA> <NA> S0 <NA> <NA>\n", match="negative")


@check("nemo segments: NeMo's own formatter ('%.3f %.3f speaker_N'), and nothing looser")
def _():
    assert bo.convert("12.000 13.500 speaker_1\n0.080 2.400 speaker_0\n", "nemo-segments") == [
        {"start": 0.08, "end": 2.4, "speaker": "speaker_0"},
        {"start": 12.0, "end": 13.5, "speaker": "speaker_1"}]
    # The card's prose says 'begin_seconds, end_seconds, speaker_index'; the code it
    # runs writes space-separated, three decimals. A line in any other shape is refused.
    for bad in ("0.08 2.4 speaker_0", "0.080, 2.400, speaker_0", "0.080 2.400 0", "2.400 0.080 speaker_0"):
        raises(bo.parse_nemo_segments, bad + "\n")


@check("turns json: exactly start/end/speaker, a string label, a forward interval")
def _():
    assert bo.convert('[{"start": 1.5, "end": 2, "speaker": "A"}]', "turns-json") == [
        {"start": 1.5, "end": 2.0, "speaker": "A"}]
    raises(bo.parse_turns_json, '[{"start": 1, "end": 2, "speaker": "A", "x": 1}]', match="exactly")
    raises(bo.parse_turns_json, '[{"start": 1, "end": 2, "speaker": 0}]', match="string")
    raises(bo.parse_turns_json, '[{"start": 2, "end": 1, "speaker": "A"}]', match="bad interval")
    raises(bo.parse_turns_json, '{"start": 1}', match="list")
    raises(bo.convert, "", "csv", match="unknown format")


@check("runner -> converter: what run_arm.py writes is what the converters read")
def _():
    class Seg:
        def __init__(self, s, e):
            self.start, self.end = s, e
    text = run_arm.turns_json([(Seg(0.123456789, 1.5), "SPEAKER_00"), (Seg(2.0, 3.25), 7)])
    assert bo.convert(text, "turns-json") == [{"start": 0.123456789, "end": 1.5, "speaker": "SPEAKER_00"},
                                              {"start": 2.0, "end": 3.25, "speaker": "7"}]
    nemo = [f"{a:.3f} {b:.3f} speaker_{int(k)}" for a, b, k in ((0.0, 1.04, 0), (1.04, 9.5, 1))]
    assert len(bo.convert("".join(f"{s}\n" for s in nemo), "nemo-segments")) == 2


@check("runner: imports with no ML package, and its pins are the plan's pins")
def _():
    assert set(run_arm.ARMS) == {"community1", "diarizen", "sortformer"}
    for name, cfg in run_arm.ARMS.items():
        assert cfg["model"]["revision"] in PLAN, f"{name}: revision not in the plan"
        for key in ("commit", "version"):
            if key in cfg["package"]:
                assert cfg["package"][key] in PLAN, f"{name}: package {key} not in the plan"
    assert run_arm.ARMS["diarizen"]["model"]["embedding"]["revision"] in PLAN
    s = run_arm.ARMS["sortformer"]["streaming"]
    assert s == {"spkcache_len": 264, "fifo_len": 40, "chunk_len": 340,
                 "chunk_right_context": 40, "spkcache_update_period": 300}
    for name in ("torch", "pyannote", "nemo", "diarizen"):
        assert name not in sys.modules, f"{name} was imported at module scope"


@check("runner: hub snapshot paths give their revision; a WAV header is read with the stdlib")
def _():
    assert run_arm.snapshot_revision("/h/models--x--y/snapshots/abc123/f.bin") == "abc123"
    assert run_arm.snapshot_revision("/h/models--x--y/snapshots/abc123") == "abc123"
    assert run_arm.snapshot_revision("/tmp/x.nemo") == ""
    import wave
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes(b"\0\0" * 48000 * 2)
        a = run_arm.audio_block(p)
        assert (a["sample_rate"], a["channels"], a["duration_s"]) == (48000, 1, 2.0), a
        assert a["sha256"] == bo.sha256_file(p)


# ------------------------------------------------------- the speaker map ----

def mapped(rows):
    """rows: [(label, {who: count})] -> map_labels over the expanded sequence."""
    labels, truth = [], []
    for lab, row in rows:
        for who, n in row.items():
            labels += [lab] * n
            truth += [who] * n
    return bo.map_labels(labels, truth)


@check("map: a student label holding mostly instructor words still maps to the student (one to one)")
def _():
    # community-1 on the real lecture had this shape: majority would map both labels to L.
    m = mapped([("P", {"L": 100, "S": 5}), ("Q", {"L": 12, "S": 10, "S2": 2})])
    assert m.speaker_map == {"P": "L", "Q": "S"} and m.folded == {}, m


@check("map: exact, not greedy; pairs share at least one word; leftovers fold by majority")
def _():
    # Greedy takes P->L (10) and is stuck; the optimum is P->S + Q->L (18).
    assert mapped([("P", {"L": 10, "S": 9}), ("Q", {"L": 9})]).paired == {"P": "S", "Q": "L"}
    # Q shares nothing with S, so it is not paired with it: it folds into L.
    m = mapped([("P", {"L": 10, "S": 2}), ("Q", {"L": 3})])
    assert m.paired == {"P": "L"} and m.folded == {"Q": "L"} and m.words_in_folded == 3, m
    # Four labels, three identities: the fourth folds into its majority.
    m = mapped([("P", {"L": 50}), ("Q", {"S": 9}), ("R", {"S2": 4}), ("T", {"S": 3, "L": 1})])
    assert m.paired == {"P": "L", "Q": "S", "R": "S2"} and m.folded == {"T": "S"}, m


@check("map: ties go to label order then L, S, S2; unknown words and unlabeled words are ignored")
def _():
    assert mapped([("P", {"L": 5, "S": 5}), ("Q", {"L": 5, "S": 5})]).speaker_map == {"P": "L", "Q": "S"}
    m = bo.map_labels(["P", "P", None, "Q", "Q"], ["L", "L", "S", None, "?"])
    assert m.speaker_map == {"P": "L", "Q": "L"} and m.folded == {"Q": "L"}, m
    assert mapped([("P", {"S": 3, "S2": 3})]).speaker_map == {"P": "S"}


# ------------------------------------------------------------- end to end ---

@check("the synthetic gold renders as designed: 4 boundaries, 9 lines, 29 words, truth as tabled")
def _():
    td, d, lab = workdir()
    with td:
        g = bo.load_gold(lab, sd.parse_speaker_map(GOLD_MAP), 3)
        assert [b["verdict"] for b in g.labels["boundaries"]] == ["real", "real", "spurious", "spurious"]
        assert len(g.stream.index) == 29 and g.truth == TRUTH, g.truth


@check("identity: the stored turns through the bake-off path reproduce score_diarization exactly")
def _():
    td, d, lab = workdir()
    with td:
        g = bo.load_gold(lab, sd.parse_speaker_map(GOLD_MAP), 3)
        r = bo.identity_check(g, 3)
        assert r["bakeoff"]["speaker_map"] == {A: "L", B: "S"}
        assert r["words"]["student_as_instructor"] == 8 and r["words"]["student_words"] == 11
        assert r["bakeoff"]["high_harm_runs"] == 2 and r["words"]["accuracy"] == round(18 / 29, 4)


@check("identity: a gold whose verdicts break the convention is refused")
def _():
    td, d, lab = workdir()
    with td:
        labels = json.loads(lab.read_text())
        labels["boundaries"][2].update(verdict="real", who="L")      # B-003: real, but no change there
        lab.write_text(json.dumps(labels))
        g = bo.load_gold(lab, sd.parse_speaker_map(GOLD_MAP), 3)
        raises(bo.identity_check, g, 3, match="identity check failed")


@check("swap: a copy with only the turns replaced; the original record is untouched")
def _():
    td, d, lab = workdir()
    with td:
        before = bo.sha256_file(d / "rec_raw.json")
        rc, out, err = score_cli(lab, write_arm(d, "good", "run1", "rttm", GOOD),
                                 "--gold-map", GOLD_MAP, "--out-dir", d / "scored")
        assert rc == 0, err
        assert bo.sha256_file(d / "rec_raw.json") == before
        orig = json.loads((d / "rec_raw.json").read_text())
        copy = json.loads((d / "scored" / "rec_bakeoff-good-run1_raw.json").read_text())
        assert copy["words"] == orig["words"] and copy["text"] == orig["text"]
        assert copy["speaker_turns"] == turns_for(GOOD) != orig["speaker_turns"]
        assert copy["diarization"]["bakeoff"]["arm"] == "good" and "bake-off copy" in copy["warnings"][-1]
        assert bo.schema.validate(copy) == []


@check("scoring: three arms, the numbers worked by hand, and the bar")
def _():
    td, d, lab = workdir()
    with td:
        provs = [write_arm(d, "community1", "run1", "turns-json", [r[0] for r in RECORD]),
                 write_arm(d, "good", "run1", "nemo-segments", GOOD),
                 write_arm(d, "oversplit", "run1", "rttm", OVERSPLIT)]
        rc, out, err = score_cli(lab, *provs, "--gold-map", GOLD_MAP, "--json", d / "r.json")
        assert rc == 0, err
        r = json.loads((d / "r.json").read_text())
        stored, rerun = r["arms"]["community1@stored"], r["arms"]["community1/run1"]
        assert bo.compare({k: v for k, v in stored.items() if k != "arm"},
                          {k: v for k, v in rerun.items() if k != "arm"}) == []
        good, over = r["arms"]["good/run1"], r["arms"]["oversplit/run1"]
        # good: every change at a boundary, every word right.
        assert good["bakeoff"]["speaker_map"] == {"speaker_0": "L", "speaker_1": "S", "speaker_2": "S2"}
        assert (good["words"]["student_as_instructor"], good["words"]["accuracy"]) == (0, 1.0)
        assert (good["boundaries"]["total"], good["boundaries"]["real"], good["missed_changes"]["count"]) == (6, 6, 0)
        # oversplit: V pairs with S; B and W are left over and fold into S and L.
        b = over["bakeoff"]
        assert b["paired"] == {A: "L", "V": "S"} and b["folded"] == {B: "S", "W": "L"}, b
        assert b["words_in_folded_labels"] == 6 and b["labels_rendered"] == 4
        assert over["words"]["student_as_instructor"] == 3 and b["high_harm_runs"] == 1
        assert b["pessimistic_fold"]["student_as_instructor"] == 6 and b["pessimistic_fold"]["high_harm_runs"] == 2
        assert (over["boundaries"]["total"], over["boundaries"]["real"], over["boundaries"]["spurious"]) == (6, 4, 2)
        assert (over["missed_changes"]["count"], over["missed_changes"]["words"]) == (2, 3)
        assert over["words"]["accuracy"] == round(26 / 29, 4)
        # Audit list: oversplit's two spurious boundaries (rows 6 and 7) sit exactly where
        # community-1's B-003 and B-004 do, so the gold judged them: neither is listed.
        assert b["unanchored_spurious"]["count"] == 0
        assert stored["bakeoff"]["unanchored_spurious"]["count"] == 0
        # The bar against community-1 (8 words, 2 runs, 62.07%): at most 4 words, 1 run, 61.07%.
        v = r["verdict"]
        assert v["per_arm"]["good"]["pass"] and not v["per_arm"]["oversplit"]["pass"]
        assert {k: c["pass"] for k, c in v["per_arm"]["oversplit"]["checks"].items()} == {
            "P1_primary": True, "P2_primary_pessimistic_fold": False, "P3_high_harm_runs": True,
            "G1_accuracy": True}
        assert v["per_arm"]["good"]["checks"]["P1_primary"]["limit"] == 4
        assert v["decision"] == "switch candidate: good", v["decision"]
        assert "decision: switch candidate: good" in out


@check("audit list: a spurious boundary more than N words from every community-1 boundary")
def _():
    stream = sd.Stream(index=list(range(10)), labels=["A"] * 2 + ["B"] * 6 + ["C"] * 2,
                       line_of=["T0001"] * 10, start=[float(k) for k in range(10)], offset=0.0,
                       boundary_at={2: "B-001", 8: "B-002"})
    labels = {"boundaries": [{"word_i": 2, "verdict": "spurious", "to": "B"},
                             {"word_i": 8, "verdict": "spurious", "to": "C"}]}
    items = bo.unanchored_spurious(stream, labels, anchors=[1], near=3)
    assert [(x["word_i"], x["label"], x["words_until_next_boundary"]) for x in items] == [(8, "C", 2)], items
    labels["boundaries"][1]["verdict"] = "real"
    assert bo.unanchored_spurious(stream, labels, anchors=[1], near=3) == []


@check("determinism: identical runs are one result; differing runs are scored at the worse")
def _():
    td, d, lab = workdir()
    with td:
        provs = [write_arm(d, "good", "run1", "rttm", GOOD), write_arm(d, "good", "run2", "turns-json", GOOD),
                 write_arm(d, "oversplit", "run1", "rttm", OVERSPLIT),
                 write_arm(d, "oversplit", "run2", "rttm", [r[0] for r in RECORD])]
        rc, out, err = score_cli(lab, *provs, "--gold-map", GOLD_MAP, "--json", d / "r.json")
        assert rc == 0, err
        r = json.loads((d / "r.json").read_text())
        assert r["determinism"]["good"]["identical"] is True
        assert r["determinism"]["oversplit"] == {"runs": ["oversplit/run1", "oversplit/run2"],
                                                 "identical": False, "scored": "oversplit/run2"}


@check("refused: the wrong audio, an output changed after its provenance, a malformed output, a repeat")
def _():
    td, d, lab = workdir()
    with td:
        rc, _, err = score_cli(lab, write_arm(d, "a", "run1", "rttm", GOOD, audio_sha="ef" * 32),
                               "--gold-map", GOLD_MAP)
        assert rc == 2 and "not the gold's" in err, err
        p = write_arm(d, "b", "run1", "rttm", GOOD)
        with open(p.parent / "out.rttm", "a") as fh:
            fh.write("SPEAKER rec 1 99.000 1.000 <NA> <NA> X <NA> <NA>\n")
        rc, _, err = score_cli(lab, p, "--gold-map", GOLD_MAP)
        assert rc == 2 and "sha256 differs" in err, err
        p = write_arm(d, "c", "run1", "rttm", GOOD)
        (p.parent / "out.rttm").write_text("SPEAKER rec 1 0.5\n")
        prov = json.loads(p.read_text()); prov["native_output"]["sha256"] = bo.sha256_file(p.parent / "out.rttm")
        p.write_text(json.dumps(prov))
        rc, _, err = score_cli(lab, p, "--gold-map", GOLD_MAP)
        assert rc == 2 and "10 fields" in err, err
        q = write_arm(d, "e", "run1", "rttm", GOOD)
        rc, _, err = score_cli(lab, q, q, "--gold-map", GOLD_MAP)
        assert rc == 2 and "given twice" in err, err


@check("output: a JSON of counts and labels, with no transcript text in it")
def _():
    td, d, lab = workdir()
    with td:
        rc, out, err = score_cli(lab, write_arm(d, "good", "run1", "rttm", GOOD),
                                 write_arm(d, "oversplit", "run1", "rttm", OVERSPLIT),
                                 "--gold-map", GOLD_MAP, "--json", d / "r.json")
        assert rc == 0, err
        blob = ((d / "r.json").read_text() + out).lower()
        for tok in ("memory", "exam", "sleep", "dreams", "questions", "ahead"):
            assert not re.search(rf"\b{tok}\b", blob), f"transcript text {tok!r} leaked"


@check("the bar's constants are the pre-registered ones")
def _():
    assert bo.BAR == {"primary_max_share_of_baseline": 0.5, "high_harm_runs_max_share_of_baseline": 0.5,
                      "accuracy_max_drop": 0.010, "close_call_words": 5}


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
