#!/usr/bin/env python3
"""Checks for scripts/diarization_candidates.py and scripts/score_candidates.py. Zero dependencies.

    python3 tests/check_candidates.py

Every candidate class of docs/diarization_candidates_plan.md (§2) on a
synthetic record built here in code, firing where the rule says and not where
it does not, with the exact window; determinism; the CLI; and the recall
scorer end to end (sheet generated, filled, checked, candidates generated,
scored), with every number worked out by hand from the table in SCORED.

Then v2 (docs/diarization_candidates_v2_plan.md): v1 frozen byte for byte
(hashes taken from PR #29's code before v2 was written), v2's registered
parameter hash, each new or changed class firing and not firing with its exact
window, the pairing, determinism, the CLI, and both versions scored on a
record worked out by hand in SCORED_V2.
"""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import diarization_candidates as dc  # noqa: E402
import diarization_labels as dl  # noqa: E402
import score_candidates as sc  # noqa: E402
import score_diarization as sd  # noqa: E402

GEN = ROOT / "scripts" / "diarization_candidates.py"
SCORE = ROOT / "scripts" / "score_candidates.py"
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


# ------------------------------------------------------------- records ------
#
# Words are 0.3 s long and abut (gap 0) unless a segment says to pause first.
# Each segment with a label becomes one raw turn covering exactly its words;
# ``extra`` adds raw turns. The stream is the record's primary, in intended mode.

A, B = "SPEAKER_00", "SPEAKER_01"


def t0(k):
    """Start of word k in a record with no pauses."""
    return round(1.0 + 0.3 * k, 2)


def t1(k):
    return round(1.3 + 0.3 * k, 2)


def make_doc(segs, extra=()):
    words, turns, t, i = [], [], 1.0, 0
    for seg in segs:
        label, text = seg[0], seg[1]
        t += seg[2] if len(seg) > 2 else 0.0
        s0 = t
        for tok in text.split():
            words.append({"i": i, "text": tok, "start": round(t, 2), "end": round(t + 0.3, 2),
                          "timing_source": "native", "speaker": None, "speaker_source": None,
                          "conf": None, "flags": []})
            i += 1; t += 0.3
        if label:
            turns.append({"start": round(s0, 2), "end": round(t, 2), "speaker": label})
    turns += [{"start": s, "end": e, "speaker": lab} for s, e, lab in extra]
    return {
        "schema_version": "1.0",
        "source": {"audio_path": "data/synthetic.wav", "audio_sha256": "ef" * 32, "duration_s": t + 1.0},
        "run": {"created_utc": "2026-10-02T00:00:00Z", "device": "cpu",
                "pipeline_version": {"commit": "0" * 40, "branch": "main", "dirty": False}},
        "asr": {"model_id": "mock", "revision": "0", "capabilities": {}, "granularity": "word",
                "params": {"mode": "intended"}, "performance": {}},
        "diarization": {"model_id": "mock-diarizer", "revision": "0", "params": {}},
        "words": words, "text": " ".join(w["text"] for w in words),
        "speaker_turns": turns, "errors": [], "warnings": [],
    }


def got(doc, cls):
    """(first_word_i, last_word_i, at_word_i) of every candidate of one class."""
    return [(c["first_word_i"], c["last_word_i"], c["at_word_i"])
            for c in dc.generate(doc)["candidates"] if c["class"] == cls]


def ev(doc, cls):
    return [c["evidence"] for c in dc.generate(doc)["candidates"] if c["class"] == cls]


def words(n, stem="w"):
    return " ".join(f"{stem}{k}" for k in range(n))


# ------------------------------------------------------------ the rules -----

@check("token: lowercased, curly apostrophes straightened, edge punctuation stripped")
def _():
    assert dc.token("Yeah.") == "yeah" and dc.token("“I’m") == "i'm" and dc.token("Mm-hmm,") == "mm-hmm"
    assert dc.token("you?)") == "you" and dc.token("...") == ""


@check("the parameters are the pre-registered ones, hashed canonically")
def _():
    p = dc.default_params()
    assert (p["near_words"], p["zero_gap_s"], p["island_max_words"], p["backchannel_max_words"],
            p["address_max_words"], p["register_window_words"], p["register_min_words"],
            p["register_onset"]) == (3, 0.005, 15, 3, 10, 20, 5, 2)
    assert {"yeah", "okay", "right"} <= set(p["backchannel"]) and "so" not in p["answer_openers"]
    assert p["solicit_bigrams"] == [["go", "ahead"]]
    assert dc.params_sha256(p) == dc.params_sha256(json.loads(json.dumps(p)))


# ---------------------------------------------------------------- shift -----

@check("shift: every boundary p gets [p-N-1, p+N], clamped at the stream's start")
def _():
    d = make_doc([(A, words(8, "a")), (B, words(8, "b"))])
    assert got(d, "shift") == [(4, 11, 8)]
    d = make_doc([(A, "a0 a1"), (B, words(6, "b"))])
    assert got(d, "shift") == [(0, 5, 2)]


# ---------------------------------------------------------------- split -----

@check("boundary cues: lowercase after no punctuation, zero gap, and inside a sentence")
def _():
    d = make_doc([(A, words(8, "a")), (B, words(8, "b"))])
    for cls in ("no_punct_lowercase", "zero_gap", "inside_sentence"):
        assert got(d, cls) == [(7, 8, 8)], cls


@check("boundary cues: terminal punctuation, a capital, a pause each switch off the right ones")
def _():
    d = make_doc([(A, "We cover memory."), (B, "Then sleep comes.")])          # terminal, gap 0
    assert (got(d, "no_punct_lowercase"), got(d, "zero_gap"), got(d, "inside_sentence")) == ([], [(2, 3, 3)], [])
    d = make_doc([(A, "we cover memory and"), (B, "then sleep", 1.0)])           # a pause
    assert (got(d, "no_punct_lowercase"), got(d, "zero_gap"), got(d, "inside_sentence")) == ([(3, 4, 4)], [], [])
    d = make_doc([(A, "we cover memory and"), (B, "Then sleep", 0.3)])           # 0.3 s, a capital
    assert (got(d, "no_punct_lowercase"), got(d, "zero_gap"), got(d, "inside_sentence")) == ([], [], [(3, 4, 4)])


@check("long_island: three words between two of the other label, no pause; 15 fires and 16 does not")
def _():
    d = make_doc([(A, "a0 a1 a2 a3"), (B, "b0 b1 b2"), (A, "a4 a5 a6 a7")])
    assert got(d, "long_island") == [(3, 7, 4)] and got(d, "edge_island") == []
    assert ev(d, "long_island")[0]["words"] == 3 and ev(d, "long_island")[0]["assigned_words"] == 3
    d = make_doc([(A, "a0 a1 a2 a3"), (B, words(15, "b")), (A, "a4 a5 a6 a7")])
    assert got(d, "long_island") == [(3, 19, 4)]
    d = make_doc([(A, "a0 a1 a2 a3"), (B, words(16, "b")), (A, "a4 a5 a6 a7")])
    assert got(d, "long_island") == []


@check("edge_island: an island beside a pause, which smoothing leaves alone")
def _():
    d = make_doc([(A, "a0 a1 a2 a3"), (B, "b0 b1", 1.0), (A, "a4 a5 a6 a7")])
    assert got(d, "edge_island") == [(3, 6, 4)] and got(d, "long_island") == []
    assert ev(d, "edge_island")[0]["gap_before_s"] == 1.0


# ---------------------------------------------------------- dropped raw -----

@check("dropped_raw_change: a raw turn outvoted on every word, far from its label: [a-N-1, b+N+1]")
def _():
    d = make_doc([(A, words(12, "a"))], extra=[(t0(5), t1(7), B)])
    assert got(d, "dropped_raw_change") == [(1, 11, 5)]
    e = ev(d, "dropped_raw_change")[0]
    assert (e["raw_label"], e["rendered"], e["words"], e["kind"]) == (B, [A], 3, "outvoted")


@check("dropped_raw_change: displaced more than N words from the rendered boundary fires; within N it does not")
def _():
    d = make_doc([(None, words(20))], extra=[(t0(0), t1(14), A), (t0(6), t1(19), B)])
    assert got(d, "shift") == [(11, 18, 15)]                  # the render switches at 15
    assert got(d, "dropped_raw_change") == [(2, 18, 6)]
    assert ev(d, "dropped_raw_change")[0]["kind"] == "displaced"
    d = make_doc([(None, words(20))], extra=[(t0(0), t1(11), A), (t0(10), t1(19), B)])
    assert got(d, "dropped_raw_change") == []                 # 2 words off: shift holds it


@check("dropped_raw_change: an island smoothing flipped is a dropped change of kind 'smoothed'")
def _():
    d = make_doc([(A, "a0 a1 a2 a3"), (B, "b0 b1"), (A, "a4 a5 a6 a7")])
    assert got(d, "long_island") == [] and got(d, "shift") == []
    assert got(d, "dropped_raw_change") == [(0, 9, 4)]
    assert ev(d, "dropped_raw_change")[0]["kind"] == "smoothed"


@check("dropped_raw_change: a raw run over silence fires at the next word, unless its label is within N")
def _():
    d = make_doc([(A, words(8, "a")), (A, words(8, "c"), 2.0)], extra=[(3.8, 4.6, B)])
    assert got(d, "dropped_raw_change") == [(4, 11, 8)]
    assert ev(d, "dropped_raw_change")[0]["kind"] == "empty"
    d = make_doc([(A, words(6, "a")), (B, words(4, "b")), (A, words(4, "c"), 2.0)], extra=[(4.6, 5.4, B)])
    assert [g for g in got(d, "dropped_raw_change")] == []


@check("dropped_raw_change agrees with the labeling sheet's raw-turn accounting on its record")
def _():
    # The labeling sheet's checks record: one island smoothing flips back, one
    # turn over silence, one turn every word of which a longer turn outvotes.
    intended = [("So", 0.5, 0.8), ("today", 0.8, 1.2), ("we", 1.2, 1.4), ("cover", 1.4, 1.8),
                ("memory.", 1.8, 2.4), ("Any", 3.0, 3.3), ("questions?", 3.3, 3.9),
                ("Is", 10.5, 10.7), ("this", 10.7, 10.9), ("on", 10.9, 11.0), ("the", 11.0, 11.1),
                ("exam?", 11.1, 11.6), ("Yes", 13.5, 13.8), ("it", 13.8, 13.9), ("is.", 13.9, 14.2),
                ("Now", 15.0, 15.2), ("the", 15.2, 15.3), ("hippocampus", 15.3, 16.0), ("matters.", 16.0, 16.5),
                ("Okay,", 26.2, 26.4), ("okay.", 26.4, 26.6), ("Can", 27.2, 27.4), ("you", 27.4, 27.5),
                ("repeat", 27.5, 27.8), ("that?", 27.8, 28.1), ("Sure,", 28.7, 29.0), ("the", 29.0, 29.1),
                ("hippocampus", 29.1, 29.8), ("again.", 29.8, 30.3)]
    turns = [(0.0, 10.0, A), (10.4, 13.2, B), (13.4, 15.2, A), (15.2, 15.3, B), (15.3, 20.0, A),
             (25.0, 25.5, B), (26.0, 31.0, A), (28.8, 29.2, B)]
    d = make_doc([])
    d["words"] = [{"i": i, "text": x, "start": s, "end": e, "timing_source": "native", "speaker": None,
                   "speaker_source": None, "conf": None, "flags": []} for i, (x, s, e) in enumerate(intended)]
    d["speaker_turns"] = [{"start": s, "end": e, "speaker": lab} for s, e, lab in turns]
    kinds = sorted(e["kind"] for e in ev(d, "dropped_raw_change"))
    assert kinds == ["empty", "outvoted", "smoothed"], kinds
    r = dc.render(d)
    acct = dl.raw_accounting(r.words, d["speaker_turns"])
    assert (acct["empty"], acct["outvoted"]) == (1, 1), acct
    assert got(d, "dropped_raw_change")[2] == (21, 28, 25)          # Sure, the, hippocampus: 25-27; clamped


# ---------------------------------------------------------------- merge -----

@check("question_answer: a question, then a sentence opening like an answer, in one label: [first(Q)-1, first(R)]")
def _():
    d = make_doc([(A, "So today we cover memory. Is this on the exam? Yes it is.")])
    assert got(d, "question_answer") == [(4, 10, 10)] and ev(d, "question_answer")[0]["opener"] == "yes"
    assert got(make_doc([(A, "So today we cover memory. Is this on the exam? So it is.")]), "question_answer") == []
    d = make_doc([(A, "So today we cover memory. Is this on the exam?"), (B, "Yes it is.")])
    assert got(d, "question_answer") == []                 # the answer is across a boundary


@check("backchannel: one to three backchannel tokens as a sentence, mid-run only")
def _():
    assert got(make_doc([(A, "We cover memory. Yeah. And then sleep.")]), "backchannel") == [(2, 4, 3)]
    assert got(make_doc([(A, "We cover memory. Okay okay. And then sleep.")]), "backchannel") == [(2, 5, 3)]
    assert got(make_doc([(A, "Yeah. We cover memory.")]), "backchannel") == []
    assert got(make_doc([(A, "We cover memory. Yeah right okay sure. And then sleep.")]), "backchannel") == []
    assert got(make_doc([(A, "We cover memory. Yeah maybe. And then sleep.")]), "backchannel") == []


@check("address_reply: a short address with you, a solicitation or 'go ahead', then a reply opener: [last(A), first(R)]")
def _():
    d = make_doc([(A, "Now the next part. Go ahead. Yeah so my question is about sleep.")])
    assert got(d, "address_reply") == [(5, 6, 6)] and ev(d, "address_reply")[0]["cue"] == "go ahead"
    assert got(d, "question_answer") == []
    d = make_doc([(A, "What do you think? I think it is memory.")])
    assert got(d, "address_reply") == [(3, 4, 4)] and got(d, "question_answer") == [(0, 4, 4)]
    assert got(make_doc([(A, "You see this here. So we go on.")]), "address_reply") == []
    long_ = "What do you think about all of this stuff we covered today? Yes."
    assert got(make_doc([(A, long_ + " More.")]), "address_reply") == []


@check("register_change: a first-person onset after twenty words with none (and up to a window early)")
def _():
    lecture = "The hippocampus binds features. Then the cortex stores them over weeks. Sleep helps that process a lot here."
    d = make_doc([(A, lecture + " I think my question is about the thing I saw.")])
    # The 20 words read from a sentence start reach into the next sentences, so
    # the sentence before the onset fires too: the rule as registered.
    assert got(d, "register_change") == [(10, 11, 11), (17, 18, 18)]
    e = ev(d, "register_change")[1]
    assert (e["first_person_before"], e["first_person_after"], e["before_words"]) == (0, 3, 18)
    assert got(make_doc([(A, lecture.replace("Then", "I") + " I think my question is about it.")]),
               "register_change") == []
    assert got(make_doc([(A, lecture + " I wonder about the thing we saw.")]), "register_change") == []


# ---------------------------------------------------------------- sites -----

@check("sites: overlapping windows merge into one judgment; words covered is their union")
def _():
    res = dc.generate(make_doc([(A, words(8, "a")), (B, words(8, "b"))]))
    assert [s["classes"] for s in res["sites"]] == [["shift", "no_punct_lowercase", "zero_gap", "inside_sentence"]]
    assert (res["sites"][0]["first_word_i"], res["sites"][0]["last_word_i"]) == (4, 11)
    assert res["counts"]["words_covered"] == 8 and res["counts"]["candidates"] == 4
    res = dc.generate(make_doc([(A, words(10, "a")), (B, words(16, "b")), (A, words(10, "c"))]))
    assert [(s["first_word_i"], s["last_word_i"]) for s in res["sites"]] == [(6, 13), (22, 29)]
    res = dc.generate(make_doc([(A, words(10, "a")), (B, words(10, "b")), (A, words(10, "c"))]))
    assert [(s["first_word_i"], s["last_word_i"]) for s in res["sites"]] == [(6, 23)]   # an island joins them


# ---------------------------------------------------------- determinism -----

def busy_doc():
    lecture = "The hippocampus binds features. Then the cortex stores them over weeks. Sleep helps that process a lot here."
    return make_doc([(A, "So today we cover memory. Is this on the exam? Yes it is. We go on. Yeah. Next"),
                     (B, "and then we", 0.2), (A, lecture + " I think my question is about the thing I saw."),
                     (B, "Go ahead.", 1.0), (A, "Okay so my point is this. Any questions? Yeah I have one.")],
                    extra=[(t0(3), t1(5), B), (40.0, 40.5, B)])


@check("deterministic: the same record twice gives the same output, ids in window order")
def _():
    a, b = dc.generate(busy_doc()), dc.generate(busy_doc())
    assert json.dumps(a) == json.dumps(b)
    assert [c["id"] for c in a["candidates"]] == [f"C-{k:04d}" for k in range(1, len(a["candidates"]) + 1)]
    keys = [(c["first_word_i"], c["last_word_i"], dc.CLASSES.index(c["class"])) for c in a["candidates"]]
    assert keys == sorted(keys)
    assert len({c["class"] for c in a["candidates"]}) >= 10, sorted({c["class"] for c in a["candidates"]})


@check("CLI: exit 0, byte-identical output on a second run, source and parameters hashed, no transcript text")
def _():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "rec_raw.json").write_text(json.dumps(busy_doc()))
        r1 = subprocess.run([sys.executable, str(GEN), str(d / "rec_raw.json")], capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        out = d / "rec_speaker_candidates.json"
        first = out.read_bytes()
        r2 = subprocess.run([sys.executable, str(GEN), str(d / "rec_raw.json")], capture_output=True, text=True)
        assert r2.returncode == 0 and out.read_bytes() == first
        res = json.loads(first)
        assert res["source"] == "rec_raw.json" and res["source_sha256"] == dl.sha256_file(d / "rec_raw.json")
        assert res["params_sha256"] == dc.params_sha256(res["params"]) and res["params"] == dc.default_params()
        assert "candidates:" in r1.stdout and "sites:" in r1.stdout
        blob = first.decode().lower()
        for tok in ("hippocampus", "cortex", "exam", "features"):
            assert tok not in blob, f"transcript text {tok!r} leaked into the candidates"


@check("CLI refuses a missing or invalid record (exit 2)")
def _():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = subprocess.run([sys.executable, str(GEN), str(d / "nope_raw.json")], capture_output=True, text=True)
        assert r.returncode == 2 and "refused" in r.stderr
        (d / "bad_raw.json").write_text(json.dumps({"schema_version": "0"}))
        r = subprocess.run([sys.executable, str(GEN), str(d / "bad_raw.json")], capture_output=True, text=True)
        assert r.returncode == 2 and "refused" in r.stderr


# --------------------------------------------------------------- scoring ----
#
# One sentence per row, a 1 s pause between rows. Row 2 is a student's question
# inside the instructor's run, with a raw SPEAKER_01 turn over it that the
# instructor's equal turn outvotes; row 4 is a spurious island; row 8 holds a
# student's words mid-sentence that no class can see. Positions: rows start at
# 0, 4, 8, 13, 16, 19, 22, 25, 28, 33; 37 words.
#
#   missed changes: 8 (L->S, 5 words, high harm), 13 (S->L, return),
#                   29 (L->S, 4 words, high harm), 33 (S->L, return)
#   spurious:       16 (B-001), 19 (B-002)
#   windows:        shift [12,19] and [15,22]; edge_island [15,19];
#                   dropped_raw_change [4,16]; question_answer [7,13]

MAP = {A: "L", B: "S"}
SCORED = [
    (A, "Today we study memory.", None),
    (A, "Next the hippocampus matters.", None),
    (A, "Is it on the exam?", ">> change: S"),
    (A, "Yes it is.", ">> back: L"),
    (B, "And then sleep.", ("spurious", "")),
    (A, "Sleep helps consolidation.", ("spurious", "")),
    (A, "Moving on now.", None),
    (A, "This part matters.", None),
    (A, "Okay I am lost here.", '>> change: S "I am"'),
    (A, "Right we stop here.", ">> back: L"),
]


def scored_doc(record=None, duration_s=None):
    words, turns, t, i = [], [], 0.5, 0
    for label, text, _ in (record or SCORED):
        s0 = t
        for tok in text.split():
            words.append({"i": i, "text": tok, "start": round(t, 2), "end": round(t + 0.3, 2),
                          "timing_source": "native", "speaker": None, "speaker_source": None,
                          "conf": None, "flags": []})
            i += 1; t += 0.3
        turns.append({"start": round(s0, 2), "end": round(t, 2), "speaker": label})
        t += 1.0
    if record is None:
        turns.append(dict(turns[2], speaker=B))       # outvoted: an equal turn sorted after A's
    doc = make_doc([])
    doc.update(words=words, text=" ".join(w["text"] for w in words), speaker_turns=turns)
    if duration_s is not None:
        doc["source"]["duration_s"] = duration_s
    return doc


def fill(sheet, record=None):
    record = record or SCORED
    out, lines = [], sheet.read_text().splitlines()
    for n, ln in enumerate(lines):
        if ln.startswith("T0"):
            row = record[int(ln[1:5]) - 1]
            if isinstance(row[2], str):
                out.append(row[2])
        elif ln.startswith(("boundary:", "who:")):
            below = next(x for x in lines[n:] if x.startswith("T0"))
            verdict, who = record[int(below[1:5]) - 1][2]
            ln = f"boundary: {verdict}" if ln.startswith("boundary:") else f"who: {who}".rstrip()
        out.append(ln)
    sheet.write_text("\n".join(out) + "\n")


def scored_workdir(record=None, duration_s=None, versions=(1,)):
    td = tempfile.TemporaryDirectory()
    d = Path(td.name)
    (d / "rec_raw.json").write_text(json.dumps(scored_doc(record, duration_s), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        assert dl.main([str(d / "rec_raw.json")]) == 0
        fill(d / "rec_labels.md", record)
        assert dl.main(["--check", str(d / "rec_labels.md")]) == 0, "the synthetic sheet does not check clean"
        for v in versions:
            assert dc.main([str(d / "rec_raw.json"), "--version", str(v)]) == 0
    return td, d


def evaluated(d, name="rec_speaker_candidates.json", record=None, duration_s=None):
    labels = json.loads((d / "rec_labels.json").read_text())
    cdoc = json.loads((d / name).read_text())
    stream = sd.rendered_stream(labels, d / "rec_labels.json")
    truth = sd.derive_truth(labels, stream, MAP)
    return sc.evaluate(labels, stream, truth, MAP, cdoc, dc.render(scored_doc(record, duration_s)))


@check("surfacing rule: window [a, b] holds transition t iff a < t <= b")
def _():
    assert [sc.surfaces(3, 5, t) for t in (2, 3, 4, 5, 6)] == [False, False, True, True, False]


@check("scoring: the synthetic record's windows are the hand-worked ones")
def _():
    td, d = scored_workdir()
    with td:
        cdoc = json.loads((d / "rec_speaker_candidates.json").read_text())
        win = sorted((c["class"], c["first_word_i"], c["last_word_i"]) for c in cdoc["candidates"])
        assert win == sorted([("shift", 12, 19), ("shift", 15, 22), ("edge_island", 15, 19),
                              ("dropped_raw_change", 4, 16), ("question_answer", 7, 13)]), win


@check("scoring: recall overall 4/6, missed 2/4, misattributing 1/2, spurious 2/2, high harm 1/2")
def _():
    td, d = scored_workdir()
    with td:
        r = evaluated(d)
        assert {k: (v["surfaced"], v["total"]) for k, v in r["recall"].items()} == {
            "overall": (4, 6), "missed": (2, 4), "missed_misattributing": (1, 2),
            "spurious": (2, 2), "high_harm": (1, 2)}, r["recall"]


@check("scoring: per class, what each surfaces and its hit share")
def _():
    td, d = scored_workdir()
    with td:
        pc = evaluated(d)["per_class"]
        k = sc.KINDS
        assert pc["question_answer"]["surfaced"] == dict(zip(k, (2, 1, 0, 1)))
        assert pc["dropped_raw_change"]["surfaced"] == dict(zip(k, (2, 1, 1, 1)))
        assert pc["shift"]["surfaced"] == dict(zip(k, (1, 0, 2, 0)))
        assert pc["edge_island"]["surfaced"] == dict(zip(k, (0, 0, 2, 0)))
        assert all(v["only_this"] == dict(zip(k, (0, 0, 0, 0))) for v in pc.values())
        assert pc["shift"]["windows_with_error_share"] == 1.0 and pc["backchannel"]["candidates"] == 0


@check("scoring: each high-harm miss, its return, the unsurfaced misses and the bar")
def _():
    td, d = scored_workdir()
    with td:
        r = evaluated(d)
        assert [(h["word_i"], h["surfaced"], h["surfaced_by"], h["return_surfaced_by"], h["run_inside_one_site"])
                for h in r["high_harm"]] == [
            (8, True, ["dropped_raw_change", "question_answer"],
             ["shift", "dropped_raw_change", "question_answer"], True),
            (29, False, [], [], False)]
        assert [(u["word_i"], u["direction"], u["sentence_start"]) for u in r["unsurfaced_misses"]] == \
               [(29, "student_as_instructor", False), (33, None, True)]
        assert r["volume"]["sites"] == 1 and r["volume"]["words_covered"] == 19
        assert r["volume"]["real_boundaries"] == 0
        assert r["bar"]["high_harm_recall"]["pass"] is False and r["bar"]["pass"] is False


@check("score CLI: exit 0 with the bar line; no transcript text in the JSON")
def _():
    td, d = scored_workdir()
    with td:
        out = d / "cscore.json"
        r = subprocess.run([sys.executable, str(SCORE), str(d / "rec_labels.json"),
                            str(d / "rec_speaker_candidates.json"), "--speaker-map", f"{A}=L,{B}=S",
                            "--json", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "high_harm 1/2" in r.stdout and "bar: FAIL" in r.stdout, r.stdout
        blob = out.read_text().lower()
        for tok in ("hippocampus", "exam", "lost", "consolidation"):
            assert tok not in blob, f"transcript text {tok!r} leaked into the score"


@check("score CLI refuses candidates from another source, render or boundary set")
def _():
    td, d = scored_workdir()
    with td:
        cpath = d / "rec_speaker_candidates.json"
        good = cpath.read_text()
        args = [sys.executable, str(SCORE), str(d / "rec_labels.json"), str(cpath), "--speaker-map", f"{A}=L,{B}=S"]
        for mutate, msg in ((lambda c: c.update(source_sha256="0" * 64), "different source"),
                            (lambda c: c["render"].update(max_island=3), "render parameters"),
                            (lambda c: c["boundaries"].pop(), "boundaries"),
                            (lambda c: c["params"].update(near_words=4), "params_sha256")):
            c = json.loads(good); mutate(c); cpath.write_text(json.dumps(c))
            r = subprocess.run(args, capture_output=True, text=True)
            assert r.returncode == 2 and msg in r.stderr, (msg, r.stderr)


# ============================================================ v1 frozen =====
#
# v1 must stay byte-identical to what PR #29 measured. These hashes were taken
# from PR #29's code, before v2 was written: the parameters' and the CLI's
# whole output on two synthetic records.

V1_PARAMS_SHA = "fd44dbfe718e24b8dd9b6c8afbc0b1ed1c0a26c453228148d1d26a2d702730b6"
V1_OUTPUT_SHA = {"busy": "13112f7774e380d634584a1b42725f42bbdf70efec53f599a5068491e0565266",
                 "scored": "c306b95912c0025d8f0069c78c701198fec68a2547b06b6d3b1fd1682ec0ba31"}
V2_PARAMS_SHA = "0c77f39da15c71b6ac5c51a75ea5aaa3dda8aa8edf76be9035a6ce58dc267fa6"   # v2 plan §3, A1


def cli_output(doc, *args, name="rec_speaker_candidates.json"):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "rec_raw.json").write_text(json.dumps(doc))
        r = subprocess.run([sys.executable, str(GEN), str(d / "rec_raw.json"), *args],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return (d / name).read_bytes(), sorted(x.name for x in d.iterdir()), r.stdout


@check("v1 frozen: its parameters hash to PR #29's, and the default version is still v1")
def _():
    assert dc.params_sha256(dc.default_params()) == dc.params_sha256(dc.default_params(1)) == V1_PARAMS_SHA
    assert dc.DEFAULT_VERSION == 1 and "version" not in dc.default_params(1)
    assert dc.default_params(1)["register_window_words"] == 20 and "fillers" not in dc.default_params(1)


@check("v1 frozen: the CLI's output is byte-identical to PR #29's, with and without --version 1")
def _():
    for key, doc in (("busy", busy_doc()), ("scored", scored_doc())):
        for args in ((), ("--version", "1")):
            out, files, _ = cli_output(doc, *args)
            assert hashlib.sha256(out).hexdigest() == V1_OUTPUT_SHA[key], (key, args)
            assert "rec_speaker_candidates_v2.json" not in files
        res = json.loads(out)
        assert "version" not in res and res["plan"] == "docs/diarization_candidates_plan.md"


# ================================================================= v2 =======

def got2(doc, cls):
    return [(c["first_word_i"], c["last_word_i"], c["at_word_i"])
            for c in dc.generate(doc, version=2)["candidates"] if c["class"] == cls]


def ev2(doc, cls):
    return [c["evidence"] for c in dc.generate(doc, version=2)["candidates"] if c["class"] == cls]


@check("v2 parameters: the registered hash, v1's values for what it keeps, the revisions' values")
def _():
    p, p1 = dc.default_params(2), dc.default_params(1)
    assert dc.params_sha256(p) == V2_PARAMS_SHA
    assert p["version"] == 2 and tuple(p["classes"]) == dc.CLASSES_V2 and "register_change" not in p["classes"]
    for k in ("near_words", "zero_gap_s", "island_max_words", "backchannel_max_words", "address_max_words",
              "answer_openers", "backchannel", "second_person", "solicit", "solicit_bigrams"):
        assert p[k] == p1[k], k
    assert not any(k.startswith("register") for k in p) and "first_person" not in p
    assert set(p["question_answer_openers"]) == set(p1["answer_openers"]) | {"so"}
    assert "so" not in p["answer_openers"]                          # address_reply keeps v1's list
    assert (p["clause_pause_s"], p["fillers"], p["pair_max_words"]) == (0.3, ["uh", "um"], 60)
    assert p["clause_punct"] == [",", ":", ";", "\u2013", "\u2014"]
    assert p["pair_return_from"] == ["dropped_raw_change", "question_answer", "backchannel",
                                     "clause_backchannel", "address_reply"]
    assert p["pair_start_from"] == ["backchannel", "clause_backchannel"]
    assert not {"paired_return", "paired_start"} & set(p["pair_return_from"] + p["pair_start_from"])


@check("clauses: split at , ; : and dashes (closers stripped) and at a pause over 0.3 s, never across sentences")
def _():
    r = dc.render(make_doc([(A, "We cover memory and, yeah, I think so.")]))
    P = dc.default_params(2)
    assert dc.clauses(r, P) == [(0, 3), (4, 4), (5, 7)]
    r = dc.render(make_doc([(A, 'he said "right," then\u2014 went on; to this: here.')]))
    assert dc.clauses(r, P) == [(0, 2), (3, 3), (4, 5), (6, 7), (8, 8)]      # 9 words, 4 splits
    r = dc.render(make_doc([(A, "we cover memory and"), (A, "right", 0.32), (A, "then sleep.", 0.32)]))
    assert dc.clauses(r, P) == [(0, 3), (4, 4), (5, 6)]
    assert [dc.clause_split(r, p, P) for p in (4, 5)] == ["pause", "pause"]
    r = dc.render(make_doc([(A, "we cover memory and"), (A, "right", 0.29), (A, "then sleep.", 0.29)]))
    assert dc.clauses(r, P) == [(0, 6)]
    r = dc.render(make_doc([(A, "We cover memory. Then sleep.")]))
    assert dc.clauses(r, P) == [(0, 2), (3, 4)] == r.sentences


@check("clause_backchannel: a backchannel clause inside a sentence, mid-run: [f-1, l+1]")
def _():
    d = make_doc([(A, "We cover memory and, yeah, I think so.")])
    assert got2(d, "clause_backchannel") == [(3, 5, 4)]
    e = ev2(d, "clause_backchannel")[0]
    assert (e["tokens"], e["words"], e["split_before"], e["split_after"]) == (["yeah"], 1, "punct", "punct")
    assert got2(d, "backchannel") == [] and got(d, "clause_backchannel") == []      # v1 has no such class
    d = make_doc([(A, "We cover memory. Yeah, I think so.")])                        # sentence-initial
    assert got2(d, "clause_backchannel") == [(2, 4, 3)]
    assert ev2(d, "clause_backchannel")[0]["split_before"] == "sentence"
    d = make_doc([(A, "we cover memory and"), (A, "right okay", 0.35), (A, "then sleep comes.", 0.35)])
    assert got2(d, "clause_backchannel") == [(3, 6, 4)]                              # pause-bounded, 2 words
    assert ev2(d, "clause_backchannel")[0]["split_before"] == "pause"


@check("clause_backchannel does not fire: a whole sentence, 4 tokens, a filler, a content word, a stretch edge, no clause split")
def _():
    d = make_doc([(A, "We cover memory. Yeah. And then sleep.")])
    assert got2(d, "clause_backchannel") == [] and got2(d, "backchannel") == [(2, 4, 3)]
    for text in ("we go on, yeah right okay sure, then more.", "we go on, um, then more.",
                 "we go on, uh, then more.", "we go on, yeah maybe, then more."):
        assert got2(make_doc([(A, text)]), "clause_backchannel") == [], text
    assert got2(make_doc([(A, "We go on.", 0.0), (B, "Yeah, I think so.", 1.0)]), "clause_backchannel") == []
    assert got2(make_doc([(A, "we go on and, okay,"), (B, "then more there.", 1.0)]), "clause_backchannel") == []
    d = make_doc([(A, "we cover memory and"), (A, "right", 0.29), (A, "then sleep.", 0.29)])
    assert got2(d, "clause_backchannel") == []


@check("question_answer in v2 takes 'so' as an opener; v1 does not; address_reply keeps v1's openers")
def _():
    d = make_doc([(A, "So today we cover memory. Is this on the exam? So it is.")])
    assert got(d, "question_answer") == [] and got2(d, "question_answer") == [(4, 10, 10)]
    assert ev2(d, "question_answer")[0]["opener"] == "so"
    d = make_doc([(A, "Now the next part. Go ahead. So my question is about sleep.")])
    assert got2(d, "address_reply") == []


@check("register_change is gone from v2, where v1 fires")
def _():
    lecture = "The hippocampus binds features. Then the cortex stores them over weeks. Sleep helps that process a lot here."
    d = make_doc([(A, lecture + " I think my question is about the thing I saw.")])
    assert len(got(d, "register_change")) == 2 and got2(d, "register_change") == []
    res = dc.generate(d, version=2)
    assert "register_change" not in res["counts"]["by_class"] and "clause_backchannel" in res["counts"]["by_class"]


EXCHANGE = ("We cover memory. Okay. And that is what the rest is about here. "
            "Does this apply to sleep? So it does apply there too.")
#            0  1     2       3     4   5    6  7    8   9    10 11    12
#            13   14   15    16 17     18 19 20   21    22    23


@check("paired_return: forward from a start inside a run to the first question end: [q, q+1]")
def _():
    d = make_doc([(A, EXCHANGE)])
    assert got2(d, "backchannel") == [(2, 4, 3)] and got2(d, "question_answer") == [(12, 18, 18)]
    assert got2(d, "paired_return") == [(17, 18, 18)]
    e = ev2(d, "paired_return")[0]
    assert e["question_end_word_i"] == 17 and e["sources"] == [{"class": "backchannel", "at_word_i": 3, "words": 13}]
    assert got2(d, "paired_start") == []                           # no question before the cue


@check("paired_return: one candidate per question end, listing every source that found it")
def _():
    d = make_doc([(A, EXCHANGE)], extra=[(t0(3), t1(3), B)])          # a raw turn the render outvotes
    assert got2(d, "dropped_raw_change") == [(0, 7, 3)]
    assert got2(d, "paired_return") == [(17, 18, 18)]
    assert ev2(d, "paired_return")[0]["sources"] == [
        {"class": "dropped_raw_change", "at_word_i": 3, "words": 10},
        {"class": "backchannel", "at_word_i": 3, "words": 13}]


@check("paired_return: within 60 words of the source window's end, and never across a rendered boundary")
def _():
    head = "We cover memory. Okay. "
    d = make_doc([(A, head + words(59) + " end? So more.")])         # 'end?' at 63 = 4 + 59
    assert got2(d, "paired_return") == [(63, 64, 64)]
    d = make_doc([(A, head + words(60) + " end? So more.")])         # 'end?' at 64: one word too far
    assert got2(d, "paired_return") == []
    d = make_doc([(A, "We cover memory. Okay. And that is it."), (B, "Does this apply to sleep?", 1.0),
                  (A, "So it does.", 1.0)])
    assert got2(d, "backchannel") == [(2, 4, 3)] and got2(d, "paired_return") == []


@check("paired_start: back from an evaluation cue to the last question end before it: [q, q+1]")
def _():
    d = make_doc([(A, "Is this clear? What binds the features? The hippocampus does. Exactly. And then we move on.")])
    #                 0  1    2      3    4     5   6         7   8           9     10       11  12   13 14  15   16
    assert got2(d, "backchannel") == [(9, 11, 10)]
    assert got2(d, "paired_start") == [(6, 7, 7)]
    assert ev2(d, "paired_start")[0]["sources"] == [{"class": "backchannel", "at_word_i": 10, "words": 3}]
    assert got2(d, "paired_return") == []                          # no question after it
    d = make_doc([(A, "What binds the features? Yeah. And then we move on.")])
    assert got2(d, "paired_start") == [(3, 4, 4)]                  # inside its source's window: still emitted
    assert [s["classes"] for s in dc.generate(d, version=2)["sites"]] == [
        ["question_answer", "backchannel", "paired_start"]]


@check("paired_start: within 60 words back, and never across a rendered boundary")
def _():
    tail = " The hippocampus does. Exactly. And then we move on."
    d = make_doc([(A, "What is it? " + words(57) + tail)])          # cue window starts at 3+57+2 = 62; '?' at 2
    assert got2(d, "backchannel")[0][0] == 62 and got2(d, "paired_start") == []
    d = make_doc([(A, "What is it? " + words(56) + tail)])          # 61 - 59 = 2: just in reach
    assert got2(d, "paired_start") == [(2, 3, 3)]
    d = make_doc([(B, "What binds the features?"), (A, "The hippocampus does. Exactly. And then we move on.", 1.0)])
    assert got2(d, "backchannel") == [(6, 8, 7)] and got2(d, "paired_start") == []


def busy_v2_doc():
    return make_doc([(A, "So today we cover memory. Is this on the exam? So it is. We go on and, yeah, I see. "
                         "Okay. What binds the features? The hippocampus does. Exactly. Then more here."),
                     (B, "and then we", 0.2), (A, "Go ahead.", 1.0), (B, "Is it on the test?", 1.0),
                     (A, "Okay so my point is this. Any questions? Yeah I have one.")],
                    extra=[(t0(3), t1(5), B), (40.0, 40.5, B)])


@check("v2 deterministic: the same record twice, ids in window order, all thirteen classes in output order")
def _():
    a, b = dc.generate(busy_v2_doc(), version=2), dc.generate(busy_v2_doc(), version=2)
    assert json.dumps(a) == json.dumps(b)
    assert [c["id"] for c in a["candidates"]] == [f"C-{k:04d}" for k in range(1, len(a["candidates"]) + 1)]
    keys = [(c["first_word_i"], c["last_word_i"], dc.CLASSES_V2.index(c["class"])) for c in a["candidates"]]
    assert keys == sorted(keys)
    assert list(a["counts"]["by_class"]) == list(dc.CLASSES_V2)
    fired = {c["class"] for c in a["candidates"]}
    assert {"clause_backchannel", "paired_return", "paired_start", "question_answer", "backchannel",
            "address_reply", "shift", "dropped_raw_change"} <= fired, sorted(fired)
    assert all(s["classes"] == [x for x in dc.CLASSES_V2 if x in s["classes"]] for s in a["sites"])


@check("v2 CLI: --version 2 writes <stem>_speaker_candidates_v2.json with its version, plan and hash; no text")
def _():
    out, files, stdout = cli_output(busy_v2_doc(), "--version", "2", name="rec_speaker_candidates_v2.json")
    assert files == ["rec_raw.json", "rec_speaker_candidates_v2.json"]
    res = json.loads(out)
    assert (res["format"], res["version"], res["plan"]) == (1, 2, "docs/diarization_candidates_v2_plan.md")
    assert res["params"] == dc.default_params(2) and res["params_sha256"] == V2_PARAMS_SHA
    assert list(res)[:4] == ["source", "source_sha256", "format", "version"]
    assert "generator v2" in stdout and "clause_backchannel" in stdout and "register_change" not in stdout
    assert out == cli_output(busy_v2_doc(), "--version", "2", name="rec_speaker_candidates_v2.json")[0]
    blob = out.decode().lower()
    for tok in ("hippocampus", "features", "exam", "memory", "test"):
        assert tok not in blob, f"transcript text {tok!r} leaked into the candidates"


@check("v2 CLI refuses an unknown version")
def _():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "rec_raw.json").write_text(json.dumps(busy_v2_doc()))
        r = subprocess.run([sys.executable, str(GEN), str(Path(td) / "rec_raw.json"), "--version", "3"],
                           capture_output=True, text=True)
        assert r.returncode == 2 and "--version" in r.stderr


# ------------------------------------------------------------- v2 scoring ---
#
# One sentence per row, a 1 s pause between rows, every row the instructor's
# label (A), and an hour of audio so the volume caps are reachable. Row 1 holds
# a second speaker's words from a comma onward, which only v2's
# clause_backchannel sees; rows 4-6 are a student's turn opened by a
# backchannel sentence and closed by a question the instructor answers with
# "So". Positions: rows start at 0, 4, 12, 17, 21, 22, 28, 32, 38, 41 .. 66;
# 71 words.
#
#   missed changes: 7 (L->S, 5 words, high harm), 12 (S->L, return),
#                   21 (L->S, 11 words, high harm), 32 (S->L, return)
#   v1 windows:     address_reply [11,12]; backchannel [20,22]
#   v2 windows:     those, plus clause_backchannel [6,8], question_answer
#                   [27,32] and paired_return [31,32] (three sources)

SCORED_V2 = [
    (A, "Today we study memory.", None),
    (A, "You can bind, yeah, that makes sense now.", '>> change: S "yeah"'),
    (A, "Right we move on now.", ">> back: L"),
    (A, "Next the hippocampus matters.", None),
    (A, "Okay.", ">> change: S"),
    (A, "And that is about sleep then.", None),
    (A, "Does sleep matter here?", None),
    (A, "So it matters a great deal.", ">> back: L"),
    (A, "That is all.", None),
] + [(A, "Then we keep going with this.", None)] * 5


def scored_v2():
    td, d = scored_workdir(SCORED_V2, 3600.0, versions=(1, 2))
    return td, d, (evaluated(d, "rec_speaker_candidates.json", SCORED_V2, 3600.0),
                   evaluated(d, "rec_speaker_candidates_v2.json", SCORED_V2, 3600.0))


@check("v2 scoring: the windows of both versions on the v2 record are the hand-worked ones")
def _():
    td, d, _ = scored_v2()
    with td:
        def win(name):
            cdoc = json.loads((d / name).read_text())
            return sorted((c["class"], c["first_word_i"], c["last_word_i"]) for c in cdoc["candidates"])
        assert win("rec_speaker_candidates.json") == [("address_reply", 11, 12), ("backchannel", 20, 22)]
        assert win("rec_speaker_candidates_v2.json") == sorted([
            ("address_reply", 11, 12), ("backchannel", 20, 22), ("clause_backchannel", 6, 8),
            ("question_answer", 27, 32), ("paired_return", 31, 32)])
        cdoc = json.loads((d / "rec_speaker_candidates_v2.json").read_text())
        pr = next(c for c in cdoc["candidates"] if c["class"] == "paired_return")
        assert [(s["class"], s["at_word_i"], s["words"]) for s in pr["evidence"]["sources"]] == [
            ("clause_backchannel", 7, 23), ("address_reply", 12, 19), ("backchannel", 21, 9)]


@check("v2 scoring: v1 surfaces 1/2 high-harm misses and fails the bar; v2 surfaces 2/2 and passes")
def _():
    td, d, (r1, r2) = scored_v2()
    with td:
        rec = lambda r: {k: (v["surfaced"], v["total"]) for k, v in r["recall"].items()}   # noqa: E731
        assert rec(r1) == {"overall": (2, 4), "missed": (2, 4), "missed_misattributing": (1, 2),
                           "spurious": (0, 0), "high_harm": (1, 2)}, rec(r1)
        assert rec(r2) == {"overall": (4, 4), "missed": (4, 4), "missed_misattributing": (2, 2),
                           "spurious": (0, 0), "high_harm": (2, 2)}, rec(r2)
        assert r1["bar"]["high_harm_recall"]["pass"] is False and r1["bar"]["pass"] is False
        assert r2["bar"]["pass"] is True and r2["volume"]["sites"] == 4 and r2["volume"]["words_covered"] == 14
        assert (r1["generator_version"], r2["generator_version"]) == (1, 2)
        assert r1["params_preregistered"] is True and r2["params_preregistered"] is True
        assert [u["word_i"] for u in r1["unsurfaced_misses"]] == [7, 32] and r2["unsurfaced_misses"] == []


@check("v2 scoring: per class and per family, and the return metrics (reported, not gated)")
def _():
    td, d, (r1, r2) = scored_v2()
    with td:
        k = sc.KINDS
        pc = r2["per_class"]
        assert pc["clause_backchannel"]["surfaced"] == dict(zip(k, (1, 1, 0, 1)))
        assert pc["clause_backchannel"]["only_this"] == dict(zip(k, (1, 1, 0, 1)))
        assert pc["paired_return"]["surfaced"] == dict(zip(k, (1, 0, 0, 0)))
        assert pc["paired_return"]["only_this"] == dict(zip(k, (0, 0, 0, 0)))      # question_answer too
        assert list(pc) == list(dc.CLASSES_V2) and list(r1["per_class"]) == list(dc.CLASSES)
        assert list(r1["per_group"]) == ["split (all five)", "merge (all four)", "everything but shift"]
        assert list(r2["per_group"]) == ["split (all five)", "merge (all four)", "pair (both)",
                                         "everything but shift"]
        assert [(h["word_i"], h["return_word_i"], h["return_surfaced_by"]) for h in r2["high_harm"]] == [
            (7, 12, ["address_reply"]), (21, 32, ["question_answer", "paired_return"])]
        ret = lambda r: (r["return_surfaced"], r["bracketed"], r["high_harm_with_a_return"])   # noqa: E731
        assert ret(r1["high_harm_returns"]) == (1, 0, 2) and ret(r2["high_harm_returns"]) == (2, 2, 2)


@check("v2 score CLI: --speaker-map auto resolves the map from the labels; the bar line and no text")
def _():
    td, d, _ = scored_v2()
    with td:
        out = d / "v2score.json"
        r = subprocess.run([sys.executable, str(SCORE), str(d / "rec_labels.json"),
                            str(d / "rec_speaker_candidates_v2.json"), "--speaker-map", "auto",
                            "--json", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"speaker map (auto): {A}=L" in r.stdout and "bar: PASS" in r.stdout, r.stdout
        assert "generator v2" in r.stdout and "high-harm returns" in r.stdout
        res = json.loads(out.read_text())
        assert res["speaker_map"] == {A: "L"} and res["speaker_map_rule"]["student_words"] == 16
        blob = out.read_text().lower()
        for tok in ("hippocampus", "sleep", "matters", "memory"):
            assert tok not in blob, f"transcript text {tok!r} leaked into the score"


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
