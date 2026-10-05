#!/usr/bin/env python3
"""Checks for scripts/diarization_candidates.py and scripts/score_candidates.py. Zero dependencies.

    python3 tests/check_candidates.py

Every candidate class of docs/diarization_candidates_plan.md (§2) on a
synthetic record built here in code, firing where the rule says and not where
it does not, with the exact window; determinism; the CLI; and the recall
scorer end to end (sheet generated, filled, checked, candidates generated,
scored), with every number worked out by hand from the table in SCORED.
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


def scored_doc():
    words, turns, t, i = [], [], 0.5, 0
    for label, text, _ in SCORED:
        s0 = t
        for tok in text.split():
            words.append({"i": i, "text": tok, "start": round(t, 2), "end": round(t + 0.3, 2),
                          "timing_source": "native", "speaker": None, "speaker_source": None,
                          "conf": None, "flags": []})
            i += 1; t += 0.3
        turns.append({"start": round(s0, 2), "end": round(t, 2), "speaker": label})
        t += 1.0
    turns.append(dict(turns[2], speaker=B))           # outvoted: an equal turn sorted after A's
    doc = make_doc([])
    doc.update(words=words, text=" ".join(w["text"] for w in words), speaker_turns=turns)
    return doc


def fill(sheet):
    out, lines = [], sheet.read_text().splitlines()
    for n, ln in enumerate(lines):
        if ln.startswith("T0"):
            row = SCORED[int(ln[1:5]) - 1]
            if isinstance(row[2], str):
                out.append(row[2])
        elif ln.startswith(("boundary:", "who:")):
            below = next(x for x in lines[n:] if x.startswith("T0"))
            verdict, who = SCORED[int(below[1:5]) - 1][2]
            ln = f"boundary: {verdict}" if ln.startswith("boundary:") else f"who: {who}".rstrip()
        out.append(ln)
    sheet.write_text("\n".join(out) + "\n")


def scored_workdir():
    td = tempfile.TemporaryDirectory()
    d = Path(td.name)
    (d / "rec_raw.json").write_text(json.dumps(scored_doc(), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        assert dl.main([str(d / "rec_raw.json")]) == 0
        fill(d / "rec_labels.md")
        assert dl.main(["--check", str(d / "rec_labels.md")]) == 0, "the synthetic sheet does not check clean"
        assert dc.main([str(d / "rec_raw.json")]) == 0
    return td, d


def evaluated(d):
    labels = json.loads((d / "rec_labels.json").read_text())
    cdoc = json.loads((d / "rec_speaker_candidates.json").read_text())
    stream = sd.rendered_stream(labels, d / "rec_labels.json")
    truth = sd.derive_truth(labels, stream, MAP)
    return sc.evaluate(labels, stream, truth, MAP, cdoc, dc.render(scored_doc()))


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


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
