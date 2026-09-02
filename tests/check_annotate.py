#!/usr/bin/env python3
"""Stage-1.5 checks: the fluent view, the conjunction, and annotate.py end to end.

    python3 tests/check_annotate.py

Zero dependencies. The fluent-view cases are the ones the 2026-09-01 spike
measured; the end-to-end case builds three synthetic dissenters from the
committed fixture's own intended stream so no model is needed.
"""

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import schema  # noqa: E402
from pipeline.fluent import conjunction, fluent_view, name_like, unmatched  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "golden.json"
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


@check("fluent view: markers and partials dropped with an index map; repeats collapsed; numbers classed")
def _():
    v = fluent_view(["So", "[UM]", "the", "the", "p-", "process", "is", "2005", "twenty", "years."])
    assert v.tokens == ["so", "the", "process", "is", "<num>", "years"], v.tokens
    assert v.dropped == {1, 4}
    assert [v.index[k] for k in range(len(v.tokens))] == [0, 2, 5, 6, 7, 9]


@check("fluent view: contractions and pronoun-'s expand; hyphens split; apostrophes go")
def _():
    v = fluent_view(["gonna", "them's", "self-report", "don't"])
    assert v.tokens == ["going", "to", "them", "is", "self", "report", "dont"], v.tokens
    v2 = fluent_view(["them's"], expand_pronoun_s=False)
    assert v2.tokens == ["thems"]


@check("name_like: capitalised mid-sentence yes; sentence-initial or after a marker-then-period no")
def _():
    toks = ["and", "Eric", "Korshane", "[UH]", "it's", "pronounced.", "[laughter]", "Like", "you're"]
    assert name_like(toks, 1) and name_like(toks, 2)
    assert not name_like(toks, 7), "sentence-initial after '.' + marker"
    assert not name_like(toks, 0)


@check("conjunction: a token all dissenters miss is flagged; one they all match is not")
def _():
    prim = ["and", "Eric", "Korshane", "said", "hello."]
    d = [["and", "eric", "courchesne", "said", "hello"],
         ["and", "eric", "korschane", "said", "hello"],
         ["and", "eric", "korsheane", "said", "hello"]]
    cands, st = conjunction(prim, d)
    assert [c.token for c in cands if not c.masked] == ["Korshane"], cands
    assert st["flagged"] == 1 and st["masked"] == 0


@check("conjunction: two of three disagreeing is not enough by default; --min-dissenters lowers it")
def _():
    prim = ["a", "b", "c"]
    d = [["a", "x", "c"], ["a", "y", "c"], ["a", "b", "c"]]
    assert conjunction(prim, d)[1]["flagged"] == 0
    assert conjunction(prim, d, min_dissenters=2)[1]["flagged"] == 1


@check("adjacency mask: a residue token beside a marker is masked, a name beside a marker is kept")
def _():
    prim = ["if", "you", "[UM]", "had", "this", "[UH]", "idea", "and", "Eric", "Korshane", "[UH]", "said"]
    d = [["if", "you", "have", "this", "idea", "and", "eric", "courchesne", "said"]] * 3
    cands, st = conjunction(prim, d)
    by = {c.token: c for c in cands}
    assert by["had"].adjacent_disfluency and by["had"].masked
    assert by["Korshane"].adjacent_disfluency and by["Korshane"].name_like and not by["Korshane"].masked
    assert st["flagged"] == 1 and st["masked"] == 1
    # the mask is a rule, not a deletion: off, both are flagged
    assert conjunction(prim, d, mask_adjacent=False)[1]["flagged"] == 2
    # and the exemption can be switched off
    assert conjunction(prim, d, exempt_name_like=False)[1]["flagged"] == 0


@check("the conjunction requires at least one dissenter")
def _():
    try:
        conjunction(["a"], [])
    except ValueError:
        return
    raise AssertionError("empty dissenter list accepted")


def _dissenter_doc(base, text, model_id, k):
    """A minimal valid stage-1 record whose words are `text`."""
    d = copy.deepcopy(base)
    d["asr"] = dict(d["asr"], model_id=model_id, adapter=f"Fake{k}", revision=f"rev{k}")
    d["words"] = [{"i": i, "text": t, "start": None, "end": None, "timing_source": "none",
                   "speaker": None, "speaker_source": None, "conf": None, "flags": []}
                  for i, t in enumerate(text.split())]
    d["text"] = text; d["secondary_stream"] = None; d["speaker_turns"] = []; d["errors"] = []
    return d


@check("annotate.py end to end: flags a planted garble, masks a planted residue, records provenance")
def _():
    base = json.loads(FIXTURE.read_text())
    words = base["words"]
    # Plant: a capitalised mid-sentence word all dissenters render differently ...
    tgt = next(k for k, w in enumerate(words) if w["text"][:1].isupper() and k > 5
               and not words[k-1]["text"].endswith((".", "?", "!")) and not words[k-1]["text"].startswith("["))
    # ... and a residue: the word right after the first [UM], which the dissenters drop.
    um = next(k for k, w in enumerate(words) if w["text"] == "[UM]")
    resid = um + 1
    fluent_text = " ".join(w["text"] for k, w in enumerate(words)
                           if not w["text"].startswith("[") and not w["text"].endswith("-") and k != resid)
    variants = [fluent_text.replace(words[tgt]["text"], f"Zorblax{k}", 1) for k in range(3)]
    ids = ["Audio8/ARK-ASR-0.6B", "nvidia/parakeet-tdt-0.6b-v3", "ibm-granite/granite-speech-5.0-470m-turboctc"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        prim = td / "rec_raw.json"; schema.write(prim, base)
        dpaths = []
        for k, (v, mid) in enumerate(zip(variants, ids)):
            p = td / f"d{k}_raw.json"; schema.write(p, _dissenter_doc(base, v, mid, k)); dpaths.append(str(p))
        r = subprocess.run([sys.executable, str(ROOT / "src" / "annotate.py"), str(prim), "--dissenters", *dpaths],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        out = json.loads((td / "rec_annotated.json").read_text())
        assert schema.validate(out) == []
        flagged = [w["i"] for w in out["words"] if "proper_noun_candidate" in w["flags"]]
        assert tgt in flagged, (tgt, words[tgt]["text"], flagged[:10])
        assert resid not in flagged, "residue beside [UM] should be masked"
        masked = [c for c in out["annotation"]["primary"]["candidates"] if c["masked"]]
        assert any(c["i"] == resid for c in masked), "masked candidate must be recorded, not dropped"
        ann = out["annotation"]
        assert [d["model_id"] for d in ann["dissenters"]] == ids and all(d["revision"] for d in ann["dissenters"])
        assert "secondary_stream" in ann and ann["secondary_stream"]["mode"] == "intended"
        # a dissenter with (almost) no tokens is refused: it would vote against everything
        empty = td / "empty_raw.json"; schema.write(empty, _dissenter_doc(base, "just three words", ids[0], 9))
        r0 = subprocess.run([sys.executable, str(ROOT / "src" / "annotate.py"), str(prim), "--dissenters",
                             str(empty), dpaths[1], dpaths[2]], capture_output=True, text=True)
        assert r0.returncode == 2 and "cannot vote" in r0.stderr, r0.stderr[-300:]
        # fewer than three dissenters is refused unless asked for
        r2 = subprocess.run([sys.executable, str(ROOT / "src" / "annotate.py"), str(prim), "--dissenters", dpaths[0]],
                            capture_output=True, text=True)
        assert r2.returncode == 2
        r3 = subprocess.run([sys.executable, str(ROOT / "src" / "annotate.py"), str(prim), "--dissenters", dpaths[0],
                             "--min-dissenters", "1", "-o", str(td / "one.json")], capture_output=True, text=True)
        assert r3.returncode == 0, r3.stderr[-300:]


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
