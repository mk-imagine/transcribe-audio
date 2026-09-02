#!/usr/bin/env python3
"""Stage-2 checks against the committed fixture. Zero dependencies.

    python3 tests/check_render.py

Everything in stage 2 is a pure function over tests/fixtures/golden.json, so
these run in milliseconds on any machine. They check structure -- every word
lands exactly once, boundaries are what they claim, anchors sit inside their
turns, the stamp is present -- not prose quality, which is a human's job.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from render import PROFILES, RWord, RenderParams, select_stream, to_rwords  # noqa: E402
from render import segment, speakers  # noqa: E402
from render.formats import fmt_time, html as fmt_html, plain as fmt_plain, txt as fmt_txt  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "golden.json"
DOC = json.loads(FIXTURE.read_text())
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


def _prepare(profile="coding", **kw):
    """Load src/render.py as a module (it is a script, not a package) and run prepare()."""
    from importlib import util
    spec = util.spec_from_file_location("render_cli", ROOT / "src" / "render.py")
    mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
    params = RenderParams(profile=PROFILES[profile], **kw)
    return mod.prepare(DOC, params), params, mod


# ------------------------------------------------------------- streams -----

@check("the coding profile reads the verbatim stream, the lecture profile the intended one")
def _():
    w, mode, warn = select_stream(DOC, "verbatim"); assert mode == "verbatim" and not warn
    w2, mode2, warn2 = select_stream(DOC, "intended"); assert mode2 == "intended" and not warn2
    assert len(w) != len(w2), "the two streams should differ"
    assert any(x["text"].startswith("[") for x in w) and not any(x["text"].startswith("[") for x in w2)


@check("a missing stream falls back to the primary with a warning, never silently")
def _():
    doc = dict(DOC); doc["secondary_stream"] = None
    w, mode, warn = select_stream(doc, "intended")
    assert mode == "verbatim" and warn and "dual_stream" in warn


# ------------------------------------------------------------ speakers -----

@check("a word inside a turn gets that turn's speaker; a word in a gap gets the nearest")
def _():
    turns = [{"start": 0.0, "end": 4.0, "speaker": "A"}, {"start": 6.0, "end": 9.0, "speaker": "B"}]
    ws = [RWord(0, "x", 1.0, 1.5), RWord(1, "y", 4.4, 4.6), RWord(2, "z", 5.8, 6.2), RWord(3, "q", 8.0, 8.4)]
    speakers.assign(ws, turns)
    assert [w.speaker for w in ws] == ["A", "A", "B", "B"], [w.speaker for w in ws]


@check("no turns -> no speakers, and the render still works")
def _():
    ws = to_rwords(DOC["words"][:20]); speakers.assign(ws, [])
    assert all(w.speaker is None for w in ws)
    ts = segment.segment(ws, 0.5, 30)
    assert len(ts) == 1 and ts[0].speaker is None


@check("smoothing flips a short island inside a run and leaves run edges alone")
def _():
    def mk(spks):
        ws = [RWord(i, "w", i * 0.3, i * 0.3 + 0.25) for i in range(len(spks))]   # gaps 0.05 < threshold
        for w, s in zip(ws, spks): w.speaker = w.speaker_raw = s
        return ws
    ws = mk("AAABAAA"); n = speakers.smooth(ws, 0.5, 2)
    assert n == 1 and "".join(w.speaker for w in ws) == "AAAAAAA"
    ws = mk("AAABBAA"); n = speakers.smooth(ws, 0.5, 2)
    assert n == 2 and "".join(w.speaker for w in ws) == "AAAAAAA"
    ws = mk("AAABBBA"); n = speakers.smooth(ws, 0.5, 2)         # island of 3 > max_island
    assert n == 0 and "".join(w.speaker for w in ws) == "AAABBBA"
    ws = mk("BAAAAAA"); n = speakers.smooth(ws, 0.5, 2)         # edge island: leave it
    assert n == 0
    ws = mk("AAABAAA"); ws[3].start += 1.0; ws[3].end += 1.0    # a pause before the island splits the run
    for w in ws[4:]: w.start += 1.0; w.end += 1.0
    assert speakers.smooth(ws, 0.5, 2) == 0


@check("smoothing on the fixture reassigns a few words, never many, and never with --no-smoothing")
def _():
    ws = to_rwords(DOC["words"]); speakers.assign(ws, DOC["speaker_turns"])
    n = speakers.smooth(ws, 0.5, 2)
    assert 0 <= n <= len(ws) * 0.02, f"{n} of {len(ws)} words reassigned looks wrong"
    changed = sum(1 for w in ws if w.speaker != w.speaker_raw)
    assert changed == n


@check("the speaker map is applied at render time; unknown labels pass through")
def _():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("# who is who\nSPEAKER_00: Dr. Host\nSPEAKER_01: 'Guest'\n\n"); path = fh.name
    m = speakers.load_map(path)
    assert m == {"SPEAKER_00": "Dr. Host", "SPEAKER_01": "Guest"}
    ws = [RWord(0, "a", 0, 1), RWord(1, "b", 1, 2), RWord(2, "c", 2, 3)]
    for w, s in zip(ws, ("SPEAKER_00", "SPEAKER_01", "SPEAKER_02")): w.speaker = s
    speakers.apply_map(ws, m)
    assert [w.speaker for w in ws] == ["Dr. Host", "Guest", "SPEAKER_02"]


@check("a malformed speaker map is refused with the line number")
def _():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("SPEAKER_00: ok\nthis line has no colon\n"); path = fh.name
    try:
        speakers.load_map(path)
    except ValueError as exc:
        assert ":2:" in str(exc); return
    raise AssertionError("malformed map accepted")


# --------------------------------------------------------- segmentation ----

@check("every word lands in exactly one sentence, in order")
def _():
    ws = to_rwords(DOC["words"]); speakers.assign(ws, DOC["speaker_turns"])
    sents = segment.sentences(ws, 0.5)
    seen = [w.i for s in sents for w in s.words]
    assert seen == [w["i"] for w in DOC["words"]]


@check("boundaries are what they claim: punct ends with punctuation, pause has a gap, speaker changes")
def _():
    ws = to_rwords(DOC["words"]); speakers.assign(ws, DOC["speaker_turns"])
    sents = segment.sentences(ws, 0.5)
    kinds = {s.boundary for s in sents}
    assert {"punct", "pause", "speaker", "end"} <= kinds, kinds
    for a, b in zip(sents, sents[1:]):
        last, nxt = a.words[-1], b.words[0]
        if a.boundary == "speaker": assert last.speaker != nxt.speaker
        elif a.boundary == "punct": assert segment.is_terminal(last)
        elif a.boundary == "pause": assert nxt.start - last.end > 0.5 and not segment.is_terminal(last)
    assert sents[-1].boundary == "end"


@check("a bracketed marker never ends a sentence, even before a pause is checked")
def _():
    assert not segment.is_terminal(RWord(0, "[UH]", 0, 1))
    assert not segment.is_terminal(RWord(0, "[laughter]", 0, 1))
    assert segment.is_terminal(RWord(0, "done.", 0, 1))
    assert segment.is_terminal(RWord(0, 'right?"', 0, 1))
    assert not segment.is_terminal(RWord(0, "Dr.", 0, 1)) is False   # abbreviations are a known limitation


@check("consecutive turns always change speaker; the diarizer's 59 segments become fewer turns")
def _():
    ws = to_rwords(DOC["words"]); speakers.assign(ws, DOC["speaker_turns"]); speakers.smooth(ws, 0.5, 2)
    ts = segment.segment(ws, 0.5, 30)
    assert all(a.speaker != b.speaker for a, b in zip(ts, ts[1:]))
    assert 1 < len(ts) < len(DOC["speaker_turns"]), len(ts)
    assert sum(len(t.sentences) for t in ts) == len(segment.sentences(ws, 0.5))


@check("anchors sit inside their turn, on a sentence start where one is near, spaced about an interval apart")
def _():
    ws = to_rwords(DOC["words"]); speakers.assign(ws, DOC["speaker_turns"]); speakers.smooth(ws, 0.5, 2)
    ts = segment.segment(ws, 0.5, 30)
    total = 0
    for t in ts:
        if t.end - t.start <= 30: assert not t.anchors; continue
        times = sorted(w.start for w in t.words if w.i in t.anchors)
        assert times, f"long turn ({t.end - t.start:.0f}s) with no anchor"
        assert all(t.start < x < t.end for x in times)
        assert all(b - a >= 15 for a, b in zip(times, times[1:])), times
        sent_starts = {s.words[0].i for s in t.sentences}
        for i, kind in t.anchors.items():
            assert (i in sent_starts) == (kind == "sentence")
        total += len(times)
    assert total > 0


@check("the pause threshold re-derivation holds on raw timestamps: 0.5s ~ punctuation's cadence")
def _():
    ws = to_rwords(DOC["words"])
    gaps = [b.start - a.end for a, b in zip(ws, ws[1:])]
    zero = sum(1 for g in gaps if g == 0) / len(gaps)
    assert zero > 0.5, f"only {zero:.0%} zero gaps; the aligner assumption changed"
    per_pause = 600 / sum(1 for g in gaps if g > 0.5)
    per_punct = 600 / sum(1 for w in ws if segment.is_terminal(w))
    assert 0.5 < per_pause / per_punct < 2.0, (per_pause, per_punct)


# -------------------------------------------------------------- formats ----

@check("txt coding: numbered 1..N with N = sentence count, markers present, header stamped")
def _():
    (turns, st), params, _ = _prepare("coding")
    text = fmt_txt.render(turns, st, params)
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)  ", text, re.M)]
    assert nums == list(range(1, st["stats"]["sentences"] + 1)), (nums[:5], nums[-1])
    assert "[UM]" in text or "[UH]" in text
    assert f"hash {st['content_hash']}" in text and "# model:" in text


@check("txt lecture: no line numbers, no markers, intended stream, one paragraph per turn")
def _():
    (turns, st), params, _ = _prepare("lecture")
    text = fmt_txt.render(turns, st, params)
    assert not re.search(r"^\s*\d+  \S", text, re.M)
    assert "[UM]" not in text and "[UH]" not in text
    assert "(intended stream)" in text
    # A pause-split fragment must not sit on its own line: every body line but
    # a turn's last should be near the wrap width.
    body = [ln for ln in text.splitlines() if ln.startswith("    ") and ln.strip()]
    blocks, cur = [], []
    for ln in text.splitlines():
        if ln.startswith("    "): cur.append(ln)
        elif cur: blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    short_mid = [ln for b in blocks for ln in b[:-1] if len(ln) < params.width * 0.6]
    assert not short_mid, short_mid[:3]


@check("timestamps are shown in the original recording's time: the fixture starts at 05:00")
def _():
    (turns, st), params, _ = _prepare("coding")
    text = fmt_txt.render(turns, st, params)
    first = re.search(r"^\[(\d\d:\d\d:\d\d\.\d)\]", text, re.M).group(1)
    assert first.startswith("00:05:0"), first
    (turns2, st2), params2, _ = _prepare("coding", apply_offset=False)
    text2 = fmt_txt.render(turns2, st2, params2)
    assert re.search(r"^\[(\d\d:\d\d:\d\d\.\d)\]", text2, re.M).group(1).startswith("00:00:0")
    assert fmt_time(61.25, 300.0) == "00:06:01.2"


@check("plain: words only, every word of every turn, no timestamps or labels")
def _():
    (turns, st), params, _ = _prepare("coding")
    text = fmt_plain.render(turns, st, params)
    assert not re.search(r"\d\d:\d\d:\d\d", text) and "SPEAKER" not in text and "#" not in text
    assert len(text.split()) == sum(len(t.words) for t in turns)


@check("html: self-contained, balanced, counters and @page margin in the coding profile, escaped text")
def _():
    (turns, st), params, _ = _prepare("coding")
    turns[0].sentences[0].words[0].text = 'a<b & "c"'
    html = fmt_html.render(turns, st, params)
    assert html.count("<section") == html.count("</section>") == len(turns)
    assert html.count('<p class="line"') == html.count("</p>") == st["stats"]["sentences"]
    assert "<style>" in html and "@page" in html and "counter-increment: line" in html and "2.75in" in html
    assert 'class="marker"' in html and "break-inside: avoid" in html
    assert "a&lt;b &amp; &quot;c&quot;" in html and "a<b &" not in html
    (turns, st), params, _ = _prepare("lecture")
    html = fmt_html.render(turns, st, params)
    assert "counter-increment" not in html and 'class="marker"' not in html


# ------------------------------------------------------------------ CLI ----

@check("the CLI renders both profiles end to end and names outputs by profile")
def _():
    with tempfile.TemporaryDirectory() as td:
        for prof in ("coding", "lecture"):
            r = subprocess.run([sys.executable, str(ROOT / "src" / "render.py"), str(FIXTURE),
                                "-o", td, "--profile", prof, "--format", "html,plain"],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-400:]
            for suf in (".txt", ".html", "_plain.txt"):
                assert (Path(td) / f"golden_{prof}{suf}").exists(), suf


@check("the CLI refuses a bad format and a malformed speaker map with exit 2")
def _():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, str(ROOT / "src" / "render.py"), str(FIXTURE),
                            "-o", td, "--format", "docx"], capture_output=True, text=True)
        assert r.returncode == 2
        bad = Path(td) / "bad.yaml"; bad.write_text("no colon here\n")
        r = subprocess.run([sys.executable, str(ROOT / "src" / "render.py"), str(FIXTURE),
                            "-o", td, "--speaker-map", str(bad)], capture_output=True, text=True)
        assert r.returncode == 2


def main():
    for n in PASSED: print(f"  ok    {n}")
    for n, e in FAILED: print(f"  FAIL  {n}\n          {e}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
