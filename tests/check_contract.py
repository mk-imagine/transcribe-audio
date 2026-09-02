#!/usr/bin/env python3
"""Contract checks. Plain asserts, zero dependencies, runs anywhere.

    python3 tests/check_contract.py

Deliberately not pytest: the Mac host has no packages installed and never
will (Docker-only policy), and these checks must stay runnable on the machine
where the code is written. They cover exactly what the mock adapters exist to
cover -- the dispatch decisions, not the models.
"""

import json
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import flags, schema  # noqa: E402
from pipeline.adapters.base import Word  # noqa: E402
from pipeline.capabilities import Capabilities, ModelSpec, plan_for  # noqa: E402
from pipeline.orchestrator import (  # noqa: E402
    CapabilityConflict, apply_plan, transcribe,
)
from pipeline.registry import (  # noqa: E402
    REGISTRY, UnknownModelError, UnsupportedModelError, known_models,
    load_adapter_class, resolve,
)

PASSED = []
FAILED = []


def check(name):
    def wrap(fn):
        try:
            fn()
            PASSED.append(name)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - a stray SystemExit must
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))  # not abort the run
        return fn
    return wrap


# --------------------------------------------------------------- registry ---

@check("bug 7: a v1 checkpoint is refused, not routed to the v2 adapter")
def _():
    for bad in ("unsloth/crisperwhisper", "unsloth/CrisperWhisper",
                "nyrahealth/CrisperWhisper"):
        try:
            resolve(bad)
        except UnsupportedModelError as exc:
            assert "v1" in str(exc), exc
        else:
            raise AssertionError(f"{bad} resolved; it must be refused")


@check("bug 7: the MLX checkpoint is refused (D13), not sent to a deleted class")
def _():
    try:
        resolve("kyr0/crisperwhisper-unsloth-mlx")
    except UnsupportedModelError as exc:
        assert "D13" in str(exc)
    else:
        raise AssertionError("the MLX checkpoint resolved")


@check("Granite 4.1-plus resolves to its adapter with the end_only / silence / speaker shape")
def _():
    spec = resolve("ibm-granite/granite-speech-4.1-2b-plus")
    cls = load_adapter_class(spec)
    c = cls.capabilities
    assert (c.word_timestamps, c.silence_tokens, c.speaker_labels, c.longform, c.verbatim) == \
        ("end_only", True, True, "needs_chunking", "no"), c
    p = plan_for(c, diarize_requested=True)
    assert p.derive_starts and p.chunking and not p.needs_diarizer and not p.warnings, p
    assert not plan_for(c, mode="verbatim").ok, "verbatim must be refused"


@check("an unknown id is refused with the supported list, never guessed at")
def _():
    for unknown in ("some/whisper-like-model", "crisperwhisper", "nyralabs/CrisperWhisper3"):
        try:
            resolve(unknown)
        except UnknownModelError as exc:
            assert "not in the registry" in str(exc)
        else:
            raise AssertionError(f"{unknown} resolved by name pattern")


@check("lookup is case-folded, matching how the Hub resolves ids")
def _():
    assert resolve("NYRALABS/CrisperWhisper2.0_LARGE").model_id == "nyralabs/CrisperWhisper2.0_large"


@check("every registered adapter agrees with the registry's declaration")
def _():
    for spec in REGISTRY.values():
        if spec.unsupported:
            continue
        cls = load_adapter_class(spec)
        assert cls.capabilities == spec.capabilities, spec.model_id


@check("capability drift between registry and adapter is caught")
def _():
    spec = resolve("mock/start-end")
    drifted = ModelSpec(
        model_id=spec.model_id, adapter=spec.adapter,
        capabilities=Capabilities("end_only", "no", False, False, "native", False),
    )
    try:
        load_adapter_class(drifted)
    except RuntimeError as exc:
        assert "capability drift" in str(exc)
    else:
        raise AssertionError("drift went undetected")


@check("a ModelSpec must declare capabilities or a refusal reason")
def _():
    try:
        ModelSpec(model_id="x/y")
    except ValueError:
        return
    raise AssertionError("an empty ModelSpec was accepted")


@check("an invalid capability value is rejected at declaration time")
def _():
    try:
        Capabilities("start_end", "maybe", False, False, "native", False)
    except ValueError:
        return
    raise AssertionError("verbatim='maybe' was accepted")


# ------------------------------------------------------------- dispatch ----

@check("start_end -> native timing, no chunking, no alignment")
def _():
    p = plan_for(resolve("mock/start-end").capabilities, mode="verbatim")
    assert (p.timing_source, p.chunking, p.derive_starts, p.forced_alignment) == \
        ("native", False, False, False), p


@check("end_only + silence_tokens -> derived starts, chunking, no warning")
def _():
    p = plan_for(resolve("mock/end-only").capabilities, mode=None)
    assert p.timing_source == "derived" and p.derive_starts and p.chunking, p
    assert not p.warnings, p.warnings


@check("end_only without silence_tokens -> derives, and says why that is unsound")
def _():
    p = plan_for(resolve("mock/end-only-no-silence").capabilities, mode=None)
    assert p.derive_starts and p.warnings, p
    assert "silence_tokens" in p.warnings[0]


@check("word_timestamps='none' -> forced alignment")
def _():
    p = plan_for(resolve("mock/no-timestamps").capabilities, mode=None)
    assert p.forced_alignment and p.timing_source == "aligned", p


@check("speaker_labels=True -> no diarizer, even when diarization is requested")
def _():
    p = plan_for(resolve("mock/speaker-labels").capabilities, diarize_requested=True)
    assert not p.needs_diarizer, p


@check("speaker_labels=False -> diarizer required when requested")
def _():
    p = plan_for(resolve("mock/start-end").capabilities, diarize_requested=True)
    assert p.needs_diarizer
    assert not plan_for(resolve("mock/start-end").capabilities,
                        diarize_requested=False).needs_diarizer


@check("--mode verbatim against a verbatim='no' model is an error, not a warning")
def _():
    p = plan_for(resolve("openai/whisper-large-v3").capabilities, mode="verbatim")
    assert not p.ok and "verbatim" in p.errors[0], p


@check("hotwords on an untrained checkpoint warns; on hotwords='no' it errors")
def _():
    p = plan_for(resolve("nyralabs/CrisperWhisper2.0_large").capabilities,
                 mode="verbatim", hotwords=True)
    assert p.ok and any("never trained" in w for w in p.warnings), p
    q = plan_for(resolve("nyralabs/CrisperWhisper2.0_large_pro").capabilities,
                 mode="verbatim", hotwords=True)
    assert q.ok and not q.warnings, q
    r = plan_for(resolve("openai/whisper-large-v3").capabilities, hotwords=True)
    assert not r.ok, r


# ----------------------------------------------------------------- flags ---

@check("markers are matched case-insensitively, including lowercase [laughter]")
def _():
    assert flags.tag("[UH]") == ("filled_pause",)
    assert flags.tag("[laughter]") == ("vocalization",)
    assert flags.tag("[LAUGHTER]") == ("vocalization",)
    assert flags.tag("de-") == ("partial_word",)
    assert flags.tag("the") == ()


@check("an unrecognised bracketed token is flagged, not passed through as speech")
def _():
    assert flags.tag("[BLURP]") == ("unknown_marker",)


# ------------------------------------------------------- start derivation ---

@check("derived starts chain from the previous end and are marked as derived")
def _():
    words = [Word("a", None, 1.0), Word("b", None, 2.5), Word("c", None, 3.0)]
    from pipeline.adapters.base import AdapterResult
    result = AdapterResult(words=words)
    apply_plan(result, plan_for(resolve("mock/end-only").capabilities))
    assert [w.start for w in words] == [0.0, 1.0, 2.5], [w.start for w in words]


# ---------------------------------------------------------------- schema ---

def _silence(path, seconds=90):
    w = wave.open(str(path), "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000 * seconds); w.close()


class _Args:
    def __init__(self, **kw):
        self.model = "mock/start-end"
        self.adapter = None
        self.backend = None
        self.device = "cpu"
        self.no_diarize = True
        self.diarizer_model = "pyannote/speaker-diarization-community-1"
        self.hf_token = None
        self.mode = None
        self.hotwords_list = []
        self.timestamp_mode = "word"
        self.dual_stream = False
        self.compute_type = None
        self.source_path = None
        self.start_time = None
        self.end_time = None
        self.__dict__.update(kw)


def _run(model, **kw):
    with tempfile.TemporaryDirectory() as td:
        audio = Path(td) / "s.wav"
        _silence(audio)
        return transcribe(_Args(model=model, input_path=str(audio), **kw))


@check("a model with no modes runs without --mode; the registry supplies the default")
def _():
    # mock/end-only declares verbatim='no'. Left to argparse's old default this
    # refused itself over a mode nobody requested.
    doc = _run("mock/end-only")
    assert doc["asr"]["capabilities"]["verbatim"] == "no"


@check("a full mock run produces a valid schema-v1 document")
def _():
    doc = _run("mock/start-end")
    assert schema.validate(doc) == [], schema.validate(doc)
    assert doc["schema_version"] == "1.0"


@check("provenance carries model id, revision, capabilities and params")
def _():
    asr = _run("mock/start-end")["asr"]
    for key in ("model_id", "revision", "revision_source", "capabilities",
                "params", "granularity", "adapter", "adapter_source", "backend"):
        assert key in asr, key
    assert asr["capabilities"]["word_timestamps"] == "start_end"


@check("speaker turns are stored raw; words carry no speaker unless the ASR set it")
def _():
    doc = _run("mock/start-end")
    assert all(w["speaker"] is None and w["speaker_source"] is None for w in doc["words"])
    doc = _run("mock/speaker-labels", no_diarize=False)
    assert all(w["speaker_source"] == "asr" for w in doc["words"])
    assert doc["diarization"] is None, "a speaker_labels model must not load a diarizer"


@check("an unrecoverable range stays in the record and keeps the timeline")
def _():
    doc = _run("mock/failing")
    assert len(doc["errors"]) == 1, doc["errors"]
    e = doc["errors"][0]
    assert (e["start"], e["end"]) == (30.0, 60.0)
    starts = [w["start"] for w in doc["words"] if w["start"] is not None]
    assert not any(30.0 <= s < 60.0 for s in starts), "words survived inside the lost range"
    assert any(s >= 60.0 for s in starts), "the timeline did not resume after the gap"


@check("the unimplemented aligner is recorded as a warning, not silently skipped")
def _():
    doc = _run("mock/no-timestamps")
    assert any("forced" in w for w in doc["warnings"]), doc["warnings"]
    assert all(w["timing_source"] == "aligned" for w in doc["words"])


@check("derived timing is labelled 'derived' in every word of the record")
def _():
    doc = _run("mock/end-only", no_diarize=False)
    assert all(w["timing_source"] == "derived" for w in doc["words"])
    assert all(w["start"] is not None for w in doc["words"])


@check("an excerpt run names the original audio, not the temp file it was cut into")
def _():
    with tempfile.TemporaryDirectory() as td:
        original = Path(td) / "original.wav"
        excerpt = Path(td) / "temp_segment.wav"
        _silence(original, 120); _silence(excerpt, 90)
        doc = transcribe(_Args(model="mock/start-end", input_path=str(excerpt),
                               source_path=str(original), start_time="00:45:00",
                               end_time="00:46:30"))
    assert doc["source"]["audio_path"].endswith("original.wav"), doc["source"]
    assert doc["source"]["duration_s"] == 90.0
    assert doc["source"]["excerpt"]["offset_s"] == 2700.0, doc["source"]["excerpt"]
    assert doc["source"]["excerpt"]["timestamps_relative_to"] == "excerpt"


@check("dual-stream needs a selectable mode; it is refused where there is none")
def _():
    from pipeline.capabilities import plan_for
    ok = plan_for(resolve("nyralabs/CrisperWhisper2.0_large").capabilities,
                  mode="verbatim", dual_stream=True)
    assert ok.ok and ok.dual_stream, ok
    for model in ("openai/whisper-large-v3", "mock/end-only", "mock/no-timestamps"):
        bad = plan_for(resolve(model).capabilities, dual_stream=True)
        assert not bad.ok, model
        assert "dual-stream" in bad.errors[0]


@check("a dual-stream run records both renderings, each with its own words")
def _():
    doc = _run("mock/start-end", dual_stream=True, mode="verbatim")
    assert schema.validate(doc) == [], schema.validate(doc)
    sec = doc["secondary_stream"]
    assert sec["mode"] == "intended", sec["mode"]
    assert sec["words"], "the second stream must carry its own word timestamps"
    assert all(w["start"] is not None for w in sec["words"])
    # The two streams differ, and differ in the way the modes are supposed to.
    assert len(sec["words"]) < len(doc["words"])
    assert any("filled_pause" in w["flags"] for w in doc["words"])
    assert not any("filled_pause" in w["flags"] for w in sec["words"]), \
        "the intended stream should carry no filled pauses"


@check("without --dual_stream the second stream is absent, not empty")
def _():
    assert _run("mock/start-end")["secondary_stream"] is None


@check("a second stream repeating the primary mode is rejected")
def _():
    doc = _run("mock/start-end", dual_stream=True, mode="verbatim")
    doc["secondary_stream"]["mode"] = "verbatim"
    assert any("duplicates the primary mode" in p for p in schema.validate(doc))


@check("the preview can render either stream from the same record")
def _():
    from pipeline import preview
    doc = _run("mock/start-end", dual_stream=True, mode="verbatim")
    verbatim = preview.render(doc)
    intended = preview.render(doc, secondary=True)
    assert "[UM]" in verbatim and "[UM]" not in intended
    assert "[intended stream]" in intended and "[verbatim stream]" in verbatim


@check("CrisperWhisper picks a compute type the device can run: float32 on CPU")
def _():
    from pipeline.adapters.crisperwhisper2 import CrisperWhisper2Adapter as A
    assert A.default_compute_type("cpu") == "float32"
    assert A.default_compute_type("cuda") == "float16"
    assert A.default_compute_type("cuda:1") == "float16"
    a = A("nyralabs/CrisperWhisper2.0_large", "cpu", compute_type="int8")
    assert a.compute_type == "int8", "an explicit choice must be honoured"
    try:
        A("nyralabs/CrisperWhisper2.0_large", "cpu", compute_type="bfloat16")
    except ValueError:
        return
    raise AssertionError("an undocumented compute type was accepted")


@check("the validator catches a malformed document")
def _():
    doc = _run("mock/start-end")
    doc["words"][3]["i"] = 99
    doc["words"][5]["timing_source"] = "guessed"
    problems = schema.validate(doc)
    assert len(problems) >= 2, problems


@check("a verbatim request against a verbatim='no' model refuses before running")
def _():
    try:
        _run("mock/end-only", mode="verbatim")
    except CapabilityConflict:
        return
    raise AssertionError("a verbatim run against a non-verbatim model produced output")


@check("the record round-trips through JSON unchanged")
def _():
    doc = _run("mock/start-end")
    assert json.loads(json.dumps(doc)) == doc


# ---------------------------------------------------------------- preview --

@check("the preview renders and version-stamps itself")
def _():
    from pipeline import preview
    text = preview.render(_run("mock/start-end"))
    assert "stage-1 preview" in text and "model:" in text
    assert "[UM]" in text, "verbatim markers must survive into the preview"


@check("a word in a gap between diarizer turns still gets a speaker in the preview")
def _():
    from pipeline.preview import _speaker_at
    turns = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
             {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"}]
    assert _speaker_at(2.0, turns) == "SPEAKER_00"
    assert _speaker_at(4.2, turns) == "SPEAKER_00"   # in the gap, nearer the first
    assert _speaker_at(4.9, turns) == "SPEAKER_01"   # in the gap, nearer the second
    assert _speaker_at(0.0, []) is None


# ------------------------------------------------------------ dissenters ----

@check("the three §7b dissenters resolve, declare no timestamps, and route to forced alignment")
def _():
    for mid in ("Audio8/ARK-ASR-0.6B", "nvidia/parakeet-tdt-0.6b-v3",
                "ibm-granite/granite-speech-5.0-470m-turboctc"):
        spec = resolve(mid); cls = load_adapter_class(spec)
        c = cls.capabilities
        assert c.word_timestamps == "none" and c.verbatim == "no" and c.longform == "needs_chunking", (mid, c)
        p = plan_for(c, diarize_requested=False)
        assert p.forced_alignment and p.timing_source == "aligned" and p.chunking
        assert not plan_for(c, mode="verbatim").ok


# ------------------------------------------------------------ decode once ----

@check("a compressed source is decoded once at the boundary; PCM containers pass through")
def _():
    sys.path.insert(0, str(ROOT / "src"))
    import transcribe_audio as ta
    from pathlib import Path as P
    for ext in (".m4a", ".mp3", ".aac", ".ogg", ".M4A"):
        assert ta.needs_transcode(P(f"x{ext}")), ext
    for ext in (".wav", ".WAV", ".flac"):
        assert not ta.needs_transcode(P(f"x{ext}")), ext
    # the source block records the decode, keyed on the original suffix
    from pipeline.orchestrator import _source_block
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "temp_decoded_x.wav"; _silence(wav, 2)
        a = _Args(model="mock/start-end", input_path=str(wav), source_path=str(Path(td) / "x.m4a"))
        a.transcoded_from = ".m4a"
        (Path(td) / "x.m4a").write_bytes(b"not really audio")
        blk = _source_block(a, wav, 2.0)
        assert blk["audio_path"].endswith("x.m4a") and blk["transcoded"]["from"] == ".m4a"


@check("the two new vocalization tags are recognised; an unseen one is still flagged unknown")
def _():
    assert flags.tag("[throatclearing]") == ("vocalization",)
    assert flags.tag("[lipsmack]") == ("vocalization",)
    assert flags.tag("[zzzt]") == ("unknown_marker",)


# ---------------------------------------------------------- granite parsing --

@check("Granite timestamps unwrap across the 10-second rollover, per the card's algorithm")
def _():
    from pipeline.adapters.granite41plus import unwrap_timestamps
    # 9.8s; then 030 reads lower -> one rollover, 10.3s; 120 -> 11.2s; then 050 reads
    # lower again -> second rollover, 20.5s; 150 -> 21.5s. A rollover happens only
    # when a reading drops below the running end, never merely because ten
    # seconds of audio have passed.
    words, mono, trailing = unwrap_timestamps(
        "hello [T:980] world [T:030] _ [T:120] again [T:050] more [T:150]")
    assert [(w.text, w.end) for w in words] == \
        [("hello", 9.8), ("world", 10.3), ("_", 11.2), ("again", 20.5), ("more", 21.5)], \
        [(w.text, w.end) for w in words]
    assert mono and trailing == ""
    assert words[2].flags == ("silence",) and words[0].flags == ()
    assert all(w.start is None for w in words), "starts are derived by the orchestrator, not read"


@check("Granite timestamps: a plain centisecond reading would be wrong after ten seconds")
def _():
    from pipeline.adapters.granite41plus import unwrap_timestamps
    words, _, _ = unwrap_timestamps("a [T:995] b [T:005] c [T:010]")
    assert [w.end for w in words] == [9.95, 10.05, 10.1], [w.end for w in words]


@check("Granite timestamps: untagged trailing text is reported, not dropped silently")
def _():
    from pipeline.adapters.granite41plus import unwrap_timestamps
    words, _, trailing = unwrap_timestamps("one [T:050] two [T:100] three four")
    assert [w.text for w in words] == ["one", "two"] and trailing == "three four"


@check("Granite speaker tags parse in order, with untagged lead-in kept")
def _():
    from pipeline.adapters.granite41plus import parse_speakers
    segs = parse_speakers("[Speaker 1]: hello there [Speaker 2]: hi [Speaker 1]: ok")
    assert segs == [("Speaker 1", "hello there"), ("Speaker 2", "hi"), ("Speaker 1", "ok")], segs
    assert parse_speakers("lead in [Speaker 1]: x") == [(None, "lead in"), ("Speaker 1", "x")]
    assert parse_speakers("no tags at all") == [(None, "no tags at all")]


@check("Granite speaker alignment: TS words take the SAA speaker; silences and unmatched inherit")
def _():
    from pipeline.adapters.granite41plus import align_speakers, turns_from_labels
    ws = [Word("_", None, 0.3, flags=("silence",)), Word("hello", None, 0.6), Word("there", None, 0.9),
          Word("_", None, 1.4, flags=("silence",)), Word("hi", None, 1.7), Word("um", None, 1.9),
          Word("ok", None, 2.2)]
    segs = [("Speaker 1", "Hello, there."), ("Speaker 2", "Hi ok")]     # SAA drops the "um"
    labels = align_speakers(ws, segs)
    assert labels == ["Speaker 1", "Speaker 1", "Speaker 1", "Speaker 1", "Speaker 2", "Speaker 2", "Speaker 2"], labels
    turns = turns_from_labels(ws, labels)
    assert turns == [{"start": 0.3, "end": 0.9, "speaker": "Speaker 1"},
                     {"start": 1.4, "end": 2.2, "speaker": "Speaker 2"}], turns
    assert align_speakers(ws, [(None, "hello there hi um ok")]) == ["Speaker 1"] * 7
    assert align_speakers(ws, []) == ["Speaker 1"] * 7


@check("a silence token is flagged 'silence' by the tagger")
def _():
    assert flags.tag("_") == ("silence",)
    assert flags.tag("word_") == ()


# ---------------------------------------------------------------- fixture --

FIXTURE = ROOT / "tests" / "fixtures" / "golden.json"


@check("the committed fixture is a valid schema-v1 record")
def _():
    assert FIXTURE.exists(), f"{FIXTURE} is missing"
    doc = json.loads(FIXTURE.read_text())
    assert schema.validate(doc) == [], schema.validate(doc)
    assert doc["schema_version"] == "1.0"


@check("the fixture has what stage 2 needs: both streams, turns, flags, no holes")
def _():
    doc = json.loads(FIXTURE.read_text())
    assert doc["secondary_stream"] and doc["secondary_stream"]["mode"] == "intended"
    assert doc["secondary_stream"]["words"], "intended stream must carry words"
    assert len(doc["speaker_turns"]) >= 10, "too few turns to exercise turn grouping"
    assert len({t["speaker"] for t in doc["speaker_turns"]}) >= 2, "single-speaker fixture"
    flags = {f for w in doc["words"] for f in w["flags"]}
    assert "filled_pause" in flags, "no filled pauses: the coding profile has nothing to show"
    assert doc["errors"] == [], "the fixture must have a complete timeline"
    assert all(w["timing_source"] == "native" for w in doc["words"])


@check("the fixture carries provenance and no cluster user id")
def _():
    doc = json.loads(FIXTURE.read_text())
    asr, run = doc["asr"], doc["run"]
    assert asr["revision"] and doc["diarization"]["revision"], "model revisions missing"
    assert run["pipeline_version"]["commit"], "pipeline commit missing"
    assert doc["source"]["audio_sha256"], "audio hash missing"
    assert not doc["source"]["audio_path"].startswith("/"), "absolute cluster path leaked"
    assert "918204214" not in FIXTURE.read_text(), "cluster user id leaked into the fixture"


def main():
    for name in PASSED:
        print(f"  ok    {name}")
    for name, err in FAILED:
        print(f"  FAIL  {name}\n          {err}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
