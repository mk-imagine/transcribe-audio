# Transcription Pipeline — Plan of Attack

**Status:** design agreed, **nothing implemented yet**
**Last updated:** 2026-08-27
**Supersedes:** `docs/granite_word_timestamps_plan.md` (see §12)

> **Reading this cold?** §1–§4 give you the context and the decisions already made (with the
> reasoning, so they need not be re-litigated). §5–§8 are the design. §9–§11 are the work.
> Anything marked **OPEN** is genuinely undecided; everything else is settled unless the
> stated trigger for reopening it occurs.

---

## 1. What this project is

An audio transcription pipeline serving two research uses from **one transcription run per
recording**:

1. **Qualitative coding of research interviews** (identity & belonging research). Transcripts
   are **printed on paper** and coded by hand by undergraduate and graduate students. Requires
   verbatim text with disfluencies preserved, speaker attribution, and timestamps.
2. **Lecture recording transcription.** Readable clean prose, with timestamps so the original
   audio can be checked when the transcript looks wrong.

Disfluencies are **data**, not noise — hesitation before a disclosure about identity is
analytically meaningful. This is the single requirement that most constrains model choice.

### Operating constraints

| Constraint | Detail |
|---|---|
| Production compute | SFSU HPC cluster, SLURM, **NVIDIA A100** |
| Local machines | Apple Silicon Mac; Ubuntu 24.04 VM on Proxmox with **RTX 3070 (8 GB)** GPU passthrough (VM not yet provisioned for ML) |
| Install policy | **No bare-metal installs on the Mac** without explicit per-instance approval. Docker or remote hosts only. See root `CLAUDE.md`. |
| Data sensitivity | Interview audio is human-subjects data. All inference is local; models are fetched from HuggingFace, audio never leaves controlled machines. |
| Licensing | Non-commercial academic research. Non-commercial model weights are acceptable. |
| Branch policy | Never commit to `main`. See root `CLAUDE.md`. |

### Current state of the code

`src/transcribe_audio.py` (766 lines, everything in one file) is the existing pipeline. It
works but is being **replaced**, not extended. What is there now:

- `TranscriberFactory` — substring-sniffs model names to pick a class
- `WhisperTranscriber` / `CrisperWhisperTranscriber` / `MLXCrisperWhisperTranscriber` /
  `GraniteTranscriber` — subclasses that do **not** agree on output granularity (word vs chunk
  vs one-blob-per-segment)
- `BaseTranscriber` — fixed 300 s chunking (`SEGMENT_SIZE`)
- `TextCleaner` — none/basic/intelligent; **deletes** disfluencies inline in stage 1
- `Diarizer` — pyannote `speaker-diarization-community-1`
- `TranscriptionOrchestrator` — device selection, speaker assignment, JSON + TXT output
- `src/run_transcription.sh`, `src/transcribe.slurm` — SLURM wrappers (8 hr walltime)
- `src/transcribe_audio_mac.py` — separate Mac variant (349 lines)

Sample data: `data/251211_0009.wav`. Existing output: `transcripts/lecture transcript.txt` —
fully normalized prose, zero disfluencies, no timestamps, multi-thousand-word paragraph blocks.
Fine as lecture notes; **unusable for coding**, which is the gap this work closes.

---

## 2. Decision log

Settled. Each entry records what would reopen it.

| # | Decision | Why | Reopens if |
|---|---|---|---|
| D1 | **Split into three commands**: `transcribe` → `annotate` → `render` | Stage 1 queues on SLURM; stage 2 must be instant. Never re-queue a cluster job to change a pause threshold. | — |
| D2 | **Capability contract, not a model-name factory** | Plug-and-play comes from adapters declaring what they provide, with the orchestrator filling gaps. | — |
| D3 | **Stage 1 is verbatim and lossless.** No cleaning, filtering, or speaker assignment. | Cleaning in stage 1 destroys information before it is ever written to disk. | — |
| D17 | **Drive CrisperWhisper 2 through the `crisperwhisper` package, never `transformers.pipeline`.** | The package exposes `mode="verbatim"` (default), `hotwords`, `temperature_fallback` and `word_timestamps`. The pipeline exposes none of them, silently yields cleaned text, and runs 3x slower (12x vs 38x realtime). Measured: 121 filled pauses vs 0 on identical audio. | — |
| D19 | **Read the model's own docs before writing an integration; never assume a generic loader is correct.** | A generic loader that runs is not evidence it runs correctly — the failure is silent and produces self-consistent wrong measurements. See the callout at the head of §3 for three worked instances. | — |
| D18 | **Proper-noun detection by three-dissenter conjunction (§7b), compared on a fluent view.** | Glossaries cannot be built ahead of an arbitrary lecture, but cross-model disagreement localises garbles without one. Dissenters must come from independent lineages. | A single model gains reliable proper-noun accuracy |
| D4 | **Default model: `nyralabs/CrisperWhisper2.0_large`** | Verbatim is its explicit training objective, not an accident of its corpus. Only candidate with a documented verbatim/intended switch. | Phase 0 shows poor disfluency retention on real audio |
| D5 | **Second adapter: `ibm-granite/granite-speech-4.1-2b-plus`** | Apache 2.0, and exercises three capability paths CW2 does not (`end_only` timestamps, silence tokens, native speaker labels) — which is what proves the contract is real. | — |
| D6 | **Not pursuing Reverb ASR** | `verbatimicity` is the nicest design of the three, but it needs a WeNet fork as a second inference stack *and* has no documented word timestamps, so it needs an aligner anyway. | Its integration cost drops, or CW2 and Granite both fail |
| D7 | **Not pursuing Granite 5.0 TurboCTC** | No prompting, no timestamps. It is a throughput model. | — |
| D8 | **Disfluency flags are typed, not boolean** | The categories degrade independently across models and coders may treat them differently. | — |
| D9 | **Turn is the primary render unit**, subdivided within long turns | Confirmed by the user: interview coding is turn-level, with timestamps inside long turns for citation. | — |
| D10 | **Both profiles carry timestamps** | Lecture profile needs them to check the audio when a transcript looks wrong. | — |
| D11 | **HTML + print CSS is the default print format**; plain text always emitted; LaTeX optional; docx deferred | HTML has zero new dependencies and the fastest layout-iteration loop. | QDA tool adoption makes docx urgent |
| D12 | **Dual-stream output is an adapter-internal optimisation, not an architecture** | `transcribe_dual()` is ct2-only; falling back to two sequential `transcribe()` calls keeps the *semantics* testable on any backend. | — |
| D13 | **MPS and MLX are dropped.** Develop on CPU containers locally, CUDA on the cluster. | Docker on macOS cannot expose Metal to Linux containers, so MPS needs a bare-metal install — a policy exception for a platform that is never a deployment target. The repo already pays this tax (chunk-level timestamp workaround, whole MLX path). | — |
| D14 | **The RTX 3070 VM is general ML infrastructure, not a blocker for this project** | Phase 0 is fully answerable in a CPU container. The VM's real value is serving every ML repo and resolving the host-install friction permanently. | — |
| D15 | **Non-commercial weights accepted** | Non-commercial academic research; not researching model methods, just using them. | Work becomes commercial or ships tooling with weights |
| D16 | **QDA tool export deferred** | No budget; coding is manual on paper. The raw JSON is the hedge — an exporter is a stage-2 renderer, never a re-transcription. | A QDA tool is adopted |

---

## 3. Verified model facts

> ### Read the model's own documentation before writing any integration
>
> **Always check a model's card, README and package API before assuming a generic
> loader will work.** This applies to every open model — HuggingFace or elsewhere —
> and to every generic interface: `transformers.pipeline`, `AutoModelFor*`, or any
> other convenience wrapper.
>
> A generic loader that *runs* is not evidence that it runs *correctly*. The failure
> is quiet: the model produces plausible output, every measurement you take is
> internally consistent, and the conclusions are wrong. Three instances in one week:
>
> | model | generic assumption | what the docs said | cost |
> |---|---|---|---|
> | CrisperWhisper 2 | `transformers.pipeline` | package exposes `mode="verbatim"`, absent from the pipeline | benchmarked as emitting **zero** disfluencies; drove a ten-model search and a fine-tuning plan that were unnecessary |
> | ARK-ASR-0.6B | cast every float tensor to fp16, feed 90 s | cast **only** `inputs["audios"]`; `audio_max_length=30*16000`; `do_sample=False`; `bad_words_ids` | output collapsed into repeated CJK characters; nearly discarded a working model |
> | CTC models (Parakeet, Granite TurboCTC) | reuse the seq2seq `chunk_length_s` | CTC is frame-synchronous and needs no chunking | silently returned ~12% of the content (63 words where the reference was 545) |
>
> The tell in all three: output that is *plausible but wrong in a way no error message
> reports*. Before trusting a benchmark number, confirm the invocation matches the
> documented one — including defaults, dtype handling, input length limits and
> generation flags. Where a model ships its own package, prefer it over the generic
> wrapper and record which was used (§5 provenance).

**Verified 2026-08-27 by fetching the model cards directly.** These postdate common knowledge
cutoffs — do not "correct" them from memory. Re-verify if acting on them much later.

### CrisperWhisper 2.0 — `nyralabs/CrisperWhisper2.0_{large,turbo,medium,small}`

Source: `github.com/nyrahealth/CrisperWhisper` (+ `DOCS.md`)

- Package `crisperwhisper`, class `CrisperWhisperModel`. Install extras: `[ct2]` or `[transformers]`.
- `transcribe(audio, language, mode, hotwords, longform_strategy, word_timestamps,
  speculative_decoding, hallucination_mitigation)`
  - `mode`: `"verbatim"` | `"intended"`
  - `longform_strategy`: `"continuation"` | `"chunked_lcs"` | `"token_lcs"`
  - `hallucination_mitigation` defaults **True**
- Returns `TranscriptionResult`: `text`, `language`, `mode`, `duration`, `processing_time`,
  `words` (list of `WordTimestamp`), `chunks`
- `WordTimestamp`: `word`, **`start`**, **`end`** (seconds) — full start *and* end
- `transcribe_dual()` — verbatim + intended in one batched pass, **ct2 only**, ~1.9× faster
- `verbatimize(audio, transcript, ...)` — upgrades an existing clean transcript with the
  disfluencies actually present in the audio
- `forced_align()` — aligns a known transcript to audio for word timestamps
- **Long-form is native**: ">30s transcribed seamlessly via conditional continuation, with no
  chunk-boundary duplicates, drops, or stitching"
- Fillers surface as bracketed tokens: `[UH]`, `[UM]`
- ct2 is ~4–5× faster than transformers; compute types `float16`, `int8_float16`
- **No diarization** — pyannote is required regardless of model
- Licence: **code MIT, weights Nyra Health Non-Commercial Research.** Pro variants commercial-only.
- `large` ≈ 3.1 GB fp16 (Whisper-large-v3 derivative, 1.55 B params)
- **Apple Silicon / MPS support is undocumented.** Moot — see D13.

### Granite Speech 4.1-2b-plus — `ibm-granite/granite-speech-4.1-2b-plus`

Source: HuggingFace model card. Released **2026-04-28**. **Apache 2.0**.

- Word-level timestamps: `[T:N]` tags after each word, N in **centiseconds**, marking the
  **end** of the word. No start times.
- **Silences are transcribed as `_` with their own end timestamp.** This is why `end_only` is
  usable: with explicit pause tokens, `word_start = previous_token_end` is correct and pause
  durations are recoverable.
- Timestamp-mode prompt: `<|audio|> Timestamps: Transcribe the speech. After each word, add a
  timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]`
- Speaker attribution mode: emits `[Speaker 1]:` / `[Speaker 2]:` tags, numbered by first appearance
- Other modes: plain ASR, incremental decoding, keyword-list biasing
- **The plus variant does not produce punctuation or capitalisation** (unlike base 4.1-2b)
- No verbatim or disfluency claim anywhere on the card
- ~2 B params ≈ 4 GB bf16
- **OPEN:** whether timestamp mode and speaker-attribution mode can be combined in one prompt.
  Undocumented, and this project needs both. Phase 0 question 2.

### Measured behaviour (2026-08-31, real audio)

Two ~80 min recordings, 12 x 90 s windows, A100.

**CrisperWhisper 2 through its own package does what this project needs.** With
`mode="verbatim"` (the default) it emits bracketed disfluency tags, and four of §5's five
categories appear in real output:

| category | count over 12 windows | example |
|---|---|---|
| `filled_pause` | **121** (`[UH]` x83, `[UM]` x38) | `[UM] including Bob Knight and Eric ... [UH] it's pronounced` |
| `repetition` | 23 | "not not mirror images", "the the bone" |
| `partial_word` | 12 | "maybe the p-", "the pro- process of de- developmental" |
| `repair` | derivable | inferable from marker positions + repetitions; not tagged directly |
| `vocalization` | 1 | `[laughter]` — **lowercase**, unlike `[UH]`/`[UM]`; match tags case-insensitively |

Tag inventory over those 12 windows is exactly `{uh: 83, um: 38, laughter: 1}`. Vocalizations
are supported but rare in this material; an uppercase-only regex will miss them.

`word_timestamps=True` and the markers coexist. `hotwords=[...]` fixes proper nouns in the same
pass: without it "Eric Korshane ... pronounced Korshesney"; with it "Eric Courchesne ...
pronounced Courchesne", keeping all 22 markers in that window. `temperature_fallback=True` is
on by default and no looping was observed (0.0% duplicate 8-grams on all 12 windows).
Throughput **38x realtime** via the CTranslate2 backend.

> **Do not benchmark CrisperWhisper through `transformers.pipeline`.** It has no `mode`
> parameter, so verbatim output cannot be requested and the model looks like an ordinary
> cleaned-text ASR. A ten-model comparison run that way produced **zero** disfluencies from
> CW1 and CW2 and led to a wrong conclusion — that no accurate model preserves disfluencies
> and fine-tuning was required. The same weights through the `crisperwhisper` package emit
> 121 filled pauses on the same audio. The pipeline route is also **3x slower** (12x realtime
> against 38x).

### Other models measured

Through `transformers.pipeline`, on the same 12 windows:

| model | filled pauses | notes |
|---|---|---|
| `granite-speech-3.3-2b` | 27 | only seq2seq model to emit them; **drops ~35%** of one window, one empty output |
| `wav2vec2-large-robust-ft-swbd-300h` | 9 + 8 (2 windows) | Switchboard-only fine-tuning; uncased, unpunctuated, mangles proper nouns |
| `granite-speech-3.3-8b` | 10 | **loops on 4 of 12 windows**, up to 71% duplicate 8-grams |
| `Qwen3-ASR-1.7B` | 0 | best WER (4.95%), Apache-2.0, 19x realtime, no looping |
| `granite-speech-4.1-2b` / `-plus` | 0/1 | unsteerable: verbatim prompts change nothing |
| `parakeet-ctc-{0.6b,1.1b}` | 0 | Fisher/Switchboard in a 15-dataset mixture is not enough |
| `nyralabs/CrisperWhisper2.0_large` | 0 | **artefact of the wrong API — see above** |

Two conclusions survive. Disfluency preservation tracks **fine-tuning corpus**, not
architecture or size: `wav2vec2-swbd` (Switchboard as its sole fine-tuning set) emits them
where the identically-sized LibriSpeech checkpoint emits none. And **prompting never works** —
neither Granite 4.1 nor Qwen3-ASR's documented context injection changes disfluency output at
all, though Qwen's context injection *does* fix proper nouns.

### Considered and rejected

- **Reverb ASR** (`Revai/reverb-asr`): `verbatimicity` float 0.0–1.0 confirmed. Requires a
  **fork of WeNet** as its inference stack. Word timestamps **not documented**. Licence "other"
  (Rev Model Non-Production). → D6.
- **Granite Speech 5.0 TurboCTC** (`ibm-granite/granite-speech-5.0-470m-turboctc`): released
  2026-08-25, Apache 2.0, 470 M params, conformer + CTC, 16,384 BPE units, ~60,000 h English,
  non-autoregressive greedy decoding. **No prompting, no timestamps.** → D7.

### Currently-wired weights to replace

`unsloth/crisperwhisper` and `kyr0/crisperwhisper-unsloth-mlx` are third-party redistributions
of CrisperWhisper **v1** with murkier provenance. Move to official `nyralabs` weights.

---

## 4. Architecture

The cut is **"needs audio + GPU" vs. "pure data transformation."**

| | Command | Input | Output | Runs on |
|---|---|---|---|---|
| Stage 1 | `transcribe.py` | audio | raw JSON | HPC (SLURM), A100 |
| Stage 1.5 | `annotate.py` *(optional)* | raw JSON | enriched JSON | anywhere, cheap |
| Stage 2 | `render.py` | JSON | printable transcript | local, instant |

Stage 1 is verbatim and lossless — **no cleaning, no filtering, no speaker assignment.**
Anything recomputable from data is recomputed in stage 2.

### Proposed file layout

```
src/
  transcribe.py            # stage 1 CLI
  annotate.py              # stage 1.5 CLI
  render.py                # stage 2 CLI
  pipeline/
    capabilities.py        # Capabilities dataclass, ModelSpec
    registry.py            # REGISTRY: model_id -> ModelSpec
    adapters/
      base.py              # Adapter ABC
      crisperwhisper2.py
      granite41plus.py
      mock.py              # capability-declaring fake, no weights
    diarize.py             # pyannote wrapper
    align.py               # forced-alignment gap-filler
    schema.py              # raw JSON read/write + validation
  render/
    segment.py             # turns, within-turn anchors, sentences
    speakers.py            # assignment + smoothing, name map
    formats/
      txt.py  plain.py  html.py  tex.py
    templates/
      coding.css  lecture.css
tests/
  fixtures/golden.json     # committed; renderer tests need no models
```

### CLI shapes

```bash
# Stage 1 — SLURM
python src/transcribe.py \
    --input data/interview_003.wav \
    --output-dir transcripts/ \
    --model nyralabs/CrisperWhisper2.0_large \
    --backend ct2 \
    --dual-stream \
    --diarize

# Stage 2 — local, instant, re-runnable
python src/render.py \
    transcripts/interview_003_raw.json \
    --profile coding \
    --format txt,html \
    --speaker-map transcripts/interview_003_speakers.yaml \
    --anchor-interval 30 \
    --pause-threshold 0.6
```

### The capability contract

```python
@dataclass(frozen=True)
class Capabilities:
    word_timestamps: Literal["none", "end_only", "start_end"]
    verbatim:        Literal["no", "yes", "selectable"]   # selectable => mode parameter
    speaker_labels:  bool          # model does its own attribution
    silence_tokens:  bool          # emits explicit pause tokens
    longform:        Literal["native", "needs_chunking"]
    confidence:      bool
```

```python
REGISTRY = {
  "nyralabs/CrisperWhisper2.0_large": ModelSpec(
      adapter=CrisperWhisper2Adapter,
      capabilities=Capabilities(
          word_timestamps="start_end", verbatim="selectable",
          speaker_labels=False, silence_tokens=False,
          longform="native", confidence=True),
      defaults={"backend": "ct2", "mode": "verbatim"}),

  "ibm-granite/granite-speech-4.1-2b-plus": ModelSpec(
      adapter=Granite41PlusAdapter,
      capabilities=Capabilities(
          word_timestamps="end_only", verbatim="no",
          speaker_labels=True, silence_tokens=True,
          longform="needs_chunking", confidence=False),
      defaults={}),
}
```

Orchestrator dispatch is driven entirely by the declaration:

- `word_timestamps == "none"` → run forced alignment (`CrisperWhisperModel.forced_align()`)
- `word_timestamps == "end_only"` → derive starts from the previous token's end.
  **Valid only when `silence_tokens` is true**, else pauses fold into the following word.
- `longform == "needs_chunking"` → segment on silence; otherwise pass the whole file
- `speaker_labels == False` → diarizer required

Adding a model is a registry entry plus, at most, a thin adapter.

---

## 5. Raw JSON schema (v1)

```jsonc
{
  "schema_version": "1.0",
  "source":  { "audio_path": "...", "audio_sha256": "...", "duration_s": 3612.4 },
  "run":     { "created_utc": "...", "device": "cuda:0", "slurm_job_id": "...",
               "pipeline_version": "..." },
  "asr":     { "model_id": "...", "revision": "<hf commit sha>", "backend": "ct2",
               "mode": "verbatim", "capabilities": { ... }, "params": { ... } },
  "diarization": { "model_id": "...", "revision": "...", "params": { ... } },

  "words": [
    { "i": 0, "text": "So",   "start": 0.42, "end": 0.61,
      "timing_source": "native", "speaker_source": null, "conf": 0.98, "flags": [] },
    { "i": 1, "text": "[UM]", "start": 0.79, "end": 1.02,
      "timing_source": "native", "speaker_source": null, "flags": ["filled_pause"] }
  ],

  "speaker_turns": [ { "start": 0.0, "end": 41.2, "speaker": "SPEAKER_00" } ],
  "intended_text": "...",          // second stream when dual-stream requested
  "errors":        [ { "start": 900.0, "end": 1200.0, "message": "..." } ]
}
```

- **`speaker` is absent from words by default.** Diarization turns are stored raw and assigned
  in stage 2, keeping assignment rules tunable. When the ASR supplies its own labels
  (Granite 4.1-plus), words carry `speaker` with `speaker_source: "asr"`.
- **`timing_source`** is `"native"` | `"aligned"` | `"derived"` — a timestamp's origin is part
  of the record, not an assumption.
- **Provenance is not optional.** A coded transcript entering research must state which model,
  revision, and parameters produced it. Cheap now, unreconstructable later.
- Errors preserve the timeline rather than silently dropping it.

### Disfluency flags

`filled_pause` · `repetition` · `repair` · `partial_word` · `vocalization`

Silent hesitation and sound lengthening are **not textual** — recoverable only from word
boundaries, which is the strongest argument for accurate start *and* end times.

CW2 emits `[UH]` / `[UM]`, so filled pauses are machine-identifiable straight out of stage 1.
Richer typing is what `annotate.py` is for; keeping it a separate pass means re-tagging with a
better detector never means re-transcribing. The existing `TextCleaner` BERT model is the right
detector — repurposed from **deleter to tagger**.

---

## 6. Stage 2: rendering

### Structure

- **Turn** is the primary unit — a contiguous stretch by one speaker. Boundary on speaker change.
- **Within-turn anchors** every `--anchor-interval` seconds, snapped to the nearest sentence or
  pause boundary, so a long turn stays citable.
- **Sentence boundary** = terminal punctuation ∨ pause > threshold ∨ speaker change.

#### Measured against real output (2026-08-28)

Two ~80 min recordings, CrisperWhisper word-level timestamps on an A100, `--clean_mode none`:

| Signal | Interview (14,709 w, 182 wpm) | Lecture (11,588 w, 147 wpm) |
|---|---|---|
| terminal punctuation | every 4.7 s | every 7.0 s |
| pause > 0.2 s | every 2.8 s | every 2.8 s |
| **pause > 0.3 s** | **every 5.1 s** | **every 4.5 s** |
| pause > 0.5 s | every 14.5 s | every 12.8 s |
| speaker change | every 43.6 s | every 60.7 s |

**Punctuation is not sparse.** It alone yields a sentence every 4.7–7.0 s, so it is the primary
signal and the pause rule supplements it — the reverse of what was assumed here previously.

**Provisional default: `--pause-threshold 0.3`.** Both recordings agree closely despite
differing genre and speech rate; 0.2 s fragments sentences, 0.5 s merges them.

**Do not harden that number yet.** 59% of inter-word gaps are exactly 0.000 s in both files —
an artifact of `_adjust_pauses` (bug 8), which collapses every gap below `split_threshold`
(0.12 s) and shortens the rest by the same amount. The 0.3 s figure is measured against
already-distorted data and corresponds to roughly 0.42 s of real silence. Re-derive it once
pause adjustment moves to stage 2.

All thresholds are CLI flags. Re-rendering is instant, so tune them against real transcripts
rather than guessing up front.

### Profiles

| | `--profile coding` | `--profile lecture` |
|---|---|---|
| Text stream | verbatim, disfluencies visible | intended (clean) |
| Line numbers | yes — coders cite "line 47" | no |
| Margin | wide right margin for handwritten codes | normal |
| Anchors | dense | sparse |
| Page breaks | avoid splitting a turn | normal |

### Output formats

Plain text is **always** emitted alongside whatever else is requested. Two flavours: `txt`
(turn-structured, speaker labels, timestamps) and `plain` (words only).

| Format | Status | Rationale |
|---|---|---|
| `txt` / `plain` | always | free; greppable, diffable |
| `html` | **default print format** | **zero new dependencies.** CSS `counter-increment` for line numbers, `@page` for the coding margin, `page-break-inside: avoid` for turns. Edit-save-reload iteration — decisive while tuning layout. PDF via Cmd+P; headless export needs weasyprint/playwright, deferred. |
| `tex` | option | Better typography; `lineno` is purpose-built for annotatable numbered lines. Right for a thesis appendix. Not default: slower iteration, and escaping verbatim text is hairy — `% & _ # $ { }` and backslash, plus `[UM]` tokens and partial words like `th-`, must all survive or random interviews fail to compile. |
| `docx` | deferred | Right once QDA import is live (NVivo wants docx; Word has native line numbering). `python-docx` is light, but layout control is weakest. |

### Render stability

Line numbers are stable only while the render is. If a coder annotates "line 47" and the
transcript is later re-rendered with a different pause threshold, line 47 is a different
sentence. In multi-coder work where inter-rater reliability assumes everyone coded the same
text, that is a **correctness problem**, not an inconvenience.

- **The timestamp is the durable anchor**, not the line number. Every subdivision prints its
  timestamp so references survive re-rendering; line numbers serve within-session pointing.
- **Version-stamp every rendered transcript** in the header: model id, revision, render
  parameters, date, short content hash. A printout and the current render then visibly differ.

### Speaker naming

pyannote emits `SPEAKER_00`. The existing transcript already shows real names beside fallbacks
("Dr. Geisler" alongside "Speaker C"), so this step exists informally. Make it an editable map
file read at render time: identify once, re-render forever, fix a misattribution without a GPU.

### OPEN — layout detail

Whether the coding margin should be a right-hand column or double-spaced lines to write
between. Ask the students who mark these up; it is a five-minute CSS change once known.

---

## 7. Local development without SLURM

Most of this pipeline needs no GPU — and no model — at all.

- **Stage 2 is entirely model-free.** Turn grouping, anchors, speaker assignment and smoothing,
  line numbering, print CSS, both profiles, every format backend: a pure function over JSON.
  Generate one real raw JSON, **commit it as `tests/fixtures/golden.json`**, and renderer
  development and tests run in milliseconds forever on any machine.
- **Register a `mock` adapter.** It declares `Capabilities` like any other model and returns
  canned words, testing orchestrator dispatch deterministically in CI — including the paths
  hardest to trigger with real models: `end_only` derivation, forced-alignment gap-fill,
  `needs_chunking`, error-chunk timeline preservation. Register several mocks with differing
  declarations to prove the contract generalises without downloading a single weight.

### Hardware tiers

| Target | Purpose |
|---|---|
| mock adapter + `golden.json` | orchestrator dispatch, all of stage 2, CI. No models, instant. |
| **Docker + CPU on the Mac** | **primary local target for real models.** arm64 containers run natively; no host install, no policy exception. Slow, which does not matter for behavioural questions. |
| RTX 3070 (CUDA, 8 GB, Proxmox VM) | general ML infrastructure (D14). Linux + NVIDIA Container Toolkit puts the GPU inside containers, satisfying the Docker-only rule rather than excepting it. Fits production-size models: CW2 large ≈ 3.1 GB fp16, Granite 4.1-2b ≈ 4 GB bf16, pyannote ≈ 1 GB and sequential. |
| SLURM / A100 | short `--time=00:15:00` prototype jobs **and** full production runs |
| ~~Mac MPS~~ | **not a target** (D13) |

### Why MPS is out (D13)

Docker on macOS cannot expose Metal to Linux containers, so MPS requires a bare-metal install —
a policy exception for a platform that is not, and never will be, a deployment target.
Production is CUDA.

The repo already pays for pursuing it: `CrisperWhisperTranscriber` is pinned to chunk-level
timestamps *because* word-level broke on MPS, and the whole `MLXCrisperWhisperTranscriber` path
is a workaround. **Both become deletable**, so the rewrite gets smaller. Code developed against
CUDA transfers to the A100 cluster; code developed against MPS accumulates workarounds that
transfer to nothing.

### The 3070 VM and this project test each other

Once the GPU VM is provisioned (separate infrastructure task), **this pipeline is the
acceptance test for it** — a real workload exercises far more of the stack than a smoke test.
Concrete criteria it provides:

| Criterion | Why it is a real test |
|---|---|
| CUDA visible in a container | `torch.cuda.is_available()` inside the image, not on the host |
| **CTranslate2 CUDA path works** | Stricter than plain torch — a compiled CUDA library, and the backend production actually uses |
| 8 GB VRAM holds a production model | CW2 large ≈ 3.1 GB fp16 plus activations; confirms the card is usable, not just present |
| HF cache persists across runs | Volume mount correctness; otherwise every run re-downloads 3 GB |
| Bind-mounted audio in, transcripts out | UID/permission mapping — the classic Docker friction point |
| `HF_TOKEN` reaches the container | pyannote needs it; tests secret handling without baking it into the image |
| **Sustained load stability** | A 30-minute transcription is a soak test. Proxmox GPU passthrough problems typically surface under sustained load, not during a quick `nvidia-smi`. |

In return, the VM gives this project the ct2 backend and `transcribe_dual` (D12) — the only
paths that are otherwise production-only — plus fast enough iteration to run Phase 0 at
deployment model size without waiting on CPU.

**Neither blocks the other.** If the VM lands first, run Phase 0 there. If not, the CPU
container path stands.

### A short-walltime SLURM profile

`run_transcription.sh` requests 8-hour GPU jobs, which is what makes the queue slow — long
walltime requests are exactly what schedulers defer. A 90 s clip needs minutes of GPU time, and
a `--time=00:15:00` single-GPU job is the shape backfill scheduling slots in immediately. Add a
`prototype` profile with short walltime and minimal resources; the cluster then becomes viable
for iteration, not only production.

A 3070 real-time factor is **not** a substitute: different memory bandwidth and tensor cores
mean it cannot size an A100 request.

### Two traps when prototyping on small models

1. **Plumbing and behaviour are different questions.** CW2 sizes share one API, so `small`
   validates the pipeline perfectly and says *nothing* about verbatim quality — disfluency
   retention is exactly what degrades with size. Behavioural questions run at deployment size.
2. **Granite's smallest variant may lack the capabilities under test.** Word timestamps and
   speaker attribution appear specific to `-plus`; a smaller non-plus model removes the exact
   capabilities the adapter exists to exercise.

### Toy data

Slice 60–90 s clips (the existing `--start_time` / `--end_time` ffmpeg path does this) and keep
them as standard fixture inputs. **One interview excerpt and one lecture excerpt** — spontaneous
dialogue and monologue stress disfluency handling very differently.

---

## 7b. Proper-noun detection

Neither CW2 nor Qwen3-ASR spells unfamiliar researcher names correctly unaided (`Korshane`,
`Korschen` for *Courchesne*), and a glossary cannot be assembled in advance for an arbitrary
lecture. Detection therefore has to come from the audio, and **cross-model disagreement
localises it** without any prior list.

**Method.** Transcribe with the primary plus three small dissenters from *independent*
lineages, and flag any token where **all** dissenters disagree. The conjunction is what
supplies precision: a single dissenter flags 8.8% of tokens (mostly contraction formatting),
three flag ~1%.

| dissenter | lineage | decoder | RTFx | notes |
|---|---|---|---|---|
| `Audio8/ARK-ASR-0.6B` | AutoArk | causal LM | 36 | **30 s audio cap** — must be called in 30 s slices |
| `nvidia/parakeet-tdt-0.6b-v3` | NVIDIA NeMo | TDT | 182 | |
| `ibm-granite/granite-speech-5.0-470m-turboctc` | IBM | CTC | 7187 | |

Independence matters more than count: `parakeet-ctc` and `parakeet-tdt` share an encoder and
training mixture, and Canary shares them too (its dataset list is byte-identical to
Parakeet's), so their errors correlate. Swapping a Parakeet for ARK bought a real gain
(2.3% -> 1.2% against Qwen).

**Comparison must run on a *fluent view* of the primary.** With CW2 emitting disfluencies the
dissenters cannot match, raw comparison flags every `[UH]`. Before aligning: drop bracketed
tags (case-insensitive), drop trailing-hyphen partial words, collapse adjacent repetitions,
split hyphens/dashes, strip apostrophes, map number words and digits to a single class, and
expand verbatim contractions (`gonna` -> `going to`). Keep an index map so flags report against
the original tokens. Disfluency tokens then cannot be flagged, which is correct: no dissenter
can confirm them.

Measured over 12 x 90 s windows with all three dissenters:

| primary | flagged | note |
|---|---|---|
| Qwen3-ASR | **0.8%** | speaks the same normalised register as the dissenters |
| CrisperWhisper 2 (verbatim) | **2.0%** | residue is largely CW2 transcribing discourse markers ("Okay.", "Right.") the dissenters drop -- i.e. CW2 being *more* complete |

Flags are candidates, not errors: expect real garbles (`Korshane`, `psychatology`, `ipsi`,
`ambiguitonance`) mixed with formatting residue. Resolution is a separate step -- retrieval
against course materials or an author database, then a targeted re-run with `hotwords`. Where
resolution fails, mark the token rather than guessing; a `[NAME?]` flag is worth more to a
coder than a confident wrong spelling.

Cost: ~78% on top of a Qwen primary, ~40% on top of CW2. Batch-only.

---

## 8. Phasing

### Phase 0 — Prototype (CPU container, deployment-size models)

No production code. Answer what documentation cannot:

1. **What does verbatim output actually look like on real audio** — which of the five
   disfluency categories survive?
2. **Granite 4.1-plus: can timestamp mode and speaker-attribution mode be combined in one
   prompt?** If not: two passes, or fall back to pyannote for speakers.
3. Rough real-time factor to size SLURM requests. **Requires the cluster** — neither CPU nor a
   3070 can size an A100 request.

Questions 1–2 are properties of the *model*, not the device: same weights, same output. Both
are fully answerable in a CPU container on the Mac, slowly, at zero policy cost. Run at
deployment model size — they are meaningless on `small`.

### Phase 1 — Stage 1 and the schema

Capability contract, registry, `mock` adapter, CrisperWhisper 2 adapter, raw JSON writer with
full provenance. Keep pyannote. Bypass chunking when `longform == "native"`. Delete the MLX
path and the MPS workaround. Update the SLURM wrapper and add the `prototype` profile.

### Phase 2 — Stage 2 renderer

Speaker assignment with boundary smoothing, turn grouping, within-turn anchors, both profiles,
print CSS, speaker map file, `txt`/`plain`/`html`. Developed entirely against `golden.json`.

### Phase 3 — Second adapter

Granite 4.1-plus. Exercises `end_only` timestamps, silence tokens and native speaker labels,
proving the contract holds rather than having been written around one model.

### Phase 4 — Benchmark *(only if needed)*

Only if Phase 0 leaves the model choice genuinely ambiguous. If so, measure **per-category
disfluency recall**, not WER. Standard ASR evaluation normalises text before scoring, stripping
fillers from *both* hypothesis and reference — so a model that deletes every "um" pays no
penalty and may score better. **Leaderboard WER is structurally incapable of choosing a model
for this project.**

---

## 9. Bugs in the current code

Fix or delete during the rewrite; the first two are costing data today.

1. `max_new_tokens=500` on a 300 s Granite segment (`src/transcribe_audio.py:505`) — five
   minutes of speech far exceeds 500 tokens, so **every chunk's tail is silently truncated**.
2. `Diarizer.run` references `diarization` in its `except` block before assignment — a
   `NameError` that **masks the real exception**.
3. Fixed 300 s boundaries cut mid-word. Moot for CW2 (native long-form); still applies to any
   `needs_chunking` model, which should split on silence.
4. `TextCleaner` runs inline in stage 1 and destroys information before it is written. Moves to
   `annotate.py` as a tagger, not a deleter.
5. Chunks that clean to empty are `continue`d, silently punching holes in the timeline.
6. `_adjust_timestamps` uses absolute `end_time` as the fallback for a missing relative end.
7. `TranscriberFactory` substring-sniffs model names; replaced by the registry.
8. `_adjust_pauses` runs in stage 1 and is lossy. It redistributes every gap below
   `split_threshold` (0.12 s) into the adjacent words and shortens larger gaps by the same
   amount. Measured on two ~80 min recordings, **59% of inter-word gaps come out exactly
   0.000 s** — destroying the pause signal stage 2's sentence rule depends on. Same defect as
   bug 4: pure data transformation belongs in stage 2, not ahead of the write.
9. Per-segment `librosa.load(path, offset=...)` re-decodes a compressed source from byte zero
   for every segment, so decode cost grows with duration. Measured wall-to-wall on the same
   file and GPU: 79 min of `.m4a` took **91 min**, against **57 min** for the same audio
   pre-converted to `.wav` — a **1.6x penalty**, for 3.6 s of ffmpeg. (An earlier draft said
   ~4x; that compared a transcription-only rate against a wall time including diarization.)
   Decode once into a working array, or seek with ffmpeg.
10. `Diarizer` stored `model_name` but **never used it**: both `Pipeline.from_pretrained`
    calls hardcoded `community-1`, so `--diarizer_model` was inert, and the log line announced
    `speaker-diarization-3.1` — a third model, loaded by nobody. Any provenance taken from
    those logs named the wrong component. *(Fixed.)*
11. The Whisper pipeline passed **no loop guards**. Whisper's decoder falls into repetition
    traps: measured across 12 sampled 90 s windows, CrisperWhisper 2 looped on 3 of them,
    duplicating 46-59% of its 8-grams and once emitting "anterior commissure" 37 times in a
    single window. `compression_ratio_threshold` + temperature fallback eliminate it (0.0%
    duplicate 8-grams). Note these exist only in Whisper's *sequential* path and are rejected
    outright by transformers 4.37.2, so they must be capability-gated. *(Fixed.)*
12. The Granite adapter had three independent defects, each fatal: `sampling_rate` passed to
    `GraniteSpeechProcessor.__call__` (which has no such parameter) reached the tokenizer and
    failed every segment; `max_new_tokens=500` truncated anything past ~90 s; and the decode
    read the whole generated sequence, pasting the chat template into the transcript as if it
    were speech. *(All fixed.)*
13. A failed segment became an error placeholder and the run **exited 0**. One lecture lost
    12.7% of its audio this way, visible only by grepping the JSON. The failures are also
    recoverable: the same window fails as a whole and succeeds as two halves, because the
    trigger is where the model's internal 30 s chunk boundaries land. Retry by subdivision
    recovered every lost range; unrecoverable ranges now force a non-zero exit. *(Fixed.)*
14. CTC models must **not** be given `chunk_length_s`. Measured on 300 s where the reference
    is 545 words: `parakeet-ctc-0.6b` returned 63 words chunked and 566 unchunked;
    `granite-470m-turboctc` 68 against 581. Chunking silently discards ~88% of the content.

---

## 10. Open items

| Item | Status |
|---|---|
| **Granite timestamp + speaker mode combination** | Undocumented. Phase 0 question 2. |
| **Coding margin layout** | Right-hand column vs. double-spaced. Ask the student coders. |
| **Verbatim punctuation sparsity** | **Resolved 2026-08-28.** Not sparse — a sentence every 4.7–7.0 s across two ~80 min recordings. Punctuation is the primary signal; `pause > 0.3 s` is the provisional default, to be re-derived after bug 8. See §6. |
| **Diarization speaker count** | **Partly explained 2026-08-31.** Over 10 min excerpts community-1 gives 2 speakers (lecture) and 4 (interview) — plausible. The alarming 10-and-4 counts came from full ~80 min files, so speakers accumulate over duration rather than the model failing outright. Still unverified against the audio. |
| **Diarization model choice** | **Settled 2026-08-31: stay on community-1.** DiariZen v1/v2 benchmarked against it on 10 min excerpts: same speaker counts, near-identical speech totals and runtime (~12 s), 20% fewer/longer segments on the lecture. Not enough to justify its install — a separate env pinned to torch 2.1.1, a vendored pyannote fork, and an `LD_LIBRARY_PATH` workaround for a Rocky 8 `libstdc++` mismatch. Both DiariZen checkpoints are CC-BY-NC; community-1 is CC-BY-4.0. No DER computed: no reference labels. |
| **Fine-tuning for disfluencies** | **Dropped 2026-08-31.** Premised on no accurate model emitting disfluencies, which was an artefact of calling CrisperWhisper through `transformers.pipeline`. `mode="verbatim"` supplies them natively. The teacher-forced groundwork (Qwen3-ASR sits at ~1% filler probability against CW2's ~0.1%) is recorded in case the premise returns. |
| **Vocalization coverage** | Only `[laughter]`, once, in 12 windows of lecture + interview audio. Whether CW2 tags coughs/breaths is untested — this material may simply contain none. |
| **Diarization at turn boundaries** | Short filler tokens near a speaker change are where max-overlap assignment is noisiest; hence smoothing within pause-bounded runs. |
| **IRB / data governance** | Interview recordings are human-subjects data and stage 1 moves them to shared cluster storage. Assumed covered by the existing protocol; flagged because this pipeline automates the transfer. |
| **RTX 3070 VM provisioning** | Separate infrastructure task, in progress in another context. Not a blocker (D14). |

---

## 11. Next action

**Phase 0.** Two viable routes, whichever is ready first:

- **CPU Docker container on the Mac** — available now. Dockerfile with
  `crisperwhisper[transformers]`, `transformers`, `pyannote.audio`, `librosa`, `ffmpeg`.
- **RTX 3070 VM** — once provisioned. Use `crisperwhisper[ct2]` instead, which additionally
  exercises the production backend and `transcribe_dual`, and serves as the VM's acceptance
  test (§7).

Both need a 60–90 s interview clip and a 60–90 s lecture clip, plus `HF_TOKEN` from `.env` for
pyannote. No host installs either way. Nothing in `src/` changes until Phase 1.

---

## 12. Superseded work

`docs/granite_word_timestamps_plan.md` proposed extracting word timestamps from Granite by
finding monotonic attention heads and training an auxiliary transformer on LibriSpeech
alignments — an 11–21 hour estimate that was optimistic, required GPU training and ground-truth
alignments, and predicted **start times only**.

It is unnecessary. Granite Speech 4.1-2b-plus (released 2026-04-28, *after* that plan was
written) provides word-level timestamps natively, with explicit silence tokens. Keep the file
for provenance with a superseded header rather than deleting it.
