# Transcription Pipeline — Plan of Attack

**Status:** Phase 1 substantially complete — stage 1 rewritten under `src/pipeline/`.
Phase 2 (the renderer) is the next block of work.
**Last updated:** 2026-09-01
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

Stage 1 has been rewritten to the §4 layout. `src/transcribe_audio.py` is now a thin CLI over
`src/pipeline/`: `capabilities.py` (the contract), `registry.py`, `adapters/`
(`crisperwhisper2`, `whisper`, `mock`), `chunking.py`, `flags.py`, `diarize.py`,
`provenance.py`, `schema.py`, `orchestrator.py`, `preview.py`. `tests/check_contract.py` runs
31 dependency-free checks anywhere.

Gone: `TranscriberFactory`'s name-sniffing, the MLX path, `src/transcribe_audio_mac.py`, MPS
device selection, and stage-1 text cleaning.

Not yet written: `annotate.py` (stage 1.5), `render.py` and everything under `src/render/`
(stage 2, Phase 2), `align.py`, and the Granite adapter (Phase 3).

Sample data: `data/geisler.wav`, `data/251211_0009.wav`, `data/tate_1.m4a` (see
`docs/environments.md`). The old output `transcripts/lecture transcript.txt` — fully
normalized prose, zero disfluencies, no timestamps, multi-thousand-word paragraph blocks — is
what this work replaces: fine as lecture notes, **unusable for coding**.

---

## 2. Decision log

Settled. Each entry records what would reopen it.

| # | Decision | Why | Reopens if |
|---|---|---|---|
| D1 | **Split into three commands**: `transcribe` → `annotate` → `render` | Stage 1 queues on SLURM; stage 2 must be instant. Never re-queue a cluster job to change a pause threshold. | — |
| D2 | **Capability contract, not a model-name factory** | Plug-and-play comes from adapters declaring what they provide, with the orchestrator filling gaps. | — |
| D3 | **Stage 1 is verbatim and lossless.** No cleaning, filtering, or speaker assignment. | Cleaning in stage 1 destroys information before it is ever written to disk. | — |
| D17 | **Drive CrisperWhisper 2 through the `crisperwhisper` package, never `transformers.pipeline`.** | The package exposes `mode="verbatim"` (default), `hotwords`, `temperature_fallback` and `word_timestamps`. The pipeline exposes none of them, silently yields cleaned text, and runs 3x slower (12x vs 38x realtime). Measured: 121 filled pauses vs 0 on identical audio. | — |
| D20 | **Use `mamba` for all environment management, never `conda`.** | Faster resolution, and `src/transcribe.slurm` already activates through it — mixing the two invites drift between what a job activates and what was built. Build fresh rather than `--clone`: clones hardlink, so pip in a clone can strip packages from the source env. | — |
| D21 | **Hotwords are a declared capability, and the standard CrisperWhisper 2 checkpoints declare `untrained`.** | Hotword boosting is trained into the Pro checkpoints only. On a standard checkpoint the package accepts the argument, raises a `UserWarning` and can *degrade* transcription -- measured, it did (§3). A run may still pass them; it warns, and the warning is written into the record. | Nyra documents hotword training for the standard weights, or a Pro licence is bought |
| ~~D22~~ | **Withdrawn 2026-09-01, the same day it was added.** It claimed proper-noun spelling should come from the `intended` stream. The measurement behind it was wrong (§3): a substring regex scored `Courchesney` as a match for `Courchesne`. The number is not reused; the open question is tracked in §10. | — | — |
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
| D12 | **Dual-stream output is an adapter-internal optimisation, not an architecture** | `transcribe_dual()` is ct2-only; falling back to two sequential `transcribe()` calls keeps the *semantics* testable on any backend. **Implemented and verified 2026-09-01** — both routes produce byte-identical text (§3). | — |
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

`word_timestamps=True` and the markers coexist. `temperature_fallback=True` is
on by default and no looping was observed (0.0% duplicate 8-grams on all 12 windows).
Throughput **38x realtime** via the CTranslate2 backend.

#### Hotwords are Pro-only, and they cost something (measured 2026-09-01)

The package's own docstring says hotword boosting is trained into the **Pro** checkpoints
only: on a standard model it "can degrade transcription rather than merely doing nothing",
and `_warn_if_hotwords_unsupported()` raises a `UserWarning` for any non-`_pro` v2 id. The
default `nyralabs/CrisperWhisper2.0_large` is standard, so every `--hotwords` run so far has
been in that state, with the warning going to stderr unrecorded.

An earlier note here recorded only the upside. Re-measured on `geisler.wav` 45:00-46:30, four
runs, deterministic (a repeat was byte-identical):

| hotwords | tokens | `Courchesne` | `Korsh*` | `commissure` |
|---|---|---|---|---|
| none | 166 | 0 | 4 | 2 |
| `Courchesne` | 166 | **4** | 0 | 1 -- one became `commasure` |
| `Courchesne,commissure,Geisler` | 165 | **4** | 0 | 1 -- `commasure` **deleted** |
| (repeat of the above) | 165 | 4 | 0 | 1 |

So biasing does fix the target name, and it damaged a **different** word in the same window:
`commissure` -> `commasure` with only `Courchesne` in the list. Adding `commissure` to the
list did not restore it -- it removed the token altogether. Disfluency markers were unaffected
throughout (10 filled pauses in every run).

One 90 s window, so the direction is measured but not established. What is established is that
the previous entry recorded the benefit without checking the cost.

#### A fourth instance of D19 — this one self-inflicted (2026-09-01)

An earlier version of this section claimed **"`--mode intended` spelled `Courchesne`
correctly 4/4 with no hotwords at all"**, and D22 was written on that basis: take proper-noun
spelling from the intended stream rather than from hotwords.

It was wrong. The check counted occurrences with the regex `Courchesne`, which **matches
inside `Courchesney`**. Re-run against exact token forms:

| run | exact `Courchesne` | what the tokens actually are |
|---|---|---|
| verbatim, no hotwords | 0 | `Korshane` x2, `Korshesney` x2 |
| **`--mode intended`** | **0** | **`Courchesney` x4** |
| **`--hotwords Courchesne`** | **4** | `Courchesne` x4 |

So the intended stream is *closer* than verbatim — it recovers the French shape of the name —
but it is still wrong, consistently, four times out of four. **Hotwords produced the only
correct spelling in any run.**

That leaves a genuine trade-off with no clean side, which is why D22 was withdrawn rather
than reversed:

- **hotwords** get the target name exactly right, and damage a neighbouring word
  (`commissure` -> `commasure`, then deleted when the list grew);
- **the intended stream** costs nothing, damages nothing, and still misspells the name.

The failure shape is the one §3 opens by warning about, committed here in this project's own
measurement code rather than in a model integration: plausible output, a self-consistent
number, a wrong conclusion, and no error message anywhere. The generalisation worth keeping is
that **D19 applies to the verification as much as to the integration** — a substring test
where an exact test was meant is the measurement-side equivalent of a generic loader that
runs. Proper-noun checks in this project compare whole tokens.

#### Dual-stream costs ~11%, not 100% (measured 2026-09-01)

`transcribe_dual()` decodes both modes in one batched pass: they differ only by the decoder
prompt prefix and share the encoder output, so the autoregressive decode runs once for both
and each mode's word-timing cross-attention is captured inline. It is **ct2-only and v2-only**
— it raises `NotImplementedError` otherwise — and only `longform_strategy="continuation"` is
supported for it.

Measured on `geisler.wav` 45:00–46:30, A100, warm cache, two trials each. `processing_time`
is the package's own inference timer, not wall clock:

| run | inference | RTFx |
|---|---|---|
| single verbatim | 2.451 s / 2.390 s | 36.7 / 37.7 |
| dual (both streams) | 2.791 s / 2.595 s | 32.2 / 34.7 |

The second transcript costs **~11%** more inference time (2.42 s → 2.69 s mean), against ~100%
for a second sequential pass — so roughly **1.8× faster than two passes**, matching the ~1.9×
the package documents. Both streams report the same `processing_time`, which is what one
shared pass should look like.

**Equivalence holds.** The dual run's verbatim stream is byte-identical to a separate
`--mode verbatim` run (166 tokens), and its intended stream byte-identical to a separate
`--mode intended` run (154 tokens). Timestamps differ on **2 of 332 bounds, by at most
0.120 s** — the package documents exactly this: batching two rows is mathematically identical
per row but differs at the ULP level, which can flip a rare near-tie token or timing on
long-form audio. Which route ran is recorded in `asr.params.dual_route`.

Flag counts confirm the modes did what they claim: 10 filled pauses in the verbatim stream,
**zero** in the intended one. Tagging the second stream is a check, not decoration — a
non-zero count there would mean `intended` had not cleaned anything.

> **Do not use wall clock to size a SLURM request.** The first attempt at this comparison
> read 19.1 s against 9.7 s and looked like dual costing 2×. Both figures were dominated by
> model load and a sha256 over the source audio, and the dual run happened to go first and pay
> a cold cache. The decode phases were ~3 s and ~2 s. `asr.performance` now records the
> model's own timer for this reason.

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

### Currently-wired weights — resolved 2026-08-31

The default is now `nyralabs/CrisperWhisper2.0_large`. The v1 redistributions
(`unsloth/crisperwhisper`, `kyr0/crisperwhisper-unsloth-mlx`) are not merely lower-provenance:
on v1 the package **silently ignores `mode` and `hotwords` and emits no disfluency markers**.
It warns, and the run still succeeds — verified by getting byte-identical 158-word output for
`verbatim`, `verbatim+hotwords` and `intended`. Passing a v1 checkpoint disables the features
this pipeline depends on without failing.

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

### File layout

`[x]` exists, `[ ]` planned.

```
src/
  [x] transcribe_audio.py      # stage 1 CLI (keeps its name: the SLURM wrappers call it)
  [ ] annotate.py              # stage 1.5 CLI
  [ ] render.py                # stage 2 CLI
  pipeline/
    [x] capabilities.py        # Capabilities, ModelSpec, plan_for()
    [x] registry.py            # REGISTRY: model_id -> ModelSpec, exact match only
    adapters/
      [x] base.py              # Adapter ABC, Word, AdapterResult, ErrorRange
      [x] crisperwhisper2.py
      [x] whisper.py           # generic openai/whisper-* only
      [x] mock.py              # six capability-declaring fakes, no weights
      [ ] granite41plus.py     # Phase 3
    [x] chunking.py            # fixed-window segmentation + retry by subdivision
    [x] flags.py               # filled_pause / vocalization / partial_word tagging
    [x] diarize.py             # pyannote wrapper
    [x] provenance.py          # hashes, revisions, package + git versions
    [x] schema.py              # raw JSON build/read/write + validate
    [x] orchestrator.py        # dispatch from the declaration; assembles the record
    [x] preview.py             # stage-1 text preview, superseded by render.py
    [ ] align.py               # forced-alignment gap-filler
  [ ] render/                  # Phase 2: segment, speakers, formats, templates
tests/
  [x] check_contract.py        # 31 dependency-free checks
  [ ] fixtures/golden.json     # committed; renderer tests need no models
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

**Implemented** in `src/pipeline/capabilities.py`.

```python
@dataclass(frozen=True)
class Capabilities:
    word_timestamps: Literal["none", "end_only", "start_end"]
    verbatim:        Literal["no", "yes", "selectable"]   # selectable => mode parameter
    speaker_labels:  bool          # model does its own attribution
    silence_tokens:  bool          # emits explicit pause tokens
    longform:        Literal["native", "needs_chunking"]
    confidence:      bool
    hotwords:        Literal["no", "untrained", "trained"] = "no"   # D21
```

`hotwords` is the one field this sketch did not originally have. It exists because
"the parameter is accepted" and "the parameter works" turned out to be different
claims, and only the second is worth acting on -- exactly the D19 shape, found by
reading the package source rather than by the run failing.

```python
REGISTRY = {
  "nyralabs/CrisperWhisper2.0_large": ModelSpec(
      adapter="pipeline.adapters.crisperwhisper2:CrisperWhisper2Adapter",
      capabilities=Capabilities(
          word_timestamps="start_end", verbatim="selectable",
          speaker_labels=False, silence_tokens=False,
          longform="native", confidence=False, hotwords="untrained"),
      defaults={"backend": "auto", "mode": "verbatim"}),

  # Registered refusals. A checkpoint that runs and quietly does the wrong
  # thing is worse than one that will not run at all.
  "unsloth/crisperwhisper": ModelSpec(unsupported="CrisperWhisper v1: ..."),
  "kyr0/crisperwhisper-unsloth-mlx": ModelSpec(unsupported="MLX, D13: ..."),
  "ibm-granite/granite-speech-4.1-2b-plus": ModelSpec(
      unsupported="Adapter not implemented yet (Phase 3). Declared shape: ..."),
}
```

Adapters are named by import path and loaded lazily, so the registry imports on a
machine with no ML stack -- otherwise the mock adapters cannot do their job.
`load_adapter_class()` compares the adapter's own declaration against the registry's
and refuses on drift: the two live in different files, and a stale capability
misroutes exactly the way a stale name pattern did.

Three rules retire bug 7 rather than merely discouraging it:

1. **Lookup is exact**, case-folded to match how the Hub resolves ids. An unknown id
   is refused with the supported list, never guessed at.
2. **Known-bad checkpoints are registered refusals** carrying their reason.
3. **The adapter re-verifies on load.** `CrisperWhisper2Adapter` calls the package's
   own `detect_model_version()`, which reads the tokenizer's `[verbatim_1]` marker
   without loading weights -- so a renamed or mirrored copy of v1 weights is caught
   too, which a denylist alone would miss.

`--adapter` forces an adapter for an unregistered checkpoint. It is deliberately
explicit and is recorded as `asr.adapter_source: "override"`, so an unusual routing
decision is visible in the record instead of inferred from a name at runtime.

Orchestrator dispatch is driven entirely by the declaration (`plan_for()`, a pure
function from a declaration to the work required, which is what makes it testable
against the mocks):

- `word_timestamps == "none"` → run forced alignment (`CrisperWhisperModel.forced_align()`)
- `word_timestamps == "end_only"` → derive starts from the previous token's end.
  **Valid only when `silence_tokens` is true**, else pauses fold into the following word.
- `longform == "needs_chunking"` → segment on silence; otherwise pass the whole file
- `speaker_labels == False` → diarizer required

Adding a model is a registry entry plus, at most, a thin adapter.

---

## 5. Raw JSON schema (v1)

**Implemented** in `src/pipeline/schema.py`, with `validate()` checking structure,
word index monotonicity, timing ordering and the speaker/speaker_source coupling.

```jsonc
{
  "schema_version": "1.0",
  "source":  { "audio_path": "...", "audio_sha256": "...", "duration_s": 90.0,
               // present only for a --start_time run: the working excerpt is a
               // temp file that is deleted, so the *original* is named and hashed
               "excerpt": { "start_time": "00:45:00", "end_time": "00:46:30",
                            "offset_s": 2700.0, "timestamps_relative_to": "excerpt" } },
  "run":     { "created_utc": "...", "device": "cuda", "hostname": "...",
               "python": "3.11.x", "slurm_job_id": "...", "slurm_partition": "...",
               "pipeline_version": { "commit": "...", "branch": "...", "dirty": false },
               "packages": { "crisperwhisper": "2.0.2", "transformers": "...", ... } },
  "asr":     { "model_id": "...", "revision": "<hf commit sha>",
               "revision_source": "local cache",
               "adapter": "CrisperWhisper2Adapter", "adapter_source": "registry",
               "backend": "ct2", "granularity": "word", "timing_source": "native",
               "capabilities": { ... }, "params": { "mode": "verbatim", ... },
               "performance": { "processing_time_s": 2.79, "realtime_factor": 32.2 } },
  "diarization": { "model_id": "...", "revision": "...", "params": { ... } },

  "words": [
    { "i": 0, "text": "So",   "start": 0.42, "end": 0.61, "timing_source": "native",
      "speaker": null, "speaker_source": null, "conf": null, "flags": [] },
    { "i": 1, "text": "[UM]", "start": 0.79, "end": 1.02, "timing_source": "native",
      "speaker": null, "speaker_source": null, "conf": null, "flags": ["filled_pause"] }
  ],

  "text":          "...",          // the model's own full text
  "speaker_turns": [ { "start": 0.0, "end": 41.2, "speaker": "SPEAKER_00" } ],
  "secondary_stream": {            // present only with --dual_stream
      "mode": "intended", "text": "...", "words": [ ... ] },
  "errors":        [ { "start": 900.0, "end": 1200.0, "message": "..." } ],
  "warnings":      [ "hotwords were supplied to a checkpoint that was never ..." ]
}
```

Three fields beyond the original sketch, each earned by something that went wrong:

- **`revision_source`** — the local HF cache is consulted first, because it reports the
  revision that *was used* and needs no network on a compute node. The Hub API is a
  fallback and reports the current head, which is not the same claim. Saying which one
  answered is the difference between provenance and decoration; the same goes for
  `pipeline_version.dirty`.
- **`granularity`** — `"word"` or `"chunk"`. `--timestamp_mode chunk` is a VRAM concession
  for 8 GB cards where word-level alignments OOM. It coarsens the record, and stage 2
  cannot be left to infer which it is reading.
- **`warnings`** — the caveats were going to stderr and vanishing with the job log. The
  untrained-hotword `UserWarning` (D21) is captured from the package and written here.
- **`secondary_stream`** replaces the sketch's text-only `intended_text`. It carries word
  timestamps, which D10 requires — both render profiles carry timestamps, and the lecture
  profile renders the *intended* stream — and it is keyed by `mode` rather than by role, so
  the record never has to assume which stream is primary.
- **`performance`** — the model's own inference timer and the realtime factor it implies.
  Phase 0 question 3 asks for exactly this, and wall clock cannot answer it (§3).

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
against course materials or an author database, then a targeted re-run. Where resolution
fails, mark the token rather than guessing; a `[NAME?]` flag is worth more to a coder than a
confident wrong spelling.

**How the re-run should resolve a flagged token is open** (§10). Measured 2026-09-01 on the
45:00 window, `hotwords` gave the only exactly-correct spelling of the probe name, and cost a
neighbouring word to do it; `--mode intended` damaged nothing but still misspelled the name
(`Courchesney`). Neither is a clean resolution step. The intended stream remains cheap enough
to take from every dual-stream run and is worth keeping as evidence, but it does not settle a
spelling on its own.

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

**Done (2026-08-31):** the CrisperWhisper 2 adapter — driving the `crisperwhisper` package
directly instead of subclassing the transformers path, bypassing the 300 s segmentation via
native long-form, and exposing `--mode` and `--hotwords`.

**Done (2026-09-01):** capability contract, registry, `mock` adapters, raw JSON schema v1
with provenance, and the MLX/MPS deletion. Laid out as §4 proposed, under `src/pipeline/`.

Verified on `geisler.wav` 45:00–46:30 (A100, ct2 backend, `--no_diarize`):

| run | tokens | filled pauses | untimed tokens | `Courchesne` |
|---|---|---|---|---|
| `--mode verbatim` | 166 | **10** | **0** | 0 (`Korshane`, `Korshesney` ×4) |
| `--mode verbatim --hotwords ...` | 165 | 10 | 0 | 4 — and one unrelated word damaged, see §3 |
| `--mode intended` | 154 | 0 | 0 | 4 |

Ten markers on a 90 s window, per-word start and end on **every** token including the
markers, and `intended` suppressing them: the reference expectations, reproduced through
the new dispatch path. The three registry refusals (`unsloth/crisperwhisper`,
`kyr0/crisperwhisper-unsloth-mlx`, and `--mode verbatim` against `openai/whisper-large-v3`)
each exit 2 with the reason. `tests/check_contract.py` — 31 checks, no dependencies —
covers the dispatch decisions on both the Mac and the cluster.

**Still open in this phase:** the SLURM `prototype` profile, and the forced-alignment
gap-filler (`align.py`), which the contract dispatches to but which is not built — no
registered real model declares `word_timestamps="none"` yet, and the run records the gap
as a warning rather than pretending to have timings.

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

### Fixed (2026-08-27 → 09-01)

Kept here for the reasoning, not the code. Each was found by running real audio, not by
reading source.

| # | Defect | Why it mattered |
|---|---|---|
| 1 | `max_new_tokens=500` on a 300 s Granite segment | ~950–1200 tokens of speech at measured rates, so every segment's tail was silently truncated |
| 8 | `_adjust_pauses` ran during transcription | Collapsed every gap under 0.12 s and shortened the rest — **59% of inter-word gaps came out exactly 0.000 s**, destroying the silence durations sentence segmentation reads. Now opt-in; moot for CW2, which is never routed through it |
| 10 | `Diarizer` ignored `model_name` | Both `from_pretrained` calls hardcoded `community-1`, so `--diarizer_model` was inert while the log named a third model nobody loaded |
| 11 | No Whisper loop guards | 3 of 12 windows looped, duplicating 46–59% of 8-grams; one produced 542 words (~360 wpm) with "anterior commissure" ×37. Moot for CW2 (native `temperature_fallback`), still needed for generic Whisper |
| 12 | Three Granite adapter defects | `sampling_rate` to a processor without that parameter; 500-token cap; decode read the prompt back, pasting the chat template in as speech |
| 13 | Failed segments exited 0 | One lecture lost **12.7%** of its audio, visible only by grepping the JSON. Retry-by-subdivision recovered every range (11,588 → 13,193 words); unrecoverable ranges now force a non-zero exit |
| 15 | `mlx_whisper` and `pyannote` imported at module scope | An optional backend took the whole script down — the second cost a debugging cycle *after* the first was fixed |
| 16 | Default model was `unsloth/crisperwhisper` (**v1**) | v1 silently ignores `--mode` and `--hotwords` and emits no markers. It warns; the run still succeeds |
| 2 | `Diarizer.run` referenced `diarization` in its `except` before assignment | A `NameError` replaced the real exception **every time**, so no diarization failure ever reported its own cause |
| 4 | `TextCleaner` ran inline in stage 1 | Destroyed information before it was written to disk. Now affects the preview text only; the JSON is always verbatim (D3) |
| 5 | Chunks that cleaned to empty were `continue`d | Punched silent holes in the timeline. Moot: nothing is dropped from the record |
| 6 | `_adjust_timestamps` used the absolute `end_time` as the fallback for a missing relative end | Invented a timestamp and hid the gap from everything downstream. A missing bound now stays `None` |
| 7 | `TranscriberFactory` substring-sniffed model names | `"crisper" in name` routed **v1** into the v2 path, where `--mode`/`--hotwords` are ignored and the run exits 0. `"mlx" in name` and `"granite-speech" in name` still referenced classes deleted in e0e2542, so either id raised `NameError`. Replaced by the registry |
| 17 | The MLX path survived D13 | Deleted, with `src/transcribe_audio_mac.py` and MPS device selection. The chunk-level pin it motivated went in e0e2542; the surviving `--timestamp_mode` flag is generic-Whisper-only and has its own justification (`docs/environments.md`), so it is kept and recorded as `asr.granularity` |

**The recurring shape:** a module-scope import of an optional dependency (15), a silent
success that disables a feature (16, 7), and a failure that exits 0 (13). None announced
itself. The contract's job is to make each shape structurally impossible rather than
individually remembered: refuse instead of guessing, verify the checkpoint instead of
trusting its name, and put every caveat in the output file instead of a job log.

### Outstanding

Bugs 3 and 9 both live in `src/pipeline/chunking.py` now, documented in its module
docstring. Neither is fixable without a real `needs_chunking` model to verify against,
which arrives in Phase 3.

3. Fixed 300 s boundaries cut mid-word. Bypassed for CrisperWhisper (native long-form) but
   still applies to any `needs_chunking` model, which should split on silence.
9. Per-segment `librosa.load(path, offset=...)` re-decodes a compressed source from byte zero.
   Measured wall-to-wall: 79 min of `.m4a` took **91 min** against **57 min** for the same
   audio as `.wav` — a **1.6x penalty** for 3.6 s of ffmpeg. Bypassed for CrisperWhisper,
   still live for chunked models. *(An earlier draft said ~4x; that compared a
   transcription-only rate against a wall time including diarization.)*
14. CTC models must **not** be given `chunk_length_s` — not a bug in this code, a calling
    rule. On 300 s where the reference is 545 words, `parakeet-ctc-0.6b` returned 63 chunked
    and 566 unchunked; `granite-470m-turboctc` 68 against 581. Chunking discards ~88% of the
    content silently.

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
| **Proper nouns: how to resolve a flagged token** | **Opened 2026-09-01, and genuinely open.** On the 45:00 window, `hotwords` produced the only exactly-correct spelling of the probe name and corrupted a neighbouring word doing it; `--mode intended` corrupted nothing and still got the name wrong (`Courchesney` x4). §7b's resolution step therefore has no good default. Worth measuring over the 12-window sweep, comparing **exact token forms**, not substrings: how often does each route land the correct spelling, and how often does hotword biasing damage a non-target token? D21's cost is established; the benefit's reliability is not. |
| **Forced alignment (`align.py`)** | Dispatched to but not built. No registered real model declares `word_timestamps="none"`, so nothing needs it yet; the run records the gap in `warnings` rather than emitting untimed words that look timed. Build it when a model needs it, using `CrisperWhisperModel.forced_align()`. |
| **CrisperWhisper `confidence`** | Declared `False`. `TranscriptionResult` has no per-word confidence field, so `conf` is always null. Worth revisiting if a later package version exposes one — coders benefit from knowing which tokens the model was unsure of. |

---

## 11. Next action

**Phase 2: the renderer.** Stage 1 now emits schema v1, so the highest-value next step is to
generate one real raw JSON from a full recording and commit it as `tests/fixtures/golden.json`
(plan §7). Every part of stage 2 — turn grouping, within-turn anchors, speaker assignment and
smoothing, line numbering, print CSS, both profiles, all format backends — is then a pure
function over that file, developed and tested on any machine in milliseconds with no GPU.

Two smaller things worth doing alongside, both cheap:

- **The `prototype` SLURM profile.** `run_transcription.sh` still requests 8-hour GPU jobs,
  which is the shape a scheduler defers. A `--time=00:15:00` single-GPU job is what backfill
  slots in immediately, and the verification runs in §8 show minutes of GPU time is all a
  90 s window needs.
- **Re-derive the pause threshold.** Bug 8 is fixed, so the 0.3 s default was measured against
  distorted data and needs redoing against a current run (§6).

Phase 0's remaining question — whether Granite 4.1-plus can combine timestamp and
speaker-attribution modes in one prompt — is Phase 3's problem, and the registry already holds
its declared shape so the adapter has a target to hit.

---

## 12. Superseded work

`docs/granite_word_timestamps_plan.md` proposed extracting word timestamps from Granite by
finding monotonic attention heads and training an auxiliary transformer on LibriSpeech
alignments — an 11–21 hour estimate that was optimistic, required GPU training and ground-truth
alignments, and predicted **start times only**.

It is unnecessary. Granite Speech 4.1-2b-plus (released 2026-04-28, *after* that plan was
written) provides word-level timestamps natively, with explicit silence tokens. Keep the file
for provenance with a superseded header rather than deleting it.
