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
  Verbatim text has sparse punctuation, so the pause rule carries more weight in the coding
  profile than the lecture one.

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

---

## 10. Open items

| Item | Status |
|---|---|
| **Granite timestamp + speaker mode combination** | Undocumented. Phase 0 question 2. |
| **Coding margin layout** | Right-hand column vs. double-spaced. Ask the student coders. |
| **Verbatim punctuation sparsity** | Sentence subdivision may lean mostly on pauses. Check against real output before fixing a default. |
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
