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
| D23 | **The coding margin is a right-hand column.** | Decided 2026-09-01 without polling the coders: a wide right margin is the conventional layout for handwritten codes on a printed transcript, it is a five-minute CSS change either way, and deciding it lets Phase 2 proceed. | The student coders ask for double-spaced lines to write between instead — or anything else. **Explicitly not a point of no return.** |
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
> | ARK-ASR-0.6B | cast every float tensor to fp16, feed 90 s | cast **only** `inputs["audios"]`; `audio_max_length=30*16000`; `do_sample=False`; `bad_words_ids` | output collapsed into repeated CJK characters; nearly discarded a working model. Card script re-verified 2026-09-01 (§7b) |
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
- **N is modulo 1000 — the tag rolls over every 10 seconds.** (Re-read 2026-09-01; the
  entry above had omitted this.) "To reduce the amount of generated tokens, only the last three
  digits are provided": `N = round(t·100) mod 1000`, and `t = N/100 + 10R` with R the rollover
  count. The card's unwrap is `while end + offset < last_end: offset += 10`. An adapter that
  reads N as plain centiseconds is correct for exactly the first ten seconds of every window.
- **Silences are transcribed as `_` with their own end timestamp.** This is why `end_only` is
  usable: with explicit pause tokens, `word_start = previous_token_end` is correct and pause
  durations are recoverable.
- **Length limits:** "works well with audio segments up to 9 minutes for ASR and SAA, and up
  to 3.5 minutes for timestamps." So `longform="needs_chunking"` with a **≤210 s window in
  timestamp mode** — smaller than the 300 s `SEGMENT_SIZE` the chunker defaults to.
- Exact prompts (the card's; verbatim matters for a prompt-steered model):
  - ASR: `<|audio|> can you transcribe the speech into a written format?`
  - SAA: `<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding
    [Speaker 1]: and [Speaker 2]: tags before speaker turns.`
  - TS: `<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag
    showing the end time in centiseconds, e.g. hello [T:45] world [T:82]` — with
    `max_new_tokens=10000`
  - KWB: append `Keywords: …` to any prompt; terms absent from the audio are tolerated.
- Invocation: `AutoProcessor` + `AutoModelForSpeechSeq2Seq`, bf16, a **fixed** system prompt
  (`Today's Date: December 19, 2024` — the card's, not the wall clock), chat template with
  `add_generation_prompt=True`, `generate(do_sample=False, num_beams=1)`, decode only the new
  tokens. Native in `transformers>=5.8`.
- **Incremental decoding:** `apply_chat_template(..., prefix_text=previous_transcript)` — the
  template emits the prefix right after the generation prompt (confirmed in the snapshot's
  `chat_template.jinja`), and the *audio accumulates* too. The card names it as the way "to
  maintain the speaker numbering in SAA mode" across segments, which means **speaker numbers
  restart per chunk without it** — they are ordinals by first appearance, not identities.
- Speaker attribution mode: emits `[Speaker 1]:` / `[Speaker 2]:` tags, numbered by first appearance
- ~~The plus variant does not produce punctuation or capitalisation~~ **Wrong — it depends
  on the mode** (measured 2026-09-01). ASR and SAA output is punctuated and capitalised
  (`"Absolutely. Yeah. In fact, you're alluding to…"`); **timestamp mode is lowercase and
  unpunctuated** (`um [T:84] including [T:130] bob [T:160] knight`). Since the timestamped
  words are what the record carries, stage 2's punctuation rule gets little from this model
  and the silence tokens carry the segmentation instead.
- No verbatim or disfluency claim anywhere on the card — but **timestamp mode emits some
  fillers** where ASR mode emits none: 5 `uh`/`um` on the 45:00 reference window (CrisperWhisper
  verbatim: 10; Granite ASR/SAA mode: 0). Every word needing a tag seems to make skipping one
  harder. Not enough to revisit D4; enough to note.
- ~2 B params ≈ 4 GB bf16; snapshot `1454e6e1` cached on POLARIS
- **Phase 0 question 2 — answered 2026-09-01: the modes do not combine.** Two fusions of the
  documented prompts, on the reference window and on a multi-speaker window where SAA alone
  found two speakers and three turns, both produced timestamp-mode output with a single
  vestigial `[Speaker 1]:` at the start and no attribution. The adapter therefore runs two
  passes per window and aligns them (§8, Phase 3).

#### Phase 3 spike (2026-09-01, A100, `audio-transcribe-tf5`, two 90 s windows)

| window | mode | gen | tokens | speaker tags | `[T:N]` tags | silences | unwrap monotonic | last end |
|---|---|---|---|---|---|---|---|---|
| geisler 45:00 | ASR | 4.4 s | 212 | — | — | — | — | — |
| geisler 45:00 | SAA | 4.0 s | 217 | 1 (`Speaker 1`) | — | — | — | — |
| geisler 45:00 | **TS** | 22.7 s | 1270 | — | **206** | **42** | **yes** | 85.05 s |
| geisler 45:00 | combo ×2 | 22 s | ~1260 | 1 (vestigial) | ~204 | ~40 | yes | ~85 s |
| proseminar 11:30 | SAA | 6.0 s | 335 | **3** (`Speaker 1, 2, 1`) | — | — | — | — |
| proseminar 11:30 | **TS** | 33.6 s | 1906 | — | **313** | **45** | **yes** | 88.25 s |
| proseminar 11:30 | combo ×2 | 34 s | ~1918 | **1** (vestigial) | ~314 | ~46 | yes | ~88 s |

Timestamp mode is ~4× realtime — 1,270 tokens for 90 s of audio, against ~210 for ASR — so an
80-minute recording is roughly twenty minutes of A100. No untagged trailing text on either
window; the largest inter-token gap was 3.9 s.

**Speaker numbering across chunks.** The card says numbers are ordinals by first appearance and
names incremental decoding as the way to keep them consistent. The spike confirmed the
mechanism works (a `prefix_text` continuation carried on without re-transcribing) but the
second half-window happened to contain one speaker, so relabelling was not directly observed.
The adapter assumes the card is right and namespaces labels per window.

**The name again.** Granite is a different lineage from CrisperWhisper and garbled `Courchesne`
the same way — `korshane` in timestamp mode; `Korschane`, `Korshezeni`, `Korsheane` in one ASR
sentence. Independence of lineage does not buy independence of error on a name neither model
has seen; §7b's dissenters would all agree it is a garble, which is the point, but none would
supply the spelling.

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
  [x] annotate.py              # stage 1.5 CLI: proper-noun candidates by dissenter conjunction
  [x] render.py                # stage 2 CLI
  pipeline/
    [x] capabilities.py        # Capabilities, ModelSpec, plan_for()
    [x] registry.py            # REGISTRY: model_id -> ModelSpec, exact match only
    adapters/
      [x] base.py              # Adapter ABC, Word, AdapterResult, ErrorRange
      [x] crisperwhisper2.py
      [x] whisper.py           # generic openai/whisper-* only
      [x] mock.py              # six capability-declaring fakes, no weights
      [x] granite41plus.py     # two passes per 200 s window, aligned (Phase 3)
      [x] ark.py  parakeet_tdt.py  granite_turboctc.py   # the §7b dissenters, text only
    [x] chunking.py            # fixed-window segmentation + retry by subdivision
    [x] flags.py               # filled_pause / vocalization / partial_word / silence tagging
    [x] fluent.py              # the §7b fluent view and three-dissenter conjunction
    [x] diarize.py             # pyannote wrapper
    [x] provenance.py          # hashes, revisions, package + git versions
    [x] schema.py              # raw JSON build/read/write + validate
    [x] orchestrator.py        # dispatch from the declaration; assembles the record
    [x] preview.py             # stage-1 text preview, superseded by render.py
    [ ] align.py               # forced-alignment gap-filler
  render/
    [x] __init__.py            # RWord, Sentence, Turn, Profile, PROFILES, stream selection
    [x] speakers.py            # max-overlap assignment, island smoothing, name map
    [x] segment.py             # sentences, turns, within-turn anchors
    [x] stamp.py               # the version stamp every render carries
    formats/
      [x] txt.py  plain.py  html.py
      [ ] tex.py               # optional (D11); not built
    templates/
      [x] base.css  coding.css  lecture.css
tests/
  [x] check_contract.py        # 41 dependency-free checks: stage 1 dispatch + the fixture
  [x] check_render.py          # 26 dependency-free checks: stage 2 against the fixture
  [x] check_annotate.py        # 8: the fluent view, the conjunction, annotate.py end to end
  [x] fixtures/golden.json     # committed; renderer tests need no models
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

| Signal | Proseminar, guest + host (14,709 w, 182 wpm) | Lecture (11,588 w, 147 wpm) |
|---|---|---|
| terminal punctuation | every 4.7 s | every 7.0 s |
| pause > 0.2 s | every 2.8 s | every 2.8 s |
| **pause > 0.3 s** | **every 5.1 s** | **every 4.5 s** |
| pause > 0.5 s | every 14.5 s | every 12.8 s |
| speaker change | every 43.6 s | every 60.7 s |

**Punctuation is not sparse.** It alone yields a sentence every 4.7–7.0 s, so it is the primary
signal and the pause rule supplements it — the reverse of what was assumed here previously.

~~**Provisional default: `--pause-threshold 0.3`.**~~ Superseded below.

#### Re-derived on raw timestamps (2026-09-01)

Bug 8 is fixed and the fixture carries raw CrisperWhisper timestamps, so the number above
was re-measured. Two findings.

**The 0.000 s gaps were not bug 8.** 72% of inter-word gaps in the fixture are exactly zero
with no pause adjustment anywhere in the path. That is the Viterbi aligner's nature — it
partitions the timeline, so a word's end is usually the next word's start — not an artefact.
The earlier attribution to `_adjust_pauses` was at most half right.

**0.5 s, not 0.3.** On the 10-minute fixture (182 wpm):

| threshold | pause boundaries | one every |
|---|---|---|
| 0.3 s | 256 | 2.3 s — fragments |
| **0.5 s** | **126** | **4.8 s** |
| 0.75 s | 58 | 10.3 s |
| terminal punctuation | 114 | 5.3 s |

At 0.3 s the pause rule fires twice as often as punctuation and splits mid-phrase; at 0.5 s
it matches punctuation's cadence and supplements it. **`--pause-threshold 0.5` is the
default.** Still a flag; re-rendering is instant, and `tests/check_render.py` asserts the
ratio stays within 2× so a future recording that breaks the assumption is noticed.

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

**Built (2026-09-01).** `--speaker-map`, a file of `SPEAKER_00: Name` lines, applied after
assignment and smoothing and before turn grouping; the raw JSON is never touched, and the stamp
records which file was used and which labels it left unmapped. Three properties worth knowing:

- **It is per-recording by nature.** Labels are numbered by first appearance within one
  diarizer run, so `SPEAKER_00` in one file is unrelated to `SPEAKER_00` in the next. The
  convention is a sidecar, `<stem>_speakers.yaml` beside `<stem>_raw.json`, which the renderer
  picks up automatically; `--no-speaker-map` renders the anonymous labels on purpose.
- **Two labels may map to one name**, and turn grouping then merges them. That is the fix for
  the diarizer splitting one person into two labels (three labels on the two-person fixture).
- Sidecars naming real people are **not committed** alongside a public fixture.

### Layout detail — resolved (D23)

The coding margin is a **right-hand column**. Decided without polling the coders so Phase 2
can proceed; it is a five-minute CSS change if they ask for double-spaced lines instead.

---

## 7. Local development without SLURM

Most of this pipeline needs no GPU — and no model — at all.

- **Stage 2 is entirely model-free.** Turn grouping, anchors, speaker assignment and smoothing,
  line numbering, print CSS, both profiles, every format backend: a pure function over JSON.
  **Done 2026-09-01: `tests/fixtures/golden.json`** — 10 min of `251211_0009.wav`, dual-stream,
  diarized, 1,969 + 1,864 tokens, 59 turns over 3 labels, 61 disfluency flags. See its README
  for provenance. Renderer development and tests now run in milliseconds on any machine.
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

> **There is no interview recording in `data/` yet.** `251211_0009.wav` was labelled one here
> and in `docs/environments.md` until 2026-09-01; it is a proseminar guest lecture with host
> interaction. The coding profile — the project's primary use case — is therefore being
> developed against lecture-with-dialogue audio. Get a real participant interview in before
> Phase 2 is called done: spontaneous two-party dialogue is where turn boundaries, overlap and
> short filler tokens near speaker changes are hardest, and none of that is represented yet.

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

#### Dissenters spike (2026-09-01, A100, `audio-transcribe-tf5`, job 48746)

Do the three still run *correctly*, and does the conjunction reproduce the earlier
measurement? Each runner follows its model card (D19); the invocations this project had
already got wrong once are the point of checking.

| dissenter | invocation that works | words, 45:00 | RTFx | unmatched vs. primary |
|---|---|---|---|---|
| `Audio8/ARK-ASR-0.6B` | card script: `AutoModelForCausalLM` + `trust_remote_code`, fp16, **cast only `inputs["audios"]`**, `audio_max_length=30·16000`, `do_sample=False`, `bad_words_ids`; **30 s clips** | 157 | 17 | 3.8% |
| `nvidia/parakeet-tdt-0.6b-v3` | card: `AutoModelForTDT`, unchunked — but see below | 141 → **159 in 30 s windows** | 113 | 13.3% → lower |
| `ibm-granite/granite-speech-5.0-470m-turboctc` | card: `AutoModelForCTC`, **unchunked** (chunked: 68 words — bug 14) | 166 | 677 | 11.4% |

Reference for that window: CrisperWhisper 166 verbatim / 154 intended; Granite 4.1 ASR 158.

**Parakeet's transformers port needs 30 s windows, no overlap.** Unchunked on 90 s it returned
141 words on the lecture window (143 in 45 s sub-windows, **159 in 30 s** — matching the
reference) but a correct 264 on the proseminar window, so the drop is input-dependent, not
a fixed cap. The 210 recorded earlier for the same window was `chunk_length_s=30` with the
pipeline's stride duplicating words at every boundary. Neither the card's NeMo long-form
(local attention, 24 min) nor its streaming script applies to the port. Rule: ≤30 s windows,
like ARK; a transducer needs no overlap.

**ARK's `牵牵牵` collapse is in `hpc/logs/ark_48646.log`** for anyone who doubts the §3
callout: every float tensor cast to fp16, no `audio_max_length`, `max_new_tokens=600`.

**The conjunction, on both windows:**

| window | primary fluent tokens | flagged | of which real garbles |
|---|---|---|---|
| lecture 45:00 (has proper nouns) | 158 | **5 (3.2%)** | **4** — `Korshesney` ×2, `Korshane`, `Brooklyn.`; residue: `Okay.` |
| proseminar 11:30 (no proper nouns) | 264 | **8 (3.0%)** | **0** — `and`, `a`, `had`, `them's` ×2, `You're`, `Like`, `be` |

Every garble of the probe name was flagged, with no glossary — *in the spike*; the corrected
three-dissenter run below flags two of the name's four occurrences, because Parakeet agrees
with CrisperWhisper on `Korshane`. And the residue has a shape: **all eight false flags on the second window sit next to a disfluency**
— `a` beside `d- d-`, `had` between `[UM]` and `[UH]`, `Like` after `[laughter]`, `You're` in a
`You're you are` repair, `them's` a contraction the fluent view did not expand. The fluent
view drops the disfluency but the dissenters also drop or reshape the word *beside* it. Two
cheap refinements for `annotate.py`, measured below: **mask tokens adjacent to a dropped marker
or partial**, and expand `'s` contractions on pronouns (`them's` → `them is`) as `gonna` already
is.

**Both refinements measured, then corrected.** A first re-analysis of the spike outputs put
residue at 0.95% with recall 5/5 on the lecture window. Half of that was wrong: the spike
script wrote both windows' 30 s Parakeet output to one file, so the lecture window was scored
against the *proseminar's* Parakeet text — which disagreed with everything and quietly reduced
the lecture window to a two-dissenter conjunction. Re-run through the real pipeline (job 48749,
ARK fixed, Parakeet correct):

| window | fluent tokens | flagged | real | residue |
|---|---|---|---|---|
| lecture 45:00, three dissenters, pipeline | 165 | **4 (2.4%)** | **3** — `Korshesney` ×2, `Brooklyn.` | 1 — `Okay.` |
| proseminar 11:30, three dissenters, spike (its Parakeet file was the correct one) | 271 | **2 (0.7%)** with both refinements | 0 | 2 — `and`, `be` |

Residue over both: **3 of 436 fluent tokens, 0.7%** (through `pipeline/fluent.py` as shipped; an
earlier ad-hoc re-analysis said 4 of 429).

**What the mask buys, on the same records:**

| | flagged / real / residue |
|---|---|
| lecture 45:00, mask on **or** off | 4 / 3 / `Okay.` — no candidate sits beside a marker, so the mask is inert |
| proseminar 11:30, mask **on** | 2 / 0 / `and`, `be` |
| proseminar 11:30, mask **off** | 4 / 0 / `and`, `a`, `Like`, `be` |

Without the mask, residue over both windows is 5 of 436 (**1.1%** against 0.7%) and nothing
real is gained. Scaled to speech at ~180 wpm, the mask removes roughly seven false `[NAME?]`
marks per ten minutes — each a function word beside a filler, which a coder dismisses at a
glance — and on this data hides nothing. Its only possible cost is the lowercase-garble-beside-
marker case, which has not yet occurred. Default on; `--no-mask-adjacent` to compare. Recall on the probe name: **two of its
four occurrences, plus `Brooklyn.`** The two `Korshane` occurrences are unflagged because
**Parakeet independently produced `Korshane`** — it missed only 4 of 165 tokens on this window
— and the conjunction withholds a flag when any dissenter agrees. That is the rule working as
D18 states it, and the honest cost of requiring unanimity: a spelling that is phonetically
natural can be reached by two lineages at once. A coder still sees the name flagged twice on
the page.

The adjacency mask's exemption for capitalised mid-sentence tokens stays as the rule (it is
what keeps `Eric Korshane [UH]`-shaped tokens flaggable), but on this window its measured
benefit is nil: no candidate sat beside a marker once Parakeet was scored correctly. Both
refinements and the exemption are flags in `annotate.py`, on by default, re-measurable in a
second.

Cost is far below the earlier ~40% estimate: ARK ~17–22×, TDT ~100×, CTC ~675× realtime
against CrisperWhisper's ~33×, so all three add roughly **+8%** to a run (ARK dominates).

The script that produced the earlier 0.8%/2.0% figures was not in `hpc/jobs/` — an ad-hoc
login-node run — so these rates stand on their own rather than as a replication; the fluent
view here is written from this section's rules.

#### Built (2026-09-01)

Three adapters (`ark.py`, `parakeet_tdt.py`, `granite_turboctc.py`) registered like any other
model, `pipeline/fluent.py` for the view and the conjunction, and `src/annotate.py` as stage 1.5:

```bash
# three ordinary stage-1 runs, in audio-transcribe-tf5 (they run in ~+8% of the primary's time)
for M in Audio8/ARK-ASR-0.6B nvidia/parakeet-tdt-0.6b-v3 ibm-granite/granite-speech-5.0-470m-turboctc; do
    python src/transcribe_audio.py -i x.wav -o transcripts/ --model $M --no_diarize --job_id ${M##*/}
done
# then, anywhere, instantly
python src/annotate.py transcripts/x_raw.json --dissenters transcripts/x_job*_raw.json
python src/render.py transcripts/x_annotated.json --profile coding       # candidates print as  word [NAME?]
```

Verified end to end on the A100 (job 48749): three dissenter records through
`transcribe_audio.py` (157 / 159 / 166 tokens, each carrying the not-built-aligner warning),
`annotate.py` against the CrisperWhisper primary, `render.py --profile coding` — and
`Korshesney [NAME?]` twice and `Brooklyn. [NAME?]` on the page. The first run through the
adapter had ARK emit zero words and exit 0 (the card wants a file path, the adapter passed an
array); a dissenter adapter now fails the run on an empty transcript, and `annotate.py`
refuses a dissenter with under a quarter of the primary's tokens, since it would vote against
everything and the conjunction would quietly become a vote of the rest.

The dissenters are separate runs because they cannot share a process with the primary:
CrisperWhisper lives in `cw2diar`, these need `transformers` 5.16. That also happens to be what
D1 wants — change the masking rule and `annotate.py` re-runs in a second with no GPU. The
`annotation` block records every candidate, masked or not, with each dissenter's model id and
revision; the flag is `proper_noun_candidate`, on the primary and the secondary stream alike.

**The mask's blast radius, measured and closed.** A candidate beside a dropped marker or
partial is masked — that removes the residue — but hesitation precedes retrieval of a hard
name (`Eric Korshane [UH]` is the canonical shape), so the mask would hide exactly the tokens the
detector exists for. Exemption: a **capitalised, non-sentence-initial** token is never masked.
On the proseminar window it masks `a`, `had`, `Like` and keeps nothing it should not; on the
lecture window, scored correctly, no candidate sits beside a marker, so the exemption costs and
saves nothing there. What the mask can still hide is a *lowercase* garble beside a marker
(`psychatology`-class technical terms), which is the residual risk. And what no mask affects:
a garble two lineages reach independently (`Korshane`) is not flagged at all — the price of
unanimity.

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

**Done (2026-09-01).** `src/render.py` and `src/render/`, zero dependencies, verified against
the fixture by 21 checks that run in milliseconds anywhere:

```bash
python src/render.py transcripts/x_raw.json --profile coding            # txt + html
python src/render.py transcripts/x_raw.json --profile lecture --format html,plain \
    --speaker-map x_speakers.yaml --anchor-interval 45
```

What it does, in order: pick the stream the profile wants (verbatim for coding, intended for
lecture, with a stamped warning if the record lacks it) → max-overlap speaker assignment with
nearest-turn fallback for words in diarizer gaps → island smoothing inside pause-bounded runs
(≤2 words flanked by an agreeing speaker, edges left alone) → name map → sentences
(punctuation ∨ pause > 0.5 s ∨ speaker change) → turns (consecutive same-speaker sentences:
the diarizer's 59 segments become 7) → within-turn anchors every 30 s / 60 s snapped to a
sentence start → txt / plain / html, each carrying the version stamp.

Two things measured on the way. The pause threshold moved to 0.5 s (§6). And timestamps are
shown in the **original recording's** time: the excerpt offset stage 1 recorded is applied,
so a coder checking the audio at `00:05:30` finds it there rather than 300 s off.

One wrapping bug caught by the checks and worth remembering: `textwrap` splits on hyphens by
default, so `self-report` at a line end became `self-` — indistinguishable from a
`partial_word` in a verbatim transcript. Both text formats now wrap between tokens only.

Not built: `tex` (optional, D11), `docx` (deferred, D16), and `annotate.py` (stage 1.5),
which is where `repetition`/`repair` tagging goes when a detector is chosen.

### Phase 3 — Second adapter

Granite 4.1-plus. Exercises `end_only` timestamps, silence tokens and native speaker labels,
proving the contract holds rather than having been written around one model.

**Done (2026-09-01).** `src/pipeline/adapters/granite41plus.py`, registered; verified through
the real pipeline on the A100 (`audio-transcribe-tf5`, `251211_0009.wav`):

| | A — 11:30–13:00, one window | B — 11:00–16:00, two windows |
|---|---|---|
| tokens (spoken + silence) | 268 + 45 | 936 + 148 |
| every `start` = previous token's `end` | yes | yes |
| ends monotonic after unwrap | yes | yes |
| turns / labels | 3 — `Speaker 1`, `Speaker 2` | 6 — `w1:Speaker 1/2`, `w2:Speaker 1/2` |
| `diarization` block | the model's own SAA pass, `source: asr` | same |
| words carrying a speaker | 0 — assigned in stage 2 (D3) | 0 |
| realtime factor, both passes | 2.2 | 2.2 |
| exit / `validate()` | 0 / clean | 0 / clean |

The plan line the orchestrator logged is the whole point of the phase:
`chunking=True derive_starts=True forced_alignment=False diarizer=False timing=derived` —
three paths CrisperWhisper never takes, dispatched from the registry declaration alone.

What the spike forced (§3): the timestamp and speaker modes do not combine, so the adapter
runs **two passes per 200 s window** and aligns them by token sequence; the SAA turns land in
`speaker_turns`, where stage 2 treats them like pyannote's. Speaker numbers restart per
window, so multi-window labels are namespaced and the record says so in `warnings`; the
render-time speaker map merges them. At 2.2× realtime an 80-minute file is ~36 minutes of
A100 — fine for a contract-proving adapter; CrisperWhisper remains the production model (D4).

**Follow-ups, not blockers:** the stage-2 renderer must drop `silence` tokens before
segmentation (they would bridge every pause the pause rule looks for) — it lands once the
renderer PR is merged. And a sliding-window variant of the card's incremental decoding, with
`prefix_text` carrying the previous window's tail, might keep speaker numbers consistent
across windows; unverified, a spike item.

### Stage 1.5 — `annotate.py` and the dissenters

**Done (2026-09-01).** The §7b design built end to end: three dissenter adapters, the fluent
view with its two measured refinements and the name-like exemption, `annotate.py`, and the
`[NAME?]` mark in both text and HTML. See §7b "Built" for usage and the verified run.

What `annotate.py` does *not* do yet: `repetition` and `repair` tagging (§5), which is the
`TextCleaner` BERT model repurposed from deleter to tagger. Same file, later.

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
| **Granite timestamp + speaker mode combination** | **Resolved 2026-09-01: they do not combine.** A fused prompt yields timestamp output with no attribution (§3). The adapter runs two passes per window and aligns them. |
| **Coding margin layout** | **Resolved 2026-09-01: right-hand column (D23).** Reopens the moment a coder asks for something else. |
| **Verbatim punctuation sparsity** | **Resolved 2026-08-28.** Not sparse — a sentence every 4.7–7.0 s across two ~80 min recordings. Punctuation is the primary signal; `pause > 0.3 s` is the provisional default, to be re-derived after bug 8. See §6. |
| **Diarization speaker count** | **Partly explained 2026-08-31.** Over 10 min excerpts community-1 gives 2 speakers (lecture) and 4 (proseminar) — plausible. The alarming 10-and-4 counts came from full ~80 min files, so speakers accumulate over duration rather than the model failing outright. Still unverified against the audio. |
| **Diarization model choice** | **Settled 2026-08-31: stay on community-1.** DiariZen v1/v2 benchmarked against it on 10 min excerpts: same speaker counts, near-identical speech totals and runtime (~12 s), 20% fewer/longer segments on the lecture. Not enough to justify its install — a separate env pinned to torch 2.1.1, a vendored pyannote fork, and an `LD_LIBRARY_PATH` workaround for a Rocky 8 `libstdc++` mismatch. Both DiariZen checkpoints are CC-BY-NC; community-1 is CC-BY-4.0. No DER computed: no reference labels. |
| **Fine-tuning for disfluencies** | **Dropped 2026-08-31.** Premised on no accurate model emitting disfluencies, which was an artefact of calling CrisperWhisper through `transformers.pipeline`. `mode="verbatim"` supplies them natively. The teacher-forced groundwork (Qwen3-ASR sits at ~1% filler probability against CW2's ~0.1%) is recorded in case the premise returns. |
| **Vocalization coverage** | Only `[laughter]`, once, in 12 windows of lecture + proseminar audio. Whether CW2 tags coughs/breaths is untested — this material may simply contain none. |
| **Diarization at turn boundaries** | Short filler tokens near a speaker change are where max-overlap assignment is noisiest; hence smoothing within pause-bounded runs. |
| **IRB / data governance** | Interview recordings are human-subjects data and stage 1 moves them to shared cluster storage. Assumed covered by the existing protocol; flagged because this pipeline automates the transfer. |
| **RTX 3070 VM provisioning** | Separate infrastructure task, in progress in another context. Not a blocker (D14). |
| **Proper nouns: how to resolve a flagged token** | **Detection is built (§7b, 2026-09-01); resolution is what remains open.** On the 45:00 window, `hotwords` produced the only exactly-correct spelling of the probe name and corrupted a neighbouring word doing it; `--mode intended` corrupted nothing and still got the name wrong (`Courchesney` x4). §7b's resolution step therefore has no good default. Worth measuring over the 12-window sweep, comparing **exact token forms**, not substrings: how often does each route land the correct spelling, and how often does hotword biasing damage a non-target token? D21's cost is established; the benefit's reliability is not. **Second example, from the first print proof:** `Dr. Geisler` at 14:03 came out `Dreisler` in **both** streams — the intended stream did not help at all this time — and the listener's own judgement was that the name is fast and soft enough to mishear without knowing it. That is the case a glossary exists for, and only a glossary fixes it. A render-time corrections file (`Dreisler: Geisler`, applied like the speaker map and recorded in the stamp) would keep the raw record lossless while fixing the printout; not built. |
| **Forced alignment (`align.py`)** | Dispatched to but not built. No registered real model declares `word_timestamps="none"`, so nothing needs it yet; the run records the gap in `warnings` rather than emitting untimed words that look timed. Build it when a model needs it, using `CrisperWhisperModel.forced_align()`. |
| **Speaker identification from an enrolled embedding library** | **Far future — logged 2026-09-01, not scheduled.** Diarization *is* embedding + clustering, and `community-1` computes a per-label embedding internally (its config names an embedding model in the clustering step) and discards it. The feature: a stage-1.5 pass that takes each label's centroid, cosine-matches it against enrolled voices, and **emits a proposed `_speakers.yaml` with confidence scores** — never rewriting the raw labels (D3). It would also merge an over-split speaker automatically. **The constraint is not technical:** a library of participant voice embeddings is biometric data, a different consent and IRB category from a transcript; scope any library to non-participants (hosts, guest faculty) or make it per-study with consent first. Checked, not assumed: pyannote 4.0.3's diarization pipeline has **no `return_embeddings`** flag (only its speech-separation pipeline does; 3.x had it on diarization), so extracting the vectors is the spike's first question. Spike, per the house rule: centroids from two recordings of the same host; does cosine similarity clear the noise of different rooms and microphones? If it does, the cheap groundwork is storing per-label centroids in the raw JSON so a library can be built retroactively — stripped from any public fixture, the way the cluster user id was. |
| **CrisperWhisper `confidence`** | Declared `False`. `TranscriptionResult` has no per-word confidence field, so `conf` is always null. Worth revisiting if a later package version exposes one — coders benefit from knowing which tokens the model was unsure of. |

---

## 11. Next action

**Put a real transcript in front of the coders.** Phases 1 and 2 are built; the question now
is whether the coding profile is usable on paper, and only the students who mark these up can
answer it. Render `tests/fixtures/golden.json` (or any stage-1 record) with
`--profile coding`, print the HTML, and hand it over. Two things to watch for:

- **D23's right-hand margin** — enough room, or do they want double-spaced lines? Five-minute
  CSS change either way.
- **Sentence-per-line numbering** — a pause-split fragment is its own numbered line by design
  (it shows exactly where the speaker hesitated), but it is also more lines to cite. See
  whether that helps or annoys.

**Get a participant interview recording onto the cluster** (§7). The coding profile has been
developed against a guest lecture with host interaction. Spontaneous two-party dialogue —
overlap, short turns, fillers at speaker changes — is where assignment and smoothing are
hardest, and none of it is represented in the fixture.

Then Phase 3: the Granite 4.1-plus adapter, which exercises `end_only` timestamps, silence
tokens and native speaker labels — the paths that prove the capability contract was not
written around one model. The registry already holds its declared shape.

---

## 12. Superseded work

`docs/granite_word_timestamps_plan.md` proposed extracting word timestamps from Granite by
finding monotonic attention heads and training an auxiliary transformer on LibriSpeech
alignments — an 11–21 hour estimate that was optimistic, required GPU training and ground-truth
alignments, and predicted **start times only**.

It is unnecessary. Granite Speech 4.1-2b-plus (released 2026-04-28, *after* that plan was
written) provides word-level timestamps natively, with explicit silence tokens. Keep the file
for provenance with a superseded header rather than deleting it.
