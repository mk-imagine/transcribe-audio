# transcribe_audio

A transcription pipeline for two research uses, from **one transcription run per recording**:

1. **Qualitative coding of research interviews.** Transcripts are printed and hand-coded by
   students, so they must be *verbatim* — every `[UH]`, every `the the`, every `de-` where a
   word was abandoned — with speaker attribution, timestamps, and a wide margin to write in.
2. **Lecture transcription.** Clean prose, with timestamps so the audio can be checked when a
   line looks wrong.

**Disfluencies are data, not noise.** A hesitation before a disclosure about identity is
analytically meaningful. That one requirement constrains the model choice more than anything
else, and it is why stage 1 never cleans anything.

Everything runs locally: models come from HuggingFace, audio never leaves controlled machines.
Production compute is the SFSU HPC cluster (SLURM, A100); everything after stage 1 runs on a
laptop with no packages installed at all.

---

## What's built

| stage | command | in → out | needs |
|---|---|---|---|
| **1** | `src/transcribe_audio.py` | audio → lossless raw JSON (schema v1) | GPU, a mamba env |
| **1.5** | `src/annotate.py` | primary + dissenter JSONs → JSON with `proper_noun_candidate` flags | nothing |
| **2** | `src/render.py` | JSON → `txt` / `html` / `plain`, two profiles | nothing |

Stage 1 is **verbatim and lossless**: no cleaning, no filtering, no speaker assignment. Anything
recomputable from the JSON is recomputed in stage 2, so changing a pause threshold or fixing a
misattributed speaker never means re-queuing a GPU job.

Models are chosen by a **capability contract**, not by name. Each adapter declares what its
model provides — word timestamps, a verbatim mode, its own speaker labels, silence tokens, how
it handles long audio — and the orchestrator fills the gaps from the declaration. Nothing in the
code inspects a model id to decide what a model can do; an unknown id is refused, not guessed at.

### Models

| model | role | env on the cluster | notes |
|---|---|---|---|
| `nyralabs/CrisperWhisper2.0_large` | **primary, default** | `cw2diar` (with diarization) or `cw2native` | verbatim *and* intended modes, per-word start+end, native long-form. Weights are non-commercial research licence. |
| `pyannote/speaker-diarization-community-1` | diarizer | `cw2diar`, `audio-transcribe` | turns stored raw; assigned at render time |
| `ibm-granite/granite-speech-4.1-2b-plus` | second primary (contract-proving) | `audio-transcribe-tf5` | end-only timestamps, silence tokens, own speaker labels; two passes per 200 s window; ~2.2× realtime |
| `Audio8/ARK-ASR-0.6B`, `nvidia/parakeet-tdt-0.6b-v3`, `ibm-granite/granite-speech-5.0-470m-turboctc` | the three **dissenters** | `audio-transcribe-tf5` | text only; together ~8% of a CrisperWhisper run |
| `openai/whisper-*` | generic fallback | `audio-transcribe` | cannot do verbatim; refused if you ask for it |
| `mock/*` | six fakes for the checks | none | no weights, no ML packages |

`python src/transcribe_audio.py --help` lists every registered id.

---

## Quick start

The path a coding transcript takes, start to finish:

```bash
# 1. on the cluster, in cw2diar: verbatim + intended streams, diarized, one run
python src/transcribe_audio.py -i data/interview.wav -o transcripts/ --dual_stream

# 2. anywhere, instantly: the coding transcript, printable
python src/render.py transcripts/interview_raw.json --profile coding
#    -> transcripts/interview_coding.txt  and  interview_coding.html  (Cmd+P to print)
```

Add the three dissenter runs and `annotate.py` between those two steps to get `[NAME?]` marks
on likely proper-noun garbles — see *Stage 1.5* below.

---

## Stage 1 — transcribe

```bash
python src/transcribe_audio.py -i <audio> -o <dir> [options]
```

| option | meaning |
|---|---|
| `--model ID` | a registered checkpoint. Default `nyralabs/CrisperWhisper2.0_large`. Unknown ids are refused with the list of known ones. |
| `--mode verbatim\|intended` | CrisperWhisper's two decoders. Default from the registry: verbatim. Refused for a model that has no verbatim mode. |
| `--dual_stream` | both renderings from **one batched pass** (~11% more than one stream, not 100%). Each carries word timestamps. The lecture profile renders the intended stream, so this is what you want for a recording that serves both uses. |
| `--start_time HH:MM:SS --end_time HH:MM:SS` | transcribe an excerpt. The record names the *original* file and its hash, and all timestamps are later shown in the original's time. |
| `--no_diarize` | skip pyannote. Required in `cw2native`, which has no pyannote. |
| `--hotwords "A,B"` or `--hotwords file.txt` | bias recognition toward names. **Read the caveat below.** |
| `--backend auto\|ct2\|transformers` | CrisperWhisper backend; ct2 is 4–5× faster and is what runs in production. |
| `--compute_type` | float16 on CUDA, float32 on CPU by default; ct2 refuses float16 on CPU. |
| `--device cuda\|cpu` | CUDA if available. MPS is not a target. |
| `--adapter NAME` | force an adapter for an unregistered checkpoint. Recorded in the output as an override. |
| `--job_id X` | names the output `<stem>_jobX_raw.json`; the SLURM wrapper passes the job id. |

**Outputs:** `<stem>_raw.json` (the record — see *The record* below) and `<stem>_preview.txt`, a
readable preview that is *not* the stage-2 render. With `--dual_stream`, a second preview for
the other stream.

**Exit codes:** `1` — audio ranges were lost and the transcript has holes (the ranges are in the
record's `errors`); `2` — the model id or the requested mode was refused.

### On the cluster

```bash
./src/run_transcription.sh -i data/x.wav -o ./transcripts [-c gpu|cpu] [-E <env>] [-M <model>] [-s HH:MM:SS -e HH:MM:SS]
```

Works from any directory. Logs go to `hpc/logs/`. Default env is `cw2native`; pass `-E cw2diar`
for diarization. See `docs/environments.md` for the envs, the partitions, and how to reach the
cluster; `envs/cw2diar.sh` is the recipe for the combined environment.

### The hotwords caveat (D21)

CrisperWhisper accepts `--hotwords` on every checkpoint but was trained for it only on the Pro
ones. On the standard default it *works* — it spelled `Courchesne` correctly where the model
alone wrote `Korshane` — **and it damaged a neighbouring word** (`commissure` → `commasure`) in
the same window. The run warns, and the warning is written into the record. Use it when a name
matters; run the dissenters too, which caught the damaged word.

---

## Stage 1.5 — dissenters and `annotate.py`

Proper-noun garbles are found by **disagreement**: three small models from independent
lineages transcribe the same audio, and a token *all three* disagree with — compared on a
fluent view of the verbatim primary, so disfluencies cannot be flagged — becomes a
`proper_noun_candidate`. No glossary needed. Measured: about 1% of tokens flagged, three of
five real garbles caught on a name-dense window.

```bash
# three ordinary stage-1 runs, in audio-transcribe-tf5
for M in Audio8/ARK-ASR-0.6B nvidia/parakeet-tdt-0.6b-v3 ibm-granite/granite-speech-5.0-470m-turboctc; do
    python src/transcribe_audio.py -i data/x.wav -o transcripts/ --model $M --no_diarize --job_id ${M##*/}
done

# then, anywhere, in a second
python src/annotate.py transcripts/x_raw.json --dissenters transcripts/x_job*_raw.json
#    -> transcripts/x_annotated.json  (render this instead of x_raw.json)
```

| option | meaning |
|---|---|
| `--min-dissenters N` | how many must disagree. Default: all three; fewer is refused unless you ask. |
| `--no-mask-adjacent` | also flag a candidate beside a dropped marker or partial. The mask removes function-word residue there; a capitalised mid-sentence token is never masked. |
| `--allow TERM …` | never flag these. The primary's own `--hotwords` are allowlisted automatically; `--no-hotword-allowlist` turns that off. |

Every candidate — flagged, masked, or allowlisted — is recorded in the output's `annotation`
block with each dissenter's model id and revision. Changing a rule is a re-run in a second, no
GPU. The dissenters run separately because they need a different environment from the primary.

### Repetition and repair tags (opt-in)

```bash
python src/annotate.py transcripts/x_raw.json --disfluencies                       # alone
python src/annotate.py transcripts/x_raw.json --disfluencies --dissenters ...      # or together
```

Adds `repetition` and `repair` flags to the words, by rule, with no model: a span of up to three
words immediately restated (`the the`, `not not mirror`, `You're you are`) is a repetition; a
partial word the next word completes (`de- developmental`) or a short span restarted with one
word changed (`I went I drove`) is a repair. The abandoned material is tagged and the restart
stands, so one flag is one event. The text is never changed. These are for consistency across
coders and for counting hesitation rates, not for reading — the verbatim text already shows a
repetition — so they are off unless asked for, and the renderer marks only a repair, faintly, in
HTML. Counts and events land in the record's `annotation.disfluencies`.

Restatements and completed partials are tight; **substitution is the loose shape** — on the
fixture about 11 of its 15 events are real, the rest lists without commas and `to VERB to
PREP`. Drop it with `--repair-shapes restatement,completed_partial` for a conservative tagging.

---

## Stage 2 — render

```bash
python src/render.py <record.json> [--profile coding|lecture] [--format html,plain] [options]
```

Pure Python over the JSON: runs on the laptop with nothing installed, in milliseconds.

| profile | stream | line numbers | margin | anchors |
|---|---|---|---|---|
| `coding` (default) | verbatim | yes, one per sentence | 2.75 in right-hand column for handwritten codes | every 30 s |
| `lecture` | intended | no | normal | every 60 s |

| option | meaning |
|---|---|
| `--format html,plain` | extras beyond `txt`, which is always written. `html` (default) is the print format: open it and Cmd+P. `plain` is words only, for grep and diff. |
| `--speaker-map FILE` | `SPEAKER_00: Dr. Gard`, one per line. A `<stem>_speakers.yaml` beside the input is used automatically; `--no-speaker-map` renders the anonymous labels. Two labels may map to one name, which merges a speaker the diarizer split. Labels are per-recording. |
| `--pause-threshold S` | silence that ends a sentence. Default 0.5 s, re-derived on raw timestamps. |
| `--anchor-interval S` | within-turn timestamps, so a long turn stays citable. |
| `--no-smoothing`, `--max-island N` | speaker smoothing: a run of ≤ N words inside a pause-bounded stretch whose neighbours agree on another speaker is reassigned. |
| `--no-offset` | show excerpt-relative times instead of original-recording times. |

Every render carries a **stamp** — model, revision, render parameters, date, a content hash —
so a printout and a later render visibly differ. Line numbers hold only for one render; **the
timestamp is the durable reference.** In the coding profile a `proper_noun_candidate` prints as
`word [NAME?]`; markers print as they came from the model.

---

## The record (schema v1)

`<stem>_raw.json`, the lossless output of stage 1. The parts you will read:

```jsonc
{
  "source":       { "audio_path": "...", "audio_sha256": "...", "duration_s": 600.0,
                    "excerpt": { "start_time": "00:05:00", "offset_s": 300.0, ... } },
  "run":          { "created_utc": "...", "device": "cuda", "slurm_job_id": "...",
                    "pipeline_version": { "commit": "...", "dirty": false }, "packages": { ... } },
  "asr":          { "model_id": "...", "revision": "<hf sha>", "revision_source": "local cache",
                    "capabilities": { ... }, "params": { "mode": "verbatim", ... },
                    "performance": { "processing_time_s": 39.3, "realtime_factor": 15.3 } },
  "diarization":  { "model_id": "...", "revision": "..." },
  "words":        [ { "i": 0, "text": "[UM]", "start": 0.79, "end": 1.02,
                      "timing_source": "native", "speaker": null, "flags": ["filled_pause"] }, ... ],
  "speaker_turns": [ { "start": 0.0, "end": 41.2, "speaker": "SPEAKER_00" } ],   // raw
  "secondary_stream": { "mode": "intended", "text": "...", "words": [ ... ] },    // with --dual_stream
  "errors":       [ ],       // lost ranges, kept in the timeline
  "warnings":     [ ]        // every caveat the run raised, e.g. the hotwords warning
}
```

Flags a word can carry: `filled_pause`, `vocalization`, `partial_word`, `silence` (Granite's
pause tokens), and after stage 1.5, `proper_noun_candidate`, `repetition`, `repair`. Speakers are absent from words
unless the model itself attributed them; diarizer turns are assigned at render time.
`timing_source` says whether a timestamp is `native`, `derived` from the previous token's end,
or `aligned`.

---

## Naming recordings

The recorder names files `YYMMDD_XXXX.wav`, which says nothing about what was recorded. The
semester calendar does:

```bash
python scripts/rename_recordings.py data/            # dry run: shows old -> new and what is skipped
python scripts/rename_recordings.py data/ --apply    # rename
```

`SEMESTER_START` and `COURSES` at the top of the script are the whole configuration — the
first day of term and each course's weekdays (with an optional time window for a day that has
two courses; this semester has none). Names come out as `PSY498-week3.wav`; a course that
meets twice a week gets the weekday appended, `PSY777-week3-Mon.wav`, since one name per week
cannot hold two lectures. Two recordings on one day get `-1`, `-2`. Nothing is ever
overwritten, and a file on a day with no class is skipped and listed. `--start` overrides the
semester start for a one-off.

## Verification

Three check suites, plain asserts, no dependencies — they run on the laptop and on the login node:

```bash
python3 tests/check_contract.py    # 49: dispatch, the registry, the fixture, Granite parsing, the dissenters
python3 tests/check_render.py      # 27: stage 2 against the fixture
python3 tests/check_annotate.py    # 18: the fluent view, the conjunction, disfluency tags, annotate.py end to end
python3 tests/check_rename.py      # 9:  the recording-renaming script
```

They check dispatch and structure, which is all a unit check can. **Model behaviour is verified
on real audio**: an `sbatch` run against `data/geisler.wav` at 45:00–46:30, where the reference
expectation is ~10 filled-pause markers with per-word timestamps on every token including the
markers. This project has been burned three times by a model integration that ran, produced
plausible output, and was wrong — read a model's own card and package before writing to it.

`tests/fixtures/golden.json` is one real stage-1 record — ten minutes of a proseminar guest
lecture, dual-stream, diarized — so all of stage 2 develops and tests against a committed file.
Its README has the provenance.

---

## Where things are

```
src/
  transcribe_audio.py       stage 1 CLI            src/pipeline/      the contract, registry, adapters, schema
  annotate.py               stage 1.5 CLI          src/render/        speakers, segmentation, formats, print CSS
  render.py                 stage 2 CLI            src/run_transcription.sh, transcribe.slurm   SLURM wrappers
docs/
  pipeline_plan.md          THE design document: decision log D1–D23, measured model facts, the bug queue
  environments.md           the cluster: access, mamba envs, partitions, reference audio
tests/                      the check suites and the fixture
scripts/rename_recordings.py  recorder files -> PSY<code>-week<N> names, from the semester calendar
envs/cw2diar.sh             recipe for the combined CrisperWhisper + pyannote env
hpc/                        job scripts and SLURM logs on the cluster (gitignored; its README explains)
```

`docs/pipeline_plan.md` is the authority. Its decision log records what was decided, why, and
what would reopen it; §3 records what each model was *measured* to do, which more than once
differed from what its documentation implied.

---

## Data and licences

Audio under `data/` is not committed. `transcripts/` is not committed. The one committed
transcript, `tests/fixtures/golden.json`, is a guest lecture in a proseminar course, not a
research-participant interview — there is no interview recording in the project yet.

CrisperWhisper 2's code is MIT; its weights are under Nyra Health's non-commercial research
licence, which this project's use fits. Granite and Parakeet are Apache 2.0; pyannote
community-1 is CC-BY-4.0.
