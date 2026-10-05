# Diarization model bake-off (pre-registration)

**Status:** pre-registered 2026-10-05, **before any challenger has run on a lecture**. Nothing
has been run. This file fixes the arms, the data path, the speaker-map rule, the metrics and
the bar. The sections above §9 are not edited after the arms run. Results go in §9. Anything
changed after a result is seen is logged there as **post-hoc**, with the numbers before and
after, and the pre-registered run stays the headline.

**Why now.** On 2026-08-31 the project chose to stay on pyannote `community-1` over DiariZen
(`pipeline_plan.md` §10, *Diarization model choice*). That comparison was on 10-minute
excerpts, and it was explicitly "No DER computed: no reference labels". Reference labels now
exist: Mark labeled every speaker change in **PSY777-F26-WK2-Mon** (93.6 min), and that gold
is final (`diarization_repair_plan.md` §1.7). Against it, the current pipeline (raw turns,
then stage 2's `assign` plus `smooth`) prints **56 of 201 student words (27.9%) as the
instructor's**. Mark approved a bake-off: community-1 against **DiariZen** and **NVIDIA
Sortformer**, each through the **same** stage 2, scored by the **same** scorer.

The 2026-08-31 decision stands until this bake-off reports.

Commit order on the branch: this file alone, then the tooling (`scripts/diarization_bakeoff.py`,
`hpc/bakeoff/`, `envs/bakeoff-*.sh`, `tests/check_bakeoff.py`).

---

## 1. The arms

Every fact below was read on 2026-10-05 from the primary source linked next to it: the model
card, the README, the package source at the pinned commit, or the Hugging Face API for
revisions and licenses. **[V]** marks a fact verified there. **[I]** marks an inference, and
the reason is given. **[P]** marks a fact measured in this project. These postdate common
knowledge cutoffs. Don't "correct" them from memory (D19).

### 1.1 Arm 1: pyannote community-1 (the current diarizer)

| | |
|---|---|
| Checkpoint | `pyannote/speaker-diarization-community-1` @ `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` [V: HF API, last modified 2025-09-29]. The same revision stage 1 recorded for the gold's record (job 49864) [P] |
| License | CC-BY-4.0, gated (needs `HF_TOKEN`) [V: card, API] |
| Package | `pyannote.audio` **4.0.3**, in the production env `cw2diar` [P: the record's `run.packages`]. PyPI's latest is 4.0.7 [V]. It's not used: this arm is what production runs |
| Loader | stage 1's own `src/pipeline/diarize.py` (`Diarizer.load`), called the way `Diarizer.run` calls it: decode once with `pyannote.audio.Audio()`, pass `{waveform, sample_rate, uri}` |
| Input | "mono audio sampled at 16kHz". Stereo is downmixed and other rates resampled automatically [V: card]. Our WAVs are mono 48 kHz |
| Speakers | no cap documented. Call-time `num_speakers`, `min_speakers`, `max_speakers` [V: card; `apply()` in the 4.0.3 source] |
| Length | none documented [V: card is silent]. 94-minute and 145-minute lectures diarize in minutes since the decoded-waveform fix [P: `pipeline_plan.md` §10] |
| Output | `DiarizeOutput` with `speaker_diarization` (turns may overlap) and `exclusive_speaker_diarization` ("does not contain overlapping speech turns", "adapted to downstream transcription") [V: 4.0.3 source]. Stage 1 stores `speaker_diarization`, iterated at full float precision |
| Determinism | jobs 48770 and 49864 ran on the same audio three weeks apart, one given a path and one a decoded waveform. They produced **670 of 670 identical turns** [P: compared 2026-10-05] |

Sources: <https://huggingface.co/pyannote/speaker-diarization-community-1>,
<https://github.com/pyannote/pyannote-audio/blob/4.0.3/src/pyannote/audio/pipelines/speaker_diarization.py>,
<https://pypi.org/project/pyannote.audio/>.

### 1.2 Arm 2: DiariZen

| | |
|---|---|
| Checkpoint | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` @ `027b3e221b9d81dce2be794084f1dbd3ba2da403` [V: HF API]. Plus the embedding model its loader fetches, `pyannote/wespeaker-voxceleb-resnet34-LM` @ `837717ddb9ff5507820346191109dc79c958d614` (CC-BY-4.0, not gated) [V: `inference.py`, HF API], and the PLDA files inside the checkpoint repo |
| Why v2, large | The v2 card calls `-md` (v1) "the original (legacy) model", and v2 "supports up to four overlapping speakers" [V]. On the README's far-field single-channel sets, which are closest to one recorder at the front of a room, v2 is best or tied: AMI-SDM 13.9 / AliMeeting far 10.8 / NOTSOFAR-1 16.7 DER, against v1's 14.0 / 12.5 / 17.9 [V: README, collar 0]. `base-s80` is worse on every set |
| License | weights **CC-BY-NC-4.0**, "for research and academic purposes only". Code MIT [V: README, card]. This is acceptable under D15 (non-commercial academic research). Switching would make a non-commercial license part of default stage 1, so D15's trigger (work becomes commercial, or tooling ships with weights) would then reopen the diarizer too. The card speaks to the weights. Whether outputs inherit the NC term isn't addressed [I: the license text governs the weights only] |
| Package | `diarizen` 0.0.1 ("Development Status :: 2 - Pre-Alpha"), from git, **BUTSpeechFIT/DiariZen @ `844f5555b0a98acd0931511fc641a8c5b8ba92c7`** (main, 2026-08-04) [V: GitHub API, `pyproject.toml`]. The v2 card says "make sure to pull the most recent code before using the model, as outdated versions may lead to unexpected issues or degraded results" [V]. The README pins torch 2.1.1 / torchvision 0.16.1 / torchaudio 2.1.1 (cu121). `constraints.txt` pins numpy 1.26.4. The vendored `pyannote-audio` fork is 3.1.1 [V] |
| Loader | `DiariZenPipeline.from_pretrained` takes **no revision**. It is `snapshot_download(repo)` plus `hf_hub_download(wespeaker, pytorch_model.bin)` plus the constructor [V: `diarizen/pipelines/inference.py`]. The runner does those two downloads itself, pinned, then calls the same constructor. That's line for line what `from_pretrained` does |
| Input | a path, `BytesIO` or `ProtocolFile`. `torchaudio.load`, then **the first channel only** (`waveform[0]`, "force to use the SDM data"). A stereo file is *not* downmixed [V: source]. Resampling to 16 kHz happens in the vendored pyannote `Audio.downmix_and_resample` [V: `core/io.py`], and the model's `sample_rate` is 16000 [V] |
| Speakers | `config.toml [clustering.args] min_speakers = 1`, `max_speakers = 20`. At most 4 per chunk and per frame (v2) [V]. **No call-time speaker count**: `__call__(in_wav, sess_name=None)`. Bounds can change only through `config_parse` at construction [V: source] |
| Length | none documented. The model slides 16 s windows with step 0.1 × 16 s = 1.6 s, keeps the whole waveform in memory, and runs VBx over every window's embeddings [V: config, source]. This project hasn't run it at full lecture length: the 2026-08-31 comparison used 10-minute excerpts [P]. 94 min is about 3,500 windows, well within an A100 and 64 GB of RAM [I: arithmetic from the config] |
| Output | a `pyannote.core.Annotation` with integer labels. With `rttm_out_dir` and `sess_name` it also writes `<sess_name>.rttm` via `to_rttm()` [V] |
| POLARIS | the earlier `diarizen` env needed `LD_LIBRARY_PATH=$env/lib`: its `libicu` wants `CXXABI_1.3.15`, which Rocky 8's system `libstdc++` lacks [P: `environments.md`] |

Sources: <https://github.com/BUTSpeechFIT/DiariZen> (`README.md`, `diarizen/pipelines/inference.py`,
`constraints.txt`, `pyannote-audio/pyannote/audio/core/io.py`),
<https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2>,
<https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md>.

### 1.3 Arm 3: NVIDIA Sortformer, as `Nemotron-3-Diarization`

**The checkpoint is a deliberate choice, and it is not the obvious one.** The Sortformer
checkpoints on the Hub:

| Checkpoint | Mode | Speakers | Long audio | License | Verdict |
|---|---|---|---|---|---|
| `nvidia/diar_sortformer_4spk-v1` | offline | 4 | "For an RTX A6000 48GB model, the limit is around 12 minutes" [V: card] | CC-BY-NC-4.0 | **cannot take a 94-minute lecture natively**. Out |
| `nvidia/diar_streaming_sortformer_4spk-v2` | streaming | 4 | several hours | CC-BY-4.0 | superseded by v2.1 |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` (`cd03eee9…`) | streaming | 4 | "designed for long-form audio and can handle recordings that are several hours long, performance may degrade on very long recordings" [V] | NVIDIA Open Model License | the obvious pick, but its card opens by redirecting to the next row |
| **`nvidia/Nemotron-3-Diarization`** | streaming and offline-style | **8** | "With chunked inference, the maximum audio duration is not limited" [V] | **OpenMDW-1.1** | **chosen** |

The v2.1 card's first line: "A new version of Sortformer diarization model
Nemotron-3-Diarization has been released, supporting 8 speakers and greatly improved
accuracy" [V]. It is Sortformer by NVIDIA's own account. Its card says the model "Following
Sortformer … resolves speaker permutation by ordering its output channels according to each
speaker's first arrival", "adopts the Arrival-Order Speaker Cache (AOSC) and FIFO queue
introduced in Streaming Sortformer", and loads as the same NeMo class,
`SortformerEncLabelModel` [V]. It is chosen over v2.1 for four reasons:

1. **8 speakers instead of 4** [V]. A lecture has one instructor and an unknown number of
   students.
2. **10 ms output instead of 80 ms** [V: "The default frame stride is 10 ms"]. This
   project's measured weakness is boundaries a word or two off the true change: 8.8% exact
   precision, and most spurious boundaries are 1–3 words off (§1.7 of the repair plan). Words
   here last about 0.3 s, so an 80 ms frame is a quarter of a word.
3. On the card's own tables, at the same 30.4 s latency, against v2.1 [V; forced-alignment
   reference labels]:
   - AMI SDM: 11.14 vs 21.42 DER;
   - NOTSOFAR1 SC (single channel, far field): 11.00 vs 30.49 DER.
4. OpenMDW-1.1 is permissive and allows commercial use [V: <https://openmdw.ai/license/1-1/>].
   So it carries no D15 caveat.

Against it: it was released **2026-09-23**, twelve days ago, and NeMo supports it only on
`main` (below). If Mark would rather have v2.1, or both, that is his call before running
(§1.4).

| | |
|---|---|
| Checkpoint | `nvidia/Nemotron-3-Diarization` @ `f667ed73aee57d40cc39428eb768b4fd87a0a29e`, file `Nemotron-3-Diarization.nemo` [V: HF API]. 100M parameters [V] |
| Package | NVIDIA NeMo (`nemo_toolkit[asr]`), **from git: NVIDIA-NeMo/Speech @ `cf724ac337d1ebc7d0dda1e23fb80916f52927a5`** (main on the model's release day, 2026-09-23) [V: GitHub API]. Python 3.12: the card says "Python 3.12 or later" [V]. PyPI says `>=3.10` |
| **The D19 trap** | The card says "Runtime Engine(s): NeMo Framework v3.0" and installs `nemo-toolkit[asr]` from PyPI [V]. PyPI's latest, **3.0.0** (2026-08-07, tag `v3.0.0` = `fd6a8775`), **predates this model's inference support**. The model-level `_check_streaming_parameters()` that the quick start calls, and the 10 ms upsampling head (`upsample_factor`, `high_resolution`), first appear in NVIDIA-NeMo/Speech #16032 (`71820fbc`, 2026-08-15). The last change to the Sortformer model file before release is `085f1062` (2026-09-16) [V: the model source at each of those commits]. A 3.0.0 install would fail to load the model. With the card's own `restore_from(..., strict=False)`, it could also load part of it and run [I: `strict=False` skips missing and unexpected keys]. So the runner loads with NeMo's default `strict=True`, and the env recipe asserts the commit and the method |
| Input | "16 kHz, single-channel audio in `.wav`, `.flac`, `.opus`, or `.mp3`". A path, a numpy array (with `sample_rate`), or a manifest [V: card]. A path is loaded by NeMo's `WaveformFeaturizer` at the preprocessor's rate [V: `_setup_diarize_dataloader` passes `self.preprocessor._sample_rate`]. That it resamples a 48 kHz file correctly is read from NeMo's audio loader, not traced end to end [I] |
| Speakers | 8, architectural: the output is `[T, 8]` [V]. `DiarizeConfig.max_num_of_spks` defaults to `None` [V: source]. No count is passed |
| Length | "With chunked inference, the maximum audio duration is not limited" [V]. The card's **"Very high latency (offline)"** configuration, a 30.4 s input buffer in 80 ms frames: `SPKCACHE_LEN` 264, `FIFO_LEN` 40, `CHUNK_LEN` 340, `RIGHT_CONTEXT` 40, `UPDATE_PERIOD` 300 [V]. This is still chunked (streaming-mode) inference, so the whole lecture goes in one `diarize()` call. **Not used:** the card's Transformers route. It needs transformers from source, and its offline example runs the whole input through the encoder at once, while `config.json` has `max_position_embeddings: 5000` encoder frames, about 400 s at 80 ms [V: config.json. Reading 5000 as a length bound for that route is an inference [I]]. The runner refuses a checkpoint that loads with `streaming_mode` off, so this arm cannot fall into a non-chunked path |
| Call | `restore_from(path, map_location="cuda")`, `eval()`, the five streaming parameters set on `sortformer_modules`, `_check_streaming_parameters()`, then `diarize(audio=[path], batch_size=1)`, as in the quick start [V]. Defaults otherwise: `postprocessing_yaml=None`, fp32. The card's DER tables used bf16 [V] |
| Output | per file, a list of strings `f"{start:.3f} {end:.3f} speaker_{k}"` (NeMo's `generate_diarization_output_lines`) [V: source at the pin]. The card's prose says "'begin_seconds, end_seconds, speaker_index'" [V], but that is not the code's format. The converter parses the code's format and refuses anything else |

Sources: <https://huggingface.co/nvidia/Nemotron-3-Diarization>,
<https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>,
<https://huggingface.co/nvidia/diar_sortformer_4spk-v1>,
<https://github.com/NVIDIA-NeMo/Speech/blob/cf724ac337d1ebc7d0dda1e23fb80916f52927a5/nemo/collections/asr/models/sortformer_diar_models.py>,
<https://github.com/NVIDIA-NeMo/Speech/pull/16032>, <https://pypi.org/project/nemo-toolkit/>.

### 1.4 Optional arms: Mark's call before running, not part of this bar

Each would be an arm beyond the three approved. None is scored unless Mark adds it before the
first job runs.

- **1x: community-1's `exclusive_speaker_diarization`.** It comes out of the same call, at
  zero install cost, and the community-1 job writes it anyway (`community1-exclusive/`). It
  matters here because the 44-word missed exchange was lost to **overlap**: a raw SPEAKER_00
  turn lost its first word to an overlapping SPEAKER_01 turn by 5 ms (repair plan §1.8).
  Exclusive turns have no overlap by construction. If Mark adds it, it is scored and barred
  exactly like a challenger. Adopting it would change one line of `diarize.py`, not an
  environment.
- **3b: `nvidia/diar_streaming_sortformer_4spk-v2.1`**, in the same env as arm 3, with its
  card's 30.4 s configuration (`spkcache_len` 188). Only if Mark wants the obvious Sortformer
  checkpoint as well as, or instead of, Nemotron-3.
- Not proposed: a `max_speakers` passthrough for community-1 (`pipeline_plan.md` §10). It
  answers a different question, fragmentation into many labels. On this lecture community-1
  already emits only 2 labels.

### 1.5 The speaker-count policy, one for every arm

**Each arm runs at its documented default, and no arm is given a speaker count**: no
`num_speakers`, no `min_speakers`, no `max_speakers`, and certainly not the gold's (one
instructor, two students).

- It is how production runs. Stage 1 calls community-1 with no count, and the comparison is
  "the current pipeline vs a replacement".
- A count is unknown in advance in production. How many students speak in a given lecture
  can't be known before transcribing it, so any hint would be a guess, or an oracle.
- It is the only policy all three can share without changing an arm's call:
  - DiariZen has no call-time count at all (§1.2);
  - Nemotron-3's limit is architectural (8);
  - a common `max_speakers` would have to be at most 8, and that would change community-1
    away from production.
- What remains unequal is stated, not equalized. Nemotron-3 can't exceed 8 speakers and
  DiariZen can't exceed 20, while community-1 has no cap. The gold has 3 identities, so no
  cap binds here.

---

## 2. The data path

**The words stay fixed. Only the diarizer changes.**

1. **Run** (POLARIS, one job per arm, `hpc/bakeoff/<arm>.slurm`). Each arm runs on the same
   audio file stage 1 was given, `data/PSY777-F26-WK2-Mon.wav`. The runner
   (`hpc/bakeoff/run_arm.py`) refuses to proceed if a pinned revision or commit is not what
   loaded. It writes the native output verbatim and a `provenance.json` that records:
   - the model id and revision;
   - **the package the arm ran with and its commit or version** (D19);
   - the call and its parameters, and the speaker-count policy;
   - the audio's sha256, sample rate and channels;
   - load time and diarization time (decoding included);
   - torch's peak GPU memory, plus `nvidia-smi` samples every 500 ms.
2. **Two runs per arm** (`run1`, `run2`), in separate processes in the same job. The second is
   the determinism check. If an arm's two runs differ, both are scored and the bar is applied
   to the **worse** one.
3. **Outputs are frozen.** The job never overwrites a run directory that already has a
   `provenance.json`. The scorer refuses a native output whose sha256 differs from the one its
   provenance recorded.
4. **Convert** (`scripts/diarization_bakeoff.py`, stdlib):

   | Arm | Native output (scored) | Conversion to `speaker_turns` |
   |---|---|---|
   | community-1 | `DiarizeOutput.speaker_diarization`, iterated as stage 1 iterates it, written as `turns.json` at full float precision | identity: `{start, end, speaker}` |
   | DiariZen | the returned `Annotation`, iterated with `itertracks(yield_label=True)`, as the README does, written as `turns.json`. The pipeline's own RTTM is kept beside it | identity. Integer labels become strings |
   | Nemotron-3 | `diarize()`'s strings, one per line, verbatim (`segments.txt`) | `START END speaker_N` becomes `{start, end, speaker: "speaker_N"}` |

   RTTM (`SPEAKER file chnl tbeg tdur … name …`, end = `tbeg + tdur`) is also accepted, for
   any re-run that has only an RTTM. The pyannote-family arms are scored from the
   full-precision iteration, not the RTTM: the RTTM is rounded to 1 ms, and a near-tie in
   `assign` can be decided by less than that. The 44-word exchange was lost by 5 ms. Turns are
   sorted by (start, end, label). A malformed line is refused, never skipped.
5. **Swap.** The gold's raw JSON (`transcripts/lectures/PSY777-F26-WK2-Mon_job49864_raw.json`)
   is copied:
   - `words`, `secondary_stream` and `text` stay byte-identical;
   - `speaker_turns` is replaced by the arm's turns;
   - `diarization` is replaced by the arm's provenance, and a warning names the swap.

   The copy is written as `<stem>_bakeoff-<arm>-<run>_raw.json` and validated against schema
   v1. **The original is never rewritten** (D3).
6. **Stage 2, unchanged.** It is the render the gold was labeled on:
   - the lecture profile's `intended` stream;
   - `assign` (max overlap, nearest turn in gaps);
   - `smooth` at `pause_threshold` 0.5 s, `max_island` 2;
   - no speaker map.

   It is regenerated by `diarization_labels.build`, the function that produced the labeling
   sheet. The scorer refuses an arm whose rendered word sequence differs from the gold's.
7. **Score** (§3, §4) on the Mac, with no model.

**PSY498-F26-WK3-Tue** is run by the same jobs at the same time (§6.3). It is frozen, and not
scored until it is labeled.

---

## 3. The speaker-map rule

The scorer needs to know which identity (`L`, `S`, `S2`) each diarizer label stands for.
§1.7 used Mark's map for community-1, `SPEAKER_01=L,SPEAKER_00=S`. Challengers have
different labels and counts, so the map is **computed from the gold, by one rule, for every
arm, community-1 included**:

1. Over the scored words (true speaker known, not `?`), count the words each rendered label
   shares with each gold identity.
2. **Pair labels with identities one to one, maximizing the shared words.** Only pairs that
   share at least one word are used. This is the DER convention (`pyannote.metrics` maps
   hypothesis to reference speakers this way). It is computed exactly, by dynamic programming
   over identity subsets, not greedily. Ties go to the assignment that comes first in label
   order, with identities ordered L, S, S2…S9.
3. **Every label left over is folded** into the identity it shares the most words with. Ties
   follow the same order, and a label with no scored word goes to L. This is what the
   render-time speaker map allows: two labels may name one person (`pipeline_plan.md` §6).

**Why one-to-one first, and not plain majority.** On this lecture, community-1's student
label `SPEAKER_00` carries **159 instructor words against 145 student words** (130 `S`,
15 `S2`). Majority would map it to L, which prints every student word as the instructor's,
and the baseline would read 100% instead of 27.9%. The one-to-one step maps it to S, as Mark
did. **Checked 2026-10-05**: the rule, run on community-1's stored turns, gives exactly
`SPEAKER_01=L, SPEAKER_00=S`. It reproduces every number of §1.7 (the identity check, §4).

**Biases it introduces, stated now:**

- **It is an oracle.** It uses the gold, so it measures the diarizer's *partition* and not
  anyone's ability to name labels. It is equally optimistic for every arm, and for
  community-1 it equals the human map. In use, someone maps labels by reading, so an arm's
  label count is a real cost and is reported.
- **The fold favors over-splitting.** Under a many-to-one fold, a finer partition can't score
  worse. An arm that scatters students over many small labels gets each one folded the
  best way. Two guards:
  - every result reports the folded labels and the words they hold;
  - the bar (§5, P2) must also hold under the **pessimistic fold**, where every leftover label
    is mapped to L.
- **Under-splitting is penalized, and should be.** If an identity has no label, as `S2` has
  none under community-1, all its words are wrong.
- **The fold can't hide the high-harm direction in a big instructor-majority label.** Such a
  label folds to L, and its student words count as the instructor's. That is what the render
  would print.

---

## 4. Metrics

The per-word truth is derived **once**, from the gold, exactly as `score_diarization.py`
derives it, with `--gold-map SPEAKER_01=L,SPEAKER_00=S`. The words are the same for every
arm, so the truth is the same for every arm.

An arm's **boundaries** are its own rendered label changes. Each gets a verdict read off the
truth: `real` where the true speaker changes at that word, `spurious` where it does not,
`unsure` where either side is unknown. This is the gold's own convention (repair plan §1.7,
answer 3), which holds on all 68 of community-1's boundaries. Each arm is then scored by
**`score_diarization.score`, unchanged**.

**The identity check comes first and gates everything.** Community-1's stored turns are put
through this whole path: swap, render, derived verdicts, computed map. Every field of the
scorer's output must equal what `score_diarization.py` reports on the gold, or the run is
refused. Run on 2026-10-05 against the real gold, it passes: 56/201, accuracy 98.39%, 68
boundaries, precision 8.8%, 16 missed changes, the map above. No challenger output existed
yet; this validates the harness, not a result.

**Primary:** the share of student words printed under an instructor-mapped label
(`words.student_as_instructor` of `words.student_words`). Community-1: **56 of 201 (27.9%)**.

**Secondary**, all reported for every arm and run:

- **High-harm runs.** These are the scorer's error runs (consecutive wrong words with one true
  speaker and one label) whose direction is student-under-instructor, whatever their cause.
  Community-1: **6**. That is the 5 missed changes of §1.7 (54 words) plus one 2-word run from
  a spurious split.
- **Missed changes:** count, how many misattribute words, and how many words; and the
  scorer's `student_in_instructor_turn` count (community-1: 16; 10 misattribute 78 words; 5
  high-harm).
- **Boundaries and their precision.** Precision is `real / (real + spurious)` at the exact
  change, and within N = 1, 2, 3 words with the scorer's one-to-one near-miss pairing.
  Community-1: 68 boundaries; 8.8%, 19.1%, 22.1%, 23.5%.
- **Per-word accuracy**, by direction and by cause. Community-1: 98.39%.
- **Labels.** Raw labels in the turns, labels in the render, the map, the folded labels and
  their words, and the pessimistic-fold primary.
- **Runtime and memory:**
  - load time;
  - diarization time and the realtime factor;
  - torch's peak allocated and reserved GPU memory;
  - `nvidia-smi`'s peak, for each run.
- **Determinism.** Whether the two runs' turns are identical (sha256 of the canonical turns).
- **Audit list** (§6.2): spurious boundaries with no community-1 boundary within 3 words.

---

## 5. The bar

All numbers are fixed now. Against community-1's **56 / 201 student words, 6 high-harm runs,
98.39% accuracy**, a challenger is a **switch candidate only if every line holds**:

| | Condition | Limit on PSY777 |
|---|---|---|
| **P1** | Student words under the instructor (primary rule) at most **half** of community-1's | **≤ 28 of 201 (≤ 13.9%)** |
| **P2** | The same under the pessimistic fold (every leftover label → L) | **≤ 28** |
| **P3** | High-harm runs at most **half** of community-1's | **≤ 3** |
| **G1** | Per-word accuracy no more than **1.0 point** below community-1's | **≥ 97.39%** (at most about 143 more wrong words) |
| **G2** | Both runs complete the whole lecture in one call, natively, by the card's or README's own route, on one A100 80 GB, inside the job's 4 h, with every pin verified | — |

The limits are relative (`floor(baseline × 0.5)`, accuracy − 0.010), so the same rule applies
unchanged to PSY498. The constants live in `BAR` in `scripts/diarization_bakeoff.py`, and
`tests/check_bakeoff.py` asserts them.

**Outcomes:**

- **Keep community-1**, if no challenger meets every line. The 2026-08-31 decision then
  stands, now on measured evidence. A challenger with *more* than 56 words is reported as
  worse than community-1.
- **Switch candidate**, if exactly one challenger meets every line. That is a recommendation
  to Mark, subject to PSY498 (§6.3). The switch itself is an engineering task: a diarizer that
  declares what it provides (D2) in its own environment, and its provenance in the record.
  This bar doesn't decide that design.
- **Close call**, if both challengers meet every line within **5 words** of each other on P1.
  Then the lower install cost decides, and this plan rates **Nemotron-3** lower:
  - its license is permissive (no D15 caveat);
  - NeMo is a maintained package, needing a pinned commit, not a fork;
  - no torch 2.1.1 pin;
  - no `LD_LIBRARY_PATH` workaround.

  If they differ by more than 5 words, the lower P1 wins.

**Why halving, and why P3.**

- **One lecture shows only large differences.**
  - The primary is lumpy: one exchange holds 44 of the 56 words, so one fix moves the share by
    22 points.
  - A challenger that fixes only that exchange reaches 12 words and passes P1 on one event.
  - P3 needs the high-harm runs to halve too (6 → ≤ 3): at least three of six fixed, net of any
    it creates. So one lucky exchange can't carry a switch.
- **The install cost is real.**
  - Either challenger means a second environment and a separate diarization step in stage 1.
    CrisperWhisper runs in `cw2diar`, and neither challenger can share it: DiariZen pins torch
    2.1.1 and a vendored pyannote 3.1.1, and Nemotron-3 needs Python 3.12 and NeMo from `main`.
  - On 2026-08-31 a no-better result for DiariZen was judged not worth exactly this install
    (`pipeline_plan.md` §10).
  - Halving the high-harm error is the size of gain that pays for it. A few words' improvement
    does not.
- **Run-to-run noise is not what limits this.** Community-1 is deterministic (§1.1). The
  challengers' two runs measure their own noise, and the bar takes the worse. The limit is
  sampling: 201 student words, 22 true changes, one room.
- **G1 keeps the high-harm gain from being bought with the low-harm error.** Spurious splits
  are low harm in a lecture (repair plan §1.1). Still, an arm that halves the high-harm words
  by labeling half the instructor's speech as students is not better.

---

## 6. Caveats, stated in advance

### 6.1 One lecture, one room

- One course, one instructor, one recorder position (a Tascam DR-40X, internal X/Y mics, near
  the front), and one 94-minute session.
- The instructor speaks 14,069 of the 14,270 words. The 201 student words come from two
  students. Five missed changes carry 54 of the 56 high-harm words, and one exchange carries 44.
- Everyone is far-field: the lecturer isn't individually mic'd, and the students are farther
  away and quieter. That suits models trained on far-field single-channel sets (AMI-SDM,
  NOTSOFAR-1 SC), but the published DERs are on meetings, not lectures.
- The published DERs were scored against different reference labels, so they are not
  comparable across cards (Nemotron's card says so of its own).
- PSY498 (§6.3) is a second course and a second session. It is the only out-of-sample check
  this plan has.

### 6.2 The gold was labeled on community-1's render

Mark labeled what `diarization_labels.py` showed him: community-1's turns after `assign` and
`smooth`. Could that bias the comparison? Reasoned through:

- **Part A (each boundary judged) does not favor community-1.** Every verdict became a
  per-word truth, and the truth is the same for every arm. A boundary community-1 placed a
  word or two off was marked `spurious`, with the true change marked at the right word. A
  challenger is scored against the true position, not against community-1's.
- **Part B (missed changes, found by reading) does.** A change community-1 drew no boundary
  for had to be found in the text. Any change Mark did not find is "no change" in the gold. By
  construction, such a place has no community-1 boundary, so it is a community-1 error the
  scorer can't see. A challenger that gets it right is scored as **wrong** there: a spurious
  boundary, and its words as instructor-under-student.
- **Such misses happen.** The 44-word exchange had no marker in the first pass. It was found
  on review (answer 4).
- **Direction: the bias favors community-1, so it works against switching.**
  - On the primary, an unfound student word counts as instructor speech for every arm. It
    leaves the student-word count, so no arm gains or loses primary credit for it.
  - On G1 and on precision, it penalizes only the arm that was right.
  - The comparison is therefore conservative. A challenger that passes, passes despite a gold
    whose blind spots are community-1's.
- **What is done about it, without touching the final gold:** the **audit list**.
  - Every challenger boundary the gold calls spurious with **no community-1 boundary within 3
    words** is listed by word index and time (no text). These are the places judged only by
    reading.
  - Mark can check them by ear.
  - Any change confirmed there is reported in §9 as a **post-hoc** sensitivity result, with
    the pre-registered numbers kept as the headline. The gold file is not edited.

### 6.3 PSY498-F26-WK3-Tue is the second test set, held out

- Every arm runs on `data/PSY498-F26-WK3-Tue.wav` (49.9 min) **in the same jobs as PSY777**,
  before that lecture is labeled. The outputs are frozen like PSY777's, and their sha256s are
  recorded in §9 when scored.
- **Nobody looks at any arm's PSY498 output, or scores it, until its labels are final.** The
  labeling sheet is generated from the community-1 stage-1 record, as for PSY777. The repair
  plan's held-out measurement of the candidate generator (repair plan §1.8) needs exactly that
  sheet, and is unaffected.
- When the labels are final, **every arm is re-scored there unchanged**: the same frozen
  outputs, the same driver, the same rule, the same relative bar against PSY498's own
  community-1 baseline. Pooled numbers are reported. They are not a new bar.
- **What PSY498 can do to the decision:**
  - a switch recommended on PSY777 is **reopened** if that arm prints more student words as the
    instructor's than community-1 on PSY498 (P1 worse);
  - a "keep" is **reopened** if a challenger meets the whole bar on PSY498.
- If PSY498's community-1 baseline has fewer than 3 high-harm runs, P3's limit there is 0 or 1
  and the confirmation has little power. That is reported, not adjusted.

### 6.4 Smaller points

- **Stage 2 is held fixed on purpose.** An arm whose advantage needs a different assignment
  rule (exclusive turns, say) is measured under today's rule. That is the right question for
  "replace the diarizer", and the wrong one for "redesign stage 2".
- **Recency.** Nemotron-3 is twelve days old, and its NeMo support is a `main` commit, not a
  release. A pass is a reason to switch only once NeMo releases that support, or the pin is
  accepted as the dependency.
- **Output resolution differs**: pyannote's turns are continuous, DiariZen's are frame-based,
  and Nemotron-3 reports at 10 ms with three decimals. `assign` works on overlap seconds, so
  resolution is part of what is measured, not a nuisance to remove.

---

## 7. A conditional direction arm (only if stereo originals exist)

Mark hasn't said whether the recorder's **stereo** originals survive. Every WAV in `data/` is
mono 48 kHz/16-bit. If they do, a direction signal exists that no arm above uses. Outside the
bar, and briefly:

- **What the signal is.** An X/Y pair is coincident, so its channels differ in **level**, not
  in arrival time. Direction is an inter-channel level difference per frame. The lecturer at
  the front and students in the room should sit at different angles to the pair.
- **What feeds the arms meanwhile.** None of the three arms would use it. Community-1 and NeMo
  downmix, and DiariZen keeps only channel 0. The arms keep getting the mono files.
- **Spike first** (the house rule):
  1. Confirm the mono file is a sample-aligned downmix of the stereo original, so the gold's
     word times hold on the stereo timeline.
  2. Measure each gold word's level difference.
  3. Report how well it separates `L` words from student words (ROC AUC), on PSY777, whose
     gold already gives a speaker per word.
- **Only if it separates strongly** does it become a design: most likely a candidate signal
  for the repair plan's step 5, not a stage-1 diarizer. It would get its own pre-registration
  and would never be compared against this bar.
- The originals are human-subjects material, handled like the WAVs.

---

## 8. How to run it (when Mark says go)

```bash
# On POLARIS, login node, from the repo root. cw2diar already exists (envs/cw2diar.sh).
bash envs/bakeoff-diarizen.sh > scratch/logs/env_bakeoff-diarizen.log 2>&1
bash envs/bakeoff-nemo.sh     > scratch/logs/env_bakeoff-nemo.log 2>&1

# Three jobs, each both lectures x two runs. QOS: 2 run at once, the third waits.
for A in community1 diarizen sortformer; do
  sbatch hpc/bakeoff/$A.slurm data/PSY777-F26-WK2-Mon.wav data/PSY498-F26-WK3-Tue.wav
done

# On the Mac: copy transcripts/bakeoff/ back (no text in it, but it stays in gitignored
# transcripts/), then score PSY777 only.
S=PSY777-F26-WK2-Mon
python3 scripts/diarization_bakeoff.py score \
    transcripts/lectures/${S}_job49864_labels.normalized.json \
    --gold-map SPEAKER_01=L,SPEAKER_00=S \
    transcripts/bakeoff/$S/{community1,diarizen,sortformer}/run{1,2}/provenance.json \
    --out-dir transcripts/bakeoff/$S/scored --json transcripts/bakeoff/$S/bakeoff-score.json
```

`tests/check_bakeoff.py` covers the converters, the map rule, the swap, the identity check,
the scoring, the bar and the runner's stdlib side. It needs no model and no GPU.

---

## 9. Results

Not run.
