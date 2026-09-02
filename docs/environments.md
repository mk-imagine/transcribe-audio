# Environments and operational notes

Machine-level facts that `pipeline_plan.md` deliberately leaves out. Verified 2026-08-31.
Re-verify versions before relying on them.

---

## POLARIS (SFSU HPC) — primary compute

Login node `n1.hpc.at.sfsu.edu`, Rocky Linux 8.10, glibc 2.28, miniforge3 with
**mamba 2.0.8**. Home is shared NFS (`/Users/<id>`), **`/tmp` is node-local** — files written
there by a batch job are not visible from the login node.

> ### Use `mamba`, not `conda`, for all environment management
>
> Create, install, remove, activate and run through mamba. It resolves far faster
> and is what `src/transcribe.slurm` already activates with.
>
> ```bash
> source $HOME/miniforge3/etc/profile.d/mamba.sh   # for `mamba activate`
> mamba create -n <env> python=3.11 -y
> mamba install -n <env> -c conda-forge <pkg>
> mamba run -n <env> python script.py
> mamba env remove -n <env> -y
> ```
>
> In non-interactive `ssh -n` commands the shell function may not be initialised;
> call the binary directly at `$HOME/miniforge3/bin/mamba`, or source
> `mamba.sh` first. `pip` still runs inside an activated env as usual.

### Access

Password auth from the Mac (`ssh polaris`, entry already in `~/.ssh/config` with
`ControlMaster`). Non-interactive commands work over the control socket:

```bash
ssh -n -o BatchMode=yes polaris '<command>'
```

`-n` matters: without it, anything on the remote that reads stdin will consume it.

**`~/.bashrc` loads the GitHub key into an ssh-agent, guarded to interactive shells only:**

```bash
if [[ $- == *i* ]] && [ -t 0 ]; then
    [ -z "$SSH_AUTH_SOCK" ] && eval "$(ssh-agent -s)" >/dev/null
    ssh-add -l >/dev/null 2>&1 || ssh-add ~/.ssh/id_ed25519
fi
```

The guard is load-bearing. Unguarded, `ssh-add` runs for non-interactive sessions too, and
with no tty it reads the passphrase **from stdin** — swallowing whatever the caller was
sending. That broke VS Code Remote-SSH (17 s timeout) and silently ate a script piped over
ssh. Consequence: **`git` over SSH does not authenticate in batch jobs or `ssh -n` commands**,
since the agent is not loaded there. Fetch over HTTPS instead (the repo is public):

```bash
git fetch https://github.com/mk-imagine/transcribe-audio.git <branch>
```

### SLURM

| partition | limit | nodes | GPUs |
|---|---|---|---|
| `gpucluster` | 8 h | 2 | 4× A100 80 GB each |
| `gpuquick` | 2 h | 1 | 4× A100 |
| `cpucluster` | 18 h | 5 | — |
| `highmem` | ∞ | 3 | — |

`PriorityType=priority/basic` (strict FIFO — no fairshare, no age weighting) with
`sched/backfill`. A shorter `--time` earns **no** priority; it only helps by fitting a
backfill gap. `gpucluster` is usually idle, so jobs typically start immediately.

Chain dependent work rather than polling: `sbatch --dependency=afterany:<jobid>`.

**Gotcha:** `echo "EXIT=$?"` after `cmd | tail` captures `tail`'s status, not the command's.
Use `${PIPESTATUS[0]}`.

### Conda environments

| env | torch | transformers | purpose |
|---|---|---|---|
| **`cw2diar`** | 2.8.0 | — | **Complete stage 1**: `crisperwhisper[ct2]` 2.0.2 **and** `pyannote.audio` 4.0.3 in one env, so a single run gives verbatim words and diarization. Built 2026-09-01 for the Phase 2 fixture. The two do not conflict: pyannote pins `torch==2.8.0`, and the ct2 backend needs no torch at all. No `transformers`, so no float32 CT2 conversion here — do that in `cw2native` first if a CPU run is needed. |
| **`cw2native`** | 2.13.0 | 5.16.1 | `crisperwhisper` 2.0.2 — the supported way to run CrisperWhisper (D17); the default `transcribe.slurm` activates. Carries torch+transformers, so it can convert a CT2 model to a new compute type. **No pyannote**, so runs here need `--no_diarize`. |
| `audio-transcribe` | 2.8.0 | 4.57.3 | pyannote 4.0.3 + peft. Diarization and generic models. |
| `audio-transcribe-tf5` | 2.8.0 | 5.16.1 | Granite 4.1/`granite_speech_plus`, Qwen3-ASR, Parakeet — all need transformers ≥ 5.13. |
| `audio-transcribe-crisper` | 2.8.0 | 4.37.2 | nyrahealth transformers fork for CrisperWhisper **v1**. Obsolete under D17; delete once nothing references it. |
| `diarizen` | 2.1.1+cu121 | — | DiariZen + vendored pyannote 3.1.1 fork. **Benchmarked and rejected** — keep only if revisiting diarization. |

**Never build an env with `mamba create --clone` (or `conda create --clone`) when pip will
run in it.** Clones hardlink
package files, so a pip upgrade in the clone can strip packages from the *source* env.
Installing transformers 5.x into a clone removed `regex` and `safetensors` from both parents
and broke two working environments. Build from scratch.

**`diarizen` needs an `LD_LIBRARY_PATH` override.** Its `libicu` wants `CXXABI_1.3.15`, which
Rocky 8's system `libstdc++` lacks:

```bash
LD_LIBRARY_PATH=$HOME/miniforge3/envs/diarizen/lib mamba run -n diarizen python ...
```

### Running stage 1

```bash
# from src/, in cw2native
python -u transcribe_audio.py -i ../data/geisler.wav -o ../transcripts/out \
    --start_time 00:45:00 --end_time 00:46:30 --no_diarize --mode verbatim
```

Add `--compute_type` to override the numeric type; the default is float16 on CUDA and
float32 on CPU, because ct2 refuses float16 on CPU. The first run at a new compute type
converts the CT2 model (a minute or two; needs torch+transformers, i.e. `cw2native`) and caches
it under `~/.cache/crisperwhisper/`.

Add `--dual_stream` for both renderings — verbatim *and* intended, each with word timestamps,
from one batched pass on the ct2 backend (~11% more inference than one stream, against ~100%
for a second pass). It writes a second preview file per stream.

Outputs `<name>_raw.json` (schema v1, the record) and `<name>_preview.txt` (a stage-1
preview, not the stage-2 render). Exit codes: **1** means audio ranges were lost and the
transcript has holes; **2** means the model id or the requested mode was refused.

`python transcribe_audio.py --help` lists the registered model ids. An unregistered id is
refused rather than guessed at, so a new checkpoint needs a `ModelSpec` in
`src/pipeline/registry.py` (or `--adapter`, which is recorded in the output's provenance).

`python3 tests/check_contract.py` runs 31 dependency-free checks — it needs no models, no
GPU and no packages, so it runs on the Mac and on the login node alike.

### Running stage 2

Anywhere, including the Mac with nothing installed — it is pure Python over the JSON:

```bash
python3 src/render.py transcripts/out/<name>_raw.json --profile coding      # txt + html
python3 src/render.py transcripts/out/<name>_raw.json --profile lecture --format html,plain
```

Outputs `<name>_<profile>.txt`, `.html` (print with Cmd+P; the coding margin is in the
`@page` rule) and `_plain.txt`. `--speaker-map` takes a file of `SPEAKER_00: Name` lines; a `<name>_speakers.yaml` beside
the input is used automatically, and two labels may map to one name to merge an over-split
speaker. Labels are per-recording, so each recording gets its own sidecar.
`python3 tests/check_render.py` runs the 21 stage-2 checks against the committed fixture.

### Data

`~/Repos/transcribe-audio/data/` — human-subjects material, do not copy off-cluster.

| file | duration | notes |
|---|---|---|
| `geisler.wav` | 91.7 min | Lecture, **disfluency-rich** — the reference for verbatim testing. 45:00 window has confirmed `uh`s and the "Courchesne"/"commissure" proper-noun probes. |
| `251211_0009.wav` | 80.7 min | **Proseminar guest lecture** — a visiting professor presenting their research, with host interaction. Multi-speaker, and the source of `tests/fixtures/golden.json`. *Not* a research-participant interview: there is no interview recording in `data/` yet. |
| `tate_1.m4a` | 79.0 min | Lecture, low-disfluency speaker. Only `.m4a` source; exercises the decode-path penalty (bug 9). |

`.env` holds `HF_TOKEN` (mode 600), gitignored, required for pyannote.

---

## ubuntu-gpu — local GPU box

RTX 3070 (8 GB), driver 580.173.02, Docker 29.7.2 + nvidia-container-toolkit. Reached as
`ssh ubuntu-gpu` (password auth). Repo at `~/Repos/transcribe-audio`, container setup on the
`feat/docker-gpu` branch (CUDA 13 base — matches torchcodec's PyPI wheels, which are built
against CUDA 13 and fail to load on a CUDA 12 base).

**8 GB is the binding constraint.** whisper-large-v3-class models are ~6.2 GB in float32 —
half precision is mandatory, and word-level timestamps OOM outright. This is what
`--timestamp_mode chunk` exists for; it is unnecessary on POLARIS.

That flag applies to the **generic Whisper adapter only** — CrisperWhisper 2 always requests
word timestamps, and on the ct2 backend they cost no measurable wall time. Chunk mode
coarsens the record, so it is written to `asr.granularity` in the raw JSON rather than left
for stage 2 to infer. It survived the D13 cleanup for this reason: bug 17's "chunk-level
timestamp workaround" was CrisperWhisper's chunk-level *pin*, which is gone, whereas this
flag has its own current justification.

---

## Measurement harness

Throughput measured on 300 s of audio, A100, correctly configured:

| model | RTFx | note |
|---|---|---|
| `granite-speech-5.0-470m-turboctc` | 7187 | |
| `parakeet-ctc-0.6b` | 1508 | |
| `parakeet-tdt-0.6b-v3` | 182 | |
| **CrisperWhisper 2 (package)** | **38** | vs 12 through `transformers.pipeline` |
| CrisperWhisper 2, `--dual_stream` | 33 | both streams in one batched pass; ~11% over a single stream |
| `Qwen3-ASR-1.7B` | 23 | |
| `ARK-ASR-0.6B` | 36 | 30 s audio cap → 10 calls per 300 s |

The 12-window sweep used for every comparison: windows at 5/15/30/45/60/75 min, 90 s each,
across `geisler.wav` and `251211_0009.wav`. Outputs land in
`transcripts/wide/<tag>__<file>/`. Metrics: word count, disfluency markers, duplicate-8-gram
rate (loop detection), and cross-model disagreement.
