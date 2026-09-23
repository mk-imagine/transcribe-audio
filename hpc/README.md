# `hpc/` — SLURM entry points for POLARIS

The tracked job scripts for stage 1 on the cluster. One-off job scripts, SLURM logs and
analysis helpers are scratch, and live in the gitignored `scratch/` (its README explains).

| | |
|---|---|
| `transcribe.slurm` | one recording per job. Submitted through `src/run_transcription.sh`, which builds the flags |
| `transcribe_parallel.slurm` | several recordings in one allocation, one process per file on its own GPU (1–2 per GPU). Its header has the measurements behind the per-GPU choice |

Both write their logs to `scratch/logs/`.

`src/run_transcription.sh` is the supported way to submit one file. It derives every path
from its own location and creates the log directory before submitting, so it works from any
directory.

A direct `sbatch hpc/transcribe.slurm` or `sbatch hpc/transcribe_parallel.slurm` must be run
**from the repo root**. Their `--output` is relative to the submit directory, SLURM will not
create that directory, and a job whose log file cannot be opened dies immediately *with no
log to explain why*. `scratch/logs/.gitkeep` makes sure the directory exists in every clone,
but only relative to the root.

Neither entry point changes the job's working directory: a relative `--input_path` or
`--output_dir` stays relative to wherever you invoked it, which is the only behavior that
does not surprise the caller.
