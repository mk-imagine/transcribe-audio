# `hpc/` — cluster scratch

Job scripts and SLURM output from runs on POLARIS. **Everything here except this
file is gitignored**, so it sits beside the work it describes without ever being
committed.

| | |
|---|---|
| `hpc/jobs/` | `*.sbatch` — one-off and exploratory job scripts |
| `hpc/logs/` | `*.log` — SLURM stdout/stderr, named `<jobname>_<jobid>.log` |
| `hpc/scripts/` | analysis helpers run against the outputs |

This directory exists because these files were accumulating in the cluster home
directory — 45 job scripts and 53 logs at the point it was cleaned up — where
they had no connection to the repo they belonged to.

Two things to know when adding a job script here:

- The existing scripts use `#SBATCH --output=%x_%j.log`, which is **relative to
  the submit directory**. Submitted from `hpc/jobs/`, their logs land there
  rather than in `hpc/logs/`. Use an absolute path, or `--output=../logs/%x_%j.log`.
- Reference audio under `data/` is human-subjects material. Job scripts may name
  those paths, which is another reason nothing in here is committed.

Production runs go through `src/run_transcription.sh`, not these scripts. Its
wrapper writes to `src/logs/` — a separate path, left alone for now.
