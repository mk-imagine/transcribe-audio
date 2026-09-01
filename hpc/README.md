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

Every job script here writes to `hpc/logs/` via an **absolute** `--output`:

```
#SBATCH --output=/Users/<id>/Repos/transcribe-audio/hpc/logs/%x_%j.log
```

Absolute rather than relative because `--output` resolves against the *submit*
directory, so a relative path silently scatters logs wherever `sbatch` happened
to be run from. SLURM does not expand shell variables in `#SBATCH` directives,
so `$HOME` is not an option — and these files are gitignored and exist only on
the cluster, so a machine-specific path costs nothing.

Reference audio under `data/` is human-subjects material. Job scripts name those
paths, which is another reason nothing in here is committed.

Production runs go through `src/run_transcription.sh`, not these scripts, and it
writes to `hpc/logs/` too. It derives every path from its own location, so it
works from any directory. A direct `sbatch src/transcribe.slurm` must be run
from the repo root; it will refuse with a clear message otherwise.
