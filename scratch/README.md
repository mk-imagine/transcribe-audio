# `scratch/` — cluster scratch

One-off job scripts, SLURM output and analysis helpers from runs on POLARIS. **Everything
here except this file and `logs/.gitkeep` is gitignored**, so it sits beside the work it
describes without ever being committed.

| | |
|---|---|
| `scratch/jobs/` | `*.sbatch`: one-off and exploratory job scripts |
| `scratch/logs/` | `*.log`: SLURM stdout/stderr, named `<jobname>_<jobid>.log` |
| `scratch/scripts/` | analysis helpers run against the outputs |

This directory exists because these files were accumulating in the cluster home
directory (45 job scripts and 53 logs at the point it was cleaned up), where they had
no connection to the repo they belonged to. Until 2026-09-22 it was `hpc/`, which now holds
the tracked SLURM entry points instead.

Every job script here writes to `scratch/logs/` via an **absolute** `--output`:

```
#SBATCH --output=/Users/<id>/Repos/transcribe-audio/scratch/logs/%x_%j.log
```

Absolute rather than relative because `--output` resolves against the *submit* directory,
so a relative path silently scatters logs wherever `sbatch` happened to be run from. SLURM
does not expand shell variables in `#SBATCH` directives, so `$HOME` is not an option. These
files are gitignored and exist only on the cluster, so a machine-specific path costs
nothing.

Reference audio under `data/` is human-subjects material. Job scripts name those paths,
which is another reason nothing in here is committed.

Production runs go through the scripts in `hpc/`, not these.
