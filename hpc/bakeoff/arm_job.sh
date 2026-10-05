#!/bin/bash
# Shared body of the diarization bake-off jobs (docs/diarization_bakeoff_plan.md).
# Not submitted on its own: hpc/bakeoff/<arm>.slurm sets ARM and ARM_ENV and
# sources this with the audio paths as its arguments.
#
# For every recording, two runs of the arm, each in its own process (plan §2:
# the second run is the determinism check). Each run writes
#   transcripts/bakeoff/<stem>/<arm>/run<k>/   native output + provenance.json
# plus nvidia-smi's memory samples, so peak GPU memory is measured outside torch
# as well as inside it. A run whose provenance.json already exists is skipped,
# never overwritten: the outputs are frozen once written (plan §2).

if [ -z "${ARM:-}" ] || [ -z "${ARM_ENV:-}" ]; then
    echo "ERROR: ARM and ARM_ENV must be set; submit hpc/bakeoff/<arm>.slurm, not this file"
    exit 2
fi
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RUNNER="$REPO_ROOT/hpc/bakeoff/run_arm.py"
if [ ! -f "$RUNNER" ]; then
    echo "ERROR: no hpc/bakeoff/run_arm.py under REPO_ROOT=$REPO_ROOT; submit from the repo root"
    exit 2
fi
if [ $# -lt 1 ]; then
    echo "usage: sbatch hpc/bakeoff/$ARM.slurm <audio> [<audio> ...]"
    exit 2
fi

source "$HOME/miniforge3/etc/profile.d/mamba.sh"
mamba activate "$ARM_ENV" || { echo "ERROR: cannot activate $ARM_ENV; build it with its envs/ recipe"; exit 2; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "job $SLURM_JOB_ID on $SLURMD_NODENAME: arm $ARM, env $ARM_ENV, repo $(git -C "$REPO_ROOT" rev-parse --short HEAD)," \
     "$(python -V 2>&1), $# recording(s)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

status=0
for audio in "$@"; do
    stem=$(basename "${audio%.*}")
    for run in run1 run2; do
        out="$REPO_ROOT/transcripts/bakeoff/$stem/$ARM/$run"
        if [ -e "$out/provenance.json" ]; then
            echo "SKIP $out: already written, and frozen once written"
            continue
        fi
        mkdir -p "$out"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 \
            > "$out/nvidia-smi_memory_used_mib.txt" 2>/dev/null &
        smi=$!
        echo "START $(date +%T)  $ARM $run  $audio"
        python -u "$RUNNER" --arm "$ARM" --audio "$audio" --out "$out"
        rc=$?
        kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
        peak=$(sort -n "$out/nvidia-smi_memory_used_mib.txt" 2>/dev/null | tail -1)
        # Exit 2 from the runner: a pin did not match what loaded, or the model did not load.
        echo "EXIT=$rc $(date +%T)  $ARM $run  $audio  nvidia-smi peak ${peak:-?} MiB"
        [ "$rc" -eq 0 ] || status=$rc
    done
done
exit $status
