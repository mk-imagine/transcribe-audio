#!/bin/bash
# Stage 1 over several recordings at once, in one allocation. File i runs as its
# own process on GPU (i / PER_GPU) of the allocation, with its own log.
#
#   sbatch --gres=gpu:<G> --cpus-per-task=<8 x files> hpc/transcribe_parallel.slurm \
#       <PER_GPU> <output_dir> <audio> [<audio> ...]
#
# Needs #files <= G x PER_GPU. Submit from the repo root, as with
# transcribe.slurm: --output below and every path argument are relative to it.
# TRANSCRIBE_ENV (default cw2diar: CrisperWhisper 2 + pyannote) and
# TRANSCRIBE_FLAGS (default --dual_stream) override the environment and the
# pipeline flags, e.g. TRANSCRIBE_FLAGS="--dual_stream --no_diarize" sbatch ...
#
# Why one job with many GPUs rather than one job per file: POLARIS allows 2
# running and 5 submitted jobs per user but sets no per-job GPU or CPU limit
# (sacctmgr, 2026-09-22), so a sweep is capped by the job count, not by GPUs.
# Two of these jobs cover both GPU nodes (8 A100s).
#
# PER_GPU, measured 2026-09-22 on 80-145 min lectures, dual stream + diarization:
#   1 -- one file per GPU; a ~90 min lecture takes ~5 min, 145 min ~8 min.
#   2 -- each file runs ~1.3-1.6x slower, but a pair finishes ~1.3-1.5x sooner
#        than the same two run back to back. Use it only when files outnumber GPUs.
# Model memory is ~7 GB of the 80 GB per process, so memory is not the limit.
#SBATCH --job-name=transcribe-par
#SBATCH --partition=gpucluster
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --output=scratch/logs/%x_%j.log

if [ $# -lt 3 ]; then
    echo "usage: sbatch --gres=gpu:<G> ... $0 <PER_GPU> <output_dir> <audio> [<audio> ...]"
    exit 2
fi
PER_GPU=$1; OUT=$2; shift 2

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PIPELINE="${REPO_ROOT}/src/transcribe_audio.py"
if [ ! -f "$PIPELINE" ]; then
    echo "ERROR: no src/transcribe_audio.py under REPO_ROOT=${REPO_ROOT}; submit from the repo root."
    exit 2
fi

source "$HOME/miniforge3/etc/profile.d/mamba.sh"
CONDA_ENV="${TRANSCRIBE_ENV:-cw2diar}"
mamba activate "$CONDA_ENV"
FLAGS="${TRANSCRIBE_FLAGS:---dual_stream}"
THREADS=8

IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
if [ $# -gt $(( ${#GPUS[@]} * PER_GPU )) ]; then
    echo "ERROR: $# files for ${#GPUS[@]} GPU(s) x $PER_GPU per GPU"; exit 2
fi
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $SLURMD_NODENAME: GPUs ${GPUS[*]}, $PER_GPU per GPU, $SLURM_CPUS_PER_TASK CPUs," \
     "env $CONDA_ENV, flags '$FLAGS', $(git -C "$REPO_ROOT" rev-parse --short HEAD), $# file(s) -> $OUT"

run_one() {
    local slot=$1 audio=$2
    local gpu=${GPUS[$(( slot / PER_GPU ))]}
    local name; name=$(basename "${audio%.*}")
    local log="scratch/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${name}.log"
    echo "START $(date +%T)  gpu $gpu  $audio  (log $log)"
    # $FLAGS is unquoted on purpose: it carries several flags.
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS TORCH_NUM_THREADS=$THREADS \
        python -u "$PIPELINE" -i "$audio" -o "$OUT" $FLAGS --job_id "$SLURM_JOB_ID" > "$log" 2>&1
    local rc=$?
    # Exit 1 means audio ranges were lost and the transcript has holes; 2 means
    # the model id or the requested mode was refused.
    echo "EXIT=$rc $(date +%T)  gpu $gpu  $audio"
    # The pipeline inserts _seg_<start> into the name of an excerpt run.
    local record="$OUT/${name}_job${SLURM_JOB_ID}_raw.json" seg
    for seg in "$OUT/${name}"_seg_*"_job${SLURM_JOB_ID}_raw.json"; do
        [ -f "$seg" ] && record=$seg
    done
    python - "$REPO_ROOT/src" "$record" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from pipeline import schema
p = Path(sys.argv[2])
if not p.exists():
    print(f"RECORD {p.name}  MISSING"); sys.exit()
doc = schema.read(p)
problems = schema.validate(doc)
print(f"RECORD {p.name}  validate={'OK' if not problems else problems}  {p.stat().st_size // 1024} KB\n"
      f"RECORD   {schema.summary(doc)}")
EOF
}

slot=0
for audio in "$@"; do
    run_one $slot "$audio" &
    slot=$((slot + 1))
    sleep 5   # stagger the model loads
done
wait
echo "DONE $(date +%T)"
