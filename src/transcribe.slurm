#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./logs/transcribe_%j.log

set -e

echo "Running on node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"

# --- Environment Setup ---
# Adjust this path if your mamba/conda location is different on the cluster
if [ -f "${HOME}/miniforge3/etc/profile.d/mamba.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
fi

# Which environment to activate. cw2native carries the `crisperwhisper`
# package, which is the only supported way to run CrisperWhisper (D17) and the
# default model. Diarization lives in audio-transcribe, which has pyannote:
#   TRANSCRIBE_ENV=audio-transcribe sbatch transcribe.slurm ...
# or via run_transcription.sh -E <env>.
CONDA_ENV="${TRANSCRIBE_ENV:-cw2native}"

echo "Activating mamba environment: ${CONDA_ENV}"
mamba activate "${CONDA_ENV}"

# --- Threading Configuration ---
# Ensure the script uses the CPUs allocated by Slurm, not all 128/64
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execution ---
# The arguments ($1, $2, etc.) are passed from the sbatch command in run_transcription.sh
# $1: Audio File Path
# $2: Output Directory
# $3: Clean Mode (intelligent/basic/none)
# $4: HF Token (Empty if using .env)
# $5: Extra Flags (start/end times)

echo "Starting job $SLURM_JOB_ID at $(date)"

# `set -e` would abort before the status could be logged, so the exit code is
# checked explicitly. Exit 1 means audio ranges were lost and the transcript
# has holes; exit 2 means the model id or the requested mode was refused.
set +e
python -u transcribe_audio.py \
    --input_path "$1" \
    --output_dir "$2" \
    --clean_mode "$3" \
    --hf_token "$4" \
    --job_id $SLURM_JOB_ID \
    $5
exit_code=$?
set -e

if [ $exit_code -ne 0 ]; then
      echo "ERROR: transcribe_audio.py failed with exit code $exit_code"
      exit $exit_code
fi

echo "Job $SLURM_JOB_ID completed at $(date)"
exit 0