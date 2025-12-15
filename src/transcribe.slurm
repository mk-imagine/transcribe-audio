#!/bin/bash
#SBATCH --partition=gpucluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=./logs/transcribe_%j.log

echo "Running on node: $SLURMD_NODENAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
if [ -f "${HOME}/miniforge3/etc/profile.d/mamba.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
fi

mamba activate audio-transcribe

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'audio-transcribe'"
    exit 1
fi

# Set standard threads to avoid conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Execution ---
# $1: Input File
# $2: Output Dir
# $3: Clean Mode
# $4: HF Token
# $5: Extra Flags (timestamps, etc)

python -u transcribe_audio.py \
    --input_path "$1" \
    --output_dir "$2" \
    --clean_mode "$3" \
    --hf_token "$4" \
    $5

exit $?