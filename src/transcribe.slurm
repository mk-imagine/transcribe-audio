#!/bin/bash
#SBATCH --partition=gpucluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # 1 GPU is sufficient for ASR + BERT inference
#SBATCH --time=2:00:00
#SBATCH --output=./logs/transcribe_%j.log

echo "Running on node: $SLURMD_NODENAME"

# --- Environment Setup ---
# Adjust this path to match your Conda setup
if [ -f "${HOME}/miniforge3/etc/profile.d/mamba.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
fi

# Ensure you have the environment with transformers, torch, pyannote.audio installed
mamba activate audio-transcribe

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'audio-transcribe'"
    exit 1
fi

# --- Inputs ---
# $1: Input Wav Path
# $2: Output Directory
# $3: Cleaning Mode (none | basic | intelligent)
# $4: Additional Flags (e.g., --no_diarize)

AUDIO_IN="$1"
OUT_DIR="$2"
CLEAN_MODE="$3"
EXTRA_FLAGS="$4"

# Set HF Token if not globally set
# export HF_TOKEN="your_token_here"

python -u transcribe_audio.py \
    --input_path "$AUDIO_IN" \
    --output_dir "$OUT_DIR" \
    --clean_mode "$CLEAN_MODE" \
    --hf_token "$HF_TOKEN" \
    $EXTRA_FLAGS

exit $?