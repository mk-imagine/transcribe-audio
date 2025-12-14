#!/bin/bash

# Usage: ./run_transcription.sh <audio_file> <output_dir> [mode]
# mode options: raw (default), basic, intelligent

AUDIO_FILE=$1
OUTPUT_DIR=$2
MODE_ARG=$3

if [ -z "$AUDIO_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <path_to_wav> <output_dir> [raw|basic|intelligent]"
    echo "  raw        : No cleanup (verbatim)"
    echo "  basic      : Regex-based cleanup (fast)"
    echo "  intelligent: AI/BERT-based cleanup (best quality)"
    exit 1
fi

# Default to raw if not specified
CLEAN_MODE="none"

if [ "$MODE_ARG" == "intelligent" ]; then
    CLEAN_MODE="intelligent"
    echo "Submitting job with INTELLIGENT AI cleanup..."
elif [ "$MODE_ARG" == "basic" ]; then
    CLEAN_MODE="basic"
    echo "Submitting job with BASIC Regex cleanup..."
else
    echo "Submitting job for RAW transcription..."
fi

mkdir -p logs

# Submit to Slurm
sbatch transcribe.slurm "$AUDIO_FILE" "$OUTPUT_DIR" "$CLEAN_MODE"