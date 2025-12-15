#!/bin/bash

# Usage: ./run_transcription.sh <audio_file> <output_dir> [mode] [start_time] [end_time]

AUDIO_FILE=$1
OUTPUT_DIR=$2
MODE_ARG=$3
START_TIME=$4
END_TIME=$5

# --- SET YOUR TOKEN HERE OR EXPORT IT ---
# HF_TOKEN="hf_..." 

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Export it or edit this script."
    exit 1
fi

if [ -z "$AUDIO_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <path_to_wav> <output_dir> [raw|basic|intelligent] [start_time] [end_time]"
    exit 1
fi

# Default to raw
CLEAN_MODE="none"
if [ "$MODE_ARG" == "intelligent" ]; then CLEAN_MODE="intelligent"; fi
if [ "$MODE_ARG" == "basic" ]; then CLEAN_MODE="basic"; fi

EXTRA_FLAGS=""
if [ ! -z "$START_TIME" ]; then
    EXTRA_FLAGS="--start_time $START_TIME"
    if [ ! -z "$END_TIME" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --end_time $END_TIME"
    fi
fi

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

sbatch transcribe.slurm "$AUDIO_FILE" "$OUTPUT_DIR" "$CLEAN_MODE" "$HF_TOKEN" "$EXTRA_FLAGS"