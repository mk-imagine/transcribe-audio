#!/bin/bash

# Usage: ./run_transcription.sh <audio_file> <output_dir> [mode] [start_time] [end_time]
# Examples:
#   ./run_transcription.sh meeting.wav ./results intelligent
#   ./run_transcription.sh meeting.wav ./results intelligent 00:10:00 00:15:00

AUDIO_FILE=$1
OUTPUT_DIR=$2
MODE_ARG=$3
START_TIME=$4
END_TIME=$5

if [ -z "$AUDIO_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <path_to_wav> <output_dir> [raw|basic|intelligent] [start_time] [end_time]"
    exit 1
fi

# Determine Mode
CLEAN_MODE="none"
if [ "$MODE_ARG" == "intelligent" ]; then
    CLEAN_MODE="intelligent"
elif [ "$MODE_ARG" == "basic" ]; then
    CLEAN_MODE="basic"
fi

# Construct Time Flags
TIME_FLAGS=""
if [ ! -z "$START_TIME" ]; then
    TIME_FLAGS="--start_time $START_TIME"
    if [ ! -z "$END_TIME" ]; then
        TIME_FLAGS="$TIME_FLAGS --end_time $END_TIME"
    fi
    echo "Segment processing requested: $START_TIME to ${END_TIME:-EOF}"
fi

mkdir -p logs

# Submit
# We pass TIME_FLAGS as part of the 'Extra Flags' argument to the SLURM script
sbatch transcribe.slurm "$AUDIO_FILE" "$OUTPUT_DIR" "$CLEAN_MODE" "$TIME_FLAGS"