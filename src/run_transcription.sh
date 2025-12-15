#!/bin/bash

# Defaults
COMPUTE_MODE="gpu"
CLEAN_MODE="none"
START_TIME=""
END_TIME=""
EXTRA_FLAGS=""
OUTPUT_DIR="./"

# Help Function
usage() {
    echo "Usage: $0 -i <input_file> -o <output_dir> [-m <mode>] [-c <compute>] [-s <start_time>] [-e <end_time>]"
    echo "  -i : Input audio file (Required)"
    echo "  -o : Output directory"
    echo "  -m : Clean mode (raw, basic, intelligent). Default: raw/none"
    echo "  -c : Compute mode (gpu, cpu). Default: gpu"
    echo "  -s : Start time (e.g., 00:05:00)"
    echo "  -e : End time (e.g., 00:10:00)"
    exit 1
}

# Parse Flags
while getopts "i:o:m:c:s:e:h" opt; do
    case ${opt} in
        i) AUDIO_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        m) CLEAN_MODE="$OPTARG" ;;
        c) COMPUTE_MODE="$OPTARG" ;;
        s) START_TIME="$OPTARG" ;;
        e) END_TIME="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Validation
if [ -z "$AUDIO_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Input file (-i) and Output directory (-o) are required."
    usage
fi

# Get the absolute path of the directory containing this script (src/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# heck for .env in the parent of the script directory (Project Root)
if [ -z "$HF_TOKEN" ] && [ ! -f "$SCRIPT_DIR/../.env" ]; then
    echo "WARNING: HF_TOKEN variable not set and no .env file found in project root."
    echo "         Diarization will likely fail."
fi

# Setup Compute Resources
if [ "$COMPUTE_MODE" == "cpu" ]; then
    echo "Submitting to CPU CLUSTER (128 Cores, 18hr limit)..."
    PARTITION="cpucluster"
    CPUS="128"
    TIME_LIMIT="18:00:00"
    GRES_FLAG=""
else
    echo "Submitting to GPU CLUSTER (8 Cores, 8hr limit)..."
    PARTITION="gpucluster"
    CPUS="8"
    TIME_LIMIT="08:00:00"
    GRES_FLAG="--gres=gpu:1"
fi

# Build Time Flags
if [ ! -z "$START_TIME" ]; then
    EXTRA_FLAGS="--start_time $START_TIME"
    if [ ! -z "$END_TIME" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --end_time $END_TIME"
    fi
fi

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Submit
sbatch \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --time=$TIME_LIMIT \
    $GRES_FLAG \
    transcribe.slurm "$AUDIO_FILE" "$OUTPUT_DIR" "$CLEAN_MODE" "" "$EXTRA_FLAGS"