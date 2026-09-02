#!/bin/bash

# Defaults
COMPUTE_MODE="gpu"
CLEAN_MODE="none"
START_TIME=""
END_TIME=""
EXTRA_FLAGS=""
OUTPUT_DIR="./transcripts"
CONDA_ENV="cw2native"
MODEL=""

# Help Function
usage() {
    echo "Usage: $0 -i <input_file> -o <output_dir> [-m <mode>] [-c <compute>] [-s <start_time>] [-e <end_time>] [-E <conda_env>] [-M <model>]"
    echo "  -i : Input audio file (Required)"
    echo "  -o : Output directory"
    echo "  -m : Clean mode (raw, basic, intelligent). Default: raw/none"
    echo "  -c : Compute mode (gpu, cpu). Default: gpu"
    echo "  -s : Start time (e.g., 00:05:00)"
    echo "  -e : End time (e.g., 00:10:00)"
    echo "  -E : Mamba env to activate. Default: cw2native (CrisperWhisper 2)"
    echo "       Use audio-transcribe for diarization (it has pyannote)."
    echo "  -M : ASR model. Must be a registered id; run"
    echo "       'python src/transcribe_audio.py --help' for the list."
    echo "       Default: nyralabs/CrisperWhisper2.0_large"
    exit 1
}

# Parse Flags
while getopts "i:o:m:c:s:e:E:M:h" opt; do
    case ${opt} in
        i) AUDIO_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        m) CLEAN_MODE="$OPTARG" ;;
        c) COMPUTE_MODE="$OPTARG" ;;
        s) START_TIME="$OPTARG" ;;
        e) END_TIME="$OPTARG" ;;
        E) CONDA_ENV="$OPTARG" ;;
        M) MODEL="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Validation
if [ -z "$AUDIO_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Input file (-i) and Output directory (-o) are required."
    usage
fi

# Absolute paths to this script's directory (src/) and the repo root, so the
# wrapper works from any working directory. Every path below is derived from
# these -- previously the sbatch file, the log directory and the pipeline were
# all named relatively, so the wrapper only worked when run from inside src/.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
LOG_DIR="$REPO_ROOT/hpc/logs"

# Check for .env in the repo root
if [ -z "$HF_TOKEN" ] && [ ! -f "$REPO_ROOT/.env" ]; then
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

# Appended to the passthrough flags rather than added as a positional, so the
# transcribe.slurm argument contract stays unchanged.
if [ -n "$MODEL" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --model $MODEL"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Submit
echo "Mamba environment: $CONDA_ENV"
[ -n "$MODEL" ] && echo "Model: $MODEL"

echo "Logs: $LOG_DIR"

sbatch \
    --export=ALL,TRANSCRIBE_ENV="$CONDA_ENV",REPO_ROOT="$REPO_ROOT" \
    --output="$LOG_DIR/transcribe_%j.log" \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --time=$TIME_LIMIT \
    $GRES_FLAG \
    "$SCRIPT_DIR/transcribe.slurm" "$AUDIO_FILE" "$OUTPUT_DIR" "$CLEAN_MODE" "" "$EXTRA_FLAGS"