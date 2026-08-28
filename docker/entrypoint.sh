#!/usr/bin/env bash
set -euo pipefail

# Escape hatches for debugging the image itself.
case "${1:-}" in
    bash|sh|shell)
        shift
        exec /bin/bash "$@"
        ;;
    python)
        shift
        exec python "$@"
        ;;
esac

exec python -u /app/src/transcribe_audio.py "$@"
