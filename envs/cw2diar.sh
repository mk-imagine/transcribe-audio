#!/bin/bash
# Build the `cw2diar` environment on POLARIS: CrisperWhisper 2 (ct2 backend)
# and pyannote diarization in one env, so a single stage-1 run yields both
# verbatim words and speaker turns. This is what produced tests/fixtures/golden.json.
#
# Fresh build (D20): never --clone an env that pip will run in -- clones
# hardlink, so a pip upgrade in the clone can strip packages from the source.
# The two packages do not conflict: pyannote pins torch==2.8.0, and
# crisperwhisper[ct2] needs no torch at all. No transformers here, so a CT2
# conversion to a new compute type has to be done in cw2native first.
#
#   bash envs/cw2diar.sh > hpc/logs/env_cw2diar.log 2>&1
set -o pipefail
M=$HOME/miniforge3/bin/mamba
E=$HOME/miniforge3/envs/cw2diar
echo "START $(date)"
$M create -n cw2diar python=3.11 -y || { echo ENV_BUILD_FAILED; exit 1; }
$E/bin/pip install --no-input \
    "crisperwhisper[ct2]==2.0.2" \
    "pyannote.audio==4.0.3" \
    librosa python-dotenv \
  || { echo ENV_BUILD_FAILED; exit 1; }
echo "--- verify imports ---"
$E/bin/python - <<'PY'
import crisperwhisper, ctranslate2, torch, pyannote.audio, librosa
print("crisperwhisper", crisperwhisper.__version__)
print("ctranslate2", ctranslate2.__version__, "cuda devices:", ctranslate2.get_cuda_device_count())
print("torch", torch.__version__)
print("pyannote.audio", pyannote.audio.__version__)
PY
echo "END $(date)"; echo ENV_BUILD_OK
