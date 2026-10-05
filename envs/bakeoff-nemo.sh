#!/bin/bash
# Build the `bakeoff-nemo` environment on POLARIS: NVIDIA NeMo at the commit the
# diarization bake-off pins, for nvidia/Nemotron-3-Diarization (Sortformer)
# (docs/diarization_bakeoff_plan.md §1, arm 3).
#
#   bash envs/bakeoff-nemo.sh > scratch/logs/env_bakeoff-nemo.log 2>&1
#
# The model card's steps, adapted only where POLARIS forces it:
#   - "Python 3.12 or later": python 3.12, through mamba (D20);
#   - `apt-get install libsndfile1 ffmpeg`: no root here, so conda-forge's
#     libsndfile and ffmpeg go into the env instead;
#   - `uv pip install Cython packaging` and `uv pip install 'nemo-toolkit[asr]'`:
#     pip, inside the env (uv is not on POLARIS, and installing it is a further
#     install). And not the PyPI release: nemo-toolkit 3.0.0 (2026-08-07)
#     predates this model's inference support, which landed on main on
#     2026-08-15 (NVIDIA-NeMo/Speech #16032; read in the source, plan §1). The
#     pin is main on the model's release day, 2026-09-23.
# torch is pinned to the version cw2native already runs on POLARIS's A100s
# (environments.md), so the CUDA build is one this driver is known to load;
# NeMo asks for torch>=2.7.
#
# A fresh env, never a --clone (D20). The checkpoint is fetched here, on the
# login node, so the GPU job reads it from the cache.
set -o pipefail
M=$HOME/miniforge3/bin/mamba
NAME=bakeoff-nemo
E=$HOME/miniforge3/envs/$NAME
COMMIT=cf724ac337d1ebc7d0dda1e23fb80916f52927a5
fail() { echo "ENV_BUILD_FAILED: $*"; exit 1; }

echo "START $(date)"
[ -d "$E" ] && fail "$E already exists; remove it first (mamba env remove -n $NAME -y)"
$M create -n $NAME python=3.12 -y || fail "mamba create"
$M install -n $NAME -c conda-forge libsndfile ffmpeg -y || fail "libsndfile/ffmpeg"
$E/bin/pip install --no-input Cython packaging || fail "Cython"
$E/bin/pip install --no-input torch==2.13.0 || fail "torch"
$E/bin/pip install --no-input \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@$COMMIT" || fail "nemo_toolkit"

echo "--- verify imports and pins ---"
$E/bin/python - <<'PY' || fail "verify"
import json
from importlib.metadata import distribution
import torch, nemo
from nemo.collections.asr.models import SortformerEncLabelModel
url = json.loads(distribution("nemo_toolkit").read_text("direct_url.json") or "{}")
commit = (url.get("vcs_info") or {}).get("commit_id")
print("nemo", nemo.__version__, "from", url.get("url"), commit)
print("torch", torch.__version__)
assert commit == "cf724ac337d1ebc7d0dda1e23fb80916f52927a5", commit
# The model-level check and the 10 ms upsampling the card relies on; PyPI 3.0.0 has neither.
assert hasattr(SortformerEncLabelModel, "_check_streaming_parameters"), "NeMo lacks Nemotron-3 support"
PY

echo "--- fetch the pinned checkpoint ---"
$E/bin/python - <<'PY' || fail "checkpoint"
from huggingface_hub import hf_hub_download
print(hf_hub_download("nvidia/Nemotron-3-Diarization", "Nemotron-3-Diarization.nemo",
                      revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e"))
PY
echo "END $(date)"; echo ENV_BUILD_OK
