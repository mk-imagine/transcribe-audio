#!/bin/bash
# Build the `bakeoff-diarizen` environment on POLARIS: DiariZen at the commit the
# diarization bake-off pins (docs/diarization_bakeoff_plan.md §1, arm 2).
#
#   bash envs/bakeoff-diarizen.sh > scratch/logs/env_bakeoff-diarizen.log 2>&1
#
# The steps are the DiariZen README's own, in its order, with mamba in place of
# conda for the env itself (D20):
#   1. python 3.10;
#   2. torch 2.1.1 / torchvision 0.16.1 / torchaudio 2.1.1 from the cu121 index;
#   3. requirements.txt, then the package itself, editable;
#   4. the vendored pyannote-audio 3.1.1 fork, editable, with the repo's constraints.txt;
#   5. the dscore submodule (DER scoring only; the bake-off does not use it).
# Then the two pinned checkpoints are fetched here, on the login node, so the
# GPU job reads them from the cache.
#
# A fresh env, never a --clone (D20: clones hardlink, and pip in a clone can
# strip packages from its source). The older `diarizen` env from the
# 2026-08-31 benchmark is left alone: its code commit was not recorded, and the
# v2 card asks for current code.
set -o pipefail
M=$HOME/miniforge3/bin/mamba
NAME=bakeoff-diarizen
E=$HOME/miniforge3/envs/$NAME
SRC=${DIARIZEN_SRC:-$HOME/src/DiariZen}
COMMIT=844f5555b0a98acd0931511fc641a8c5b8ba92c7
fail() { echo "ENV_BUILD_FAILED: $*"; exit 1; }

echo "START $(date)"
[ -d "$E" ] && fail "$E already exists; remove it first (mamba env remove -n $NAME -y)"
$M create -n $NAME python=3.10 -y || fail "mamba create"

if [ ! -d "$SRC/.git" ]; then
    # HTTPS: git over SSH does not authenticate in batch or ssh -n sessions (environments.md).
    git clone https://github.com/BUTSpeechFIT/DiariZen.git "$SRC" || fail "clone"
fi
git -C "$SRC" fetch origin || fail "fetch"
git -C "$SRC" checkout --detach "$COMMIT" || fail "checkout $COMMIT"
git -C "$SRC" submodule update --init || fail "submodule"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || fail "DiariZen is not at $COMMIT"

$E/bin/pip install --no-input torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121 || fail "torch"
(cd "$SRC" && $E/bin/pip install --no-input -r requirements.txt && $E/bin/pip install --no-input -e .) \
    || fail "diarizen"
(cd "$SRC/pyannote-audio" && $E/bin/pip install --no-input -e ".[dev,testing]" -c ../constraints.txt) \
    || fail "vendored pyannote-audio"

# docs/environments.md: the env's libicu needs CXXABI_1.3.15, which Rocky 8's
# system libstdc++ lacks. The job script sets the same.
export LD_LIBRARY_PATH=$E/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "--- verify imports and pins ---"
$E/bin/python - <<'PY' || fail "verify"
import subprocess
from importlib.metadata import version
from pathlib import Path
import torch, torchaudio, numpy, pyannote.audio, diarizen
from diarizen.pipelines.inference import DiariZenPipeline  # noqa: F401
src = Path(diarizen.__file__).resolve().parent.parent
commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("diarizen", src, commit)
print("torch", torch.__version__, "torchaudio", torchaudio.__version__, "numpy", numpy.__version__)
print("pyannote.audio (vendored)", version("pyannote.audio"))
assert commit == "844f5555b0a98acd0931511fc641a8c5b8ba92c7", commit
assert torch.__version__.startswith("2.1.1"), torch.__version__
assert version("pyannote.audio") == "3.1.1", version("pyannote.audio")
PY

echo "--- fetch the pinned checkpoints ---"
$E/bin/python - <<'PY' || fail "checkpoints"
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("BUT-FIT/diarizen-wavlm-large-s80-md-v2",
                        revision="027b3e221b9d81dce2be794084f1dbd3ba2da403"))
print(hf_hub_download("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin",
                      revision="837717ddb9ff5507820346191109dc79c958d614"))
PY
echo "END $(date)"; echo ENV_BUILD_OK
