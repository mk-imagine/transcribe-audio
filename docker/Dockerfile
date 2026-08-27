# syntax=docker/dockerfile:1

# CUDA 13 base, chosen to match torchcodec's PyPI wheels (see requirements.txt).
# ubuntu-gpu runs driver 580.173.02, above CUDA 13.0's 580.65 floor, and the
# RTX 3070 is sm_86 (Ampere) which CUDA 13 still supports -- it dropped
# Maxwell/Pascal/Volta only.
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    NUMBA_CACHE_DIR=/cache/numba \
    MPLCONFIGDIR=/cache/matplotlib

# ffmpeg       - AudioHandler.prepare_segment() shells out to it for --start_time/--end_time
# libsndfile1  - soundfile backend
# python3-venv - the base image's Python is the distro one; see the venv note below
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The base image installs torch into the *distro* Python, which Ubuntu marks
# PEP 668 externally-managed, so pip refuses to install into it. Rather than
# override that marker, layer a venv on top of the system site-packages: our
# dependencies get their own prefix while torch/torchaudio stay visible from
# /usr/local/lib/python3.12/dist-packages.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python3 -m venv --system-site-packages "${VIRTUAL_ENV}"

# Freeze the torch stack that ships in the base image before installing anything
# else. Without this, resolving pyannote/transformers can silently pull a
# different torch wheel off PyPI and quietly cost us the GPU. With it, a
# dependency that genuinely needs another torch fails the build loudly.
RUN python - > /tmp/torch-constraints.txt <<'PY'
import importlib.metadata as md
for pkg in ("torch", "torchaudio", "torchvision"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
PY

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -c /tmp/torch-constraints.txt -r /tmp/requirements.txt

# Fail the build, not a two-hour transcription, if the native stack is wrong.
# torchcodec declares no torch dependency, so a CUDA-major mismatch surfaces
# only when its .so is actually dlopen'd.
RUN python -c "import torch, torchcodec, pyannote.audio, transformers, librosa, soundfile, peft; \
assert torch.version.cuda, 'CUDA torch was replaced by a CPU build'; \
print('torch', torch.__version__, '| torchcodec', torchcodec.__version__, '| pyannote.audio', pyannote.audio.__version__)"

# Run as the host user so bind-mounted transcripts/ are not written as root.
# Ubuntu 24.04 base images already ship a uid 1000, so both lookups are guarded.
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" app; fi; \
    if ! getent passwd "${UID}" >/dev/null; then useradd -m -u "${UID}" -g "${GID}" -s /bin/bash app; fi; \
    mkdir -p /cache /data /transcripts /app; \
    chown -R "${UID}:${GID}" /cache /app

WORKDIR /app
COPY --chown=${UID}:${GID} src/ /app/src/
COPY --chown=${UID}:${GID} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${UID}:${GID}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
