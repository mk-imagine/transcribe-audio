# GPU Container (ubuntu-gpu)

Runs the transcription pipeline on the local GPU box (`ubuntu-gpu`, RTX 3070 8 GB,
driver 580.173.02, above CUDA 13.0's 580.65 floor) without installing anything on the host beyond Docker, which is
already present along with `nvidia-container-toolkit`.

## Image

| | |
|---|---|
| Base | `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` |
| Python | 3.12.3 |
| Torch | 2.11.0 (CUDA 13.0, `torchaudio` included) |
| System pkgs | `ffmpeg` (segment extraction), `libsndfile1` (soundfile backend) |
| Runs as | host uid/gid (1000:1000 by default) so outputs are not root-owned |
| Python env | `/opt/venv`, created with `--system-site-packages` |

The base image puts torch in the distro Python, which Ubuntu marks PEP 668
externally-managed. The image therefore layers a venv on top of the system
site-packages instead of overriding that marker: project dependencies get their
own prefix while torch/torchaudio remain visible from the base image.

The torch stack from the base image is frozen into a pip constraints file before
the other dependencies install, so nothing can silently replace the CUDA build
with a CPU wheel. The build also asserts `torch.version.cuda` afterwards.

## One-time setup

```bash
cd ~/Repos/transcribe-audio
git pull

# HF_TOKEN is required for pyannote diarization and .env is gitignored,
# so it does not arrive with the clone. Create it once:
echo 'HF_TOKEN=hf_xxxxxxxxxxxx' > .env
chmod 600 .env

mkdir -p data transcripts
docker compose build
```

## Running

```bash
# Full file
docker compose run --rm transcribe -i /data/lecture.m4a -o /transcripts

# A slice, with intelligent cleaning
docker compose run --rm transcribe \
    -i /data/lecture.m4a -o /transcripts \
    --start_time 0:1:45 --end_time 0:2:15 \
    --clean_mode intelligent

# Poke around inside the image
docker compose run --rm transcribe bash
```

Put audio in `./data` on the host; it appears at `/data` in the container.
Transcripts land in `./transcripts` on the host.

## Models

Model weights persist in the `model-cache` named volume (`/cache` inside the
container, `HF_HOME=/cache/huggingface`), so they download once.

- Default `--model unsloth/crisperwhisper` routes to the transformers pipeline
  backend, which is the correct one for CUDA.
- The MLX backend (`kyr0/crisperwhisper-unsloth-mlx`) is Apple Silicon only and
  is unavailable in this container. Asking for it raises a clear error rather
  than failing at import time.
- Granite Speech (`--model ibm-granite/granite-speech-...`) loads in bfloat16.
  Watch VRAM: the 8 GB card will not hold the 8B variant.

To clear the cache: `docker volume rm transcribe-audio_model-cache`.
