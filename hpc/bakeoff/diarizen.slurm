#!/bin/bash
# Diarization bake-off, arm 2: DiariZen, BUT-FIT/diarizen-wavlm-large-s80-md-v2
# at the pinned code commit (docs/diarization_bakeoff_plan.md §1). Its own env,
# bakeoff-diarizen: build it first with
#   bash envs/bakeoff-diarizen.sh > scratch/logs/env_bakeoff-diarizen.log 2>&1
#
#   sbatch hpc/bakeoff/diarizen.slurm data/PSY777-F26-WK2-Mon.wav data/PSY498-F26-WK3-Tue.wav
#
# Submit from the repo root: --output is relative to it (hpc/README.md).
#SBATCH --job-name=bo-diarizen
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=scratch/logs/%x_%j.log

ARM=diarizen
ARM_ENV=bakeoff-diarizen
# docs/environments.md: DiariZen's libicu wants CXXABI_1.3.15, which Rocky 8's
# system libstdc++ lacks, so the env's own lib directory goes first.
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/$ARM_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/hpc/bakeoff/arm_job.sh"
