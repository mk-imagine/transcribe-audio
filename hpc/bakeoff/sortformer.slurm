#!/bin/bash
# Diarization bake-off, arm 3: NVIDIA Sortformer, as its current release
# nvidia/Nemotron-3-Diarization, through NeMo at the pinned commit, in the
# card's offline-style streaming configuration (docs/diarization_bakeoff_plan.md
# §1). Its own env, bakeoff-nemo: build it first with
#   bash envs/bakeoff-nemo.sh > scratch/logs/env_bakeoff-nemo.log 2>&1
#
#   sbatch hpc/bakeoff/sortformer.slurm data/PSY777-F26-WK2-Mon.wav data/PSY498-F26-WK3-Tue.wav
#
# Submit from the repo root: --output is relative to it (hpc/README.md).
#SBATCH --job-name=bo-sortformer
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=scratch/logs/%x_%j.log

ARM=sortformer
ARM_ENV=bakeoff-nemo
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/hpc/bakeoff/arm_job.sh"
