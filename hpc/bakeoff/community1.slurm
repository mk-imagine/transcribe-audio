#!/bin/bash
# Diarization bake-off, arm 1: pyannote community-1, the current diarizer
# (docs/diarization_bakeoff_plan.md §1). Runs through stage 1's own loader in
# the production env, cw2diar (envs/cw2diar.sh), so it measures what
# production runs. It also writes community-1's exclusive diarization from the
# same call, as the optional arm the plan lists for Mark; that is scored only
# if he adds it.
#
#   sbatch hpc/bakeoff/community1.slurm data/PSY777-F26-WK2-Mon.wav data/PSY498-F26-WK3-Tue.wav
#
# Submit from the repo root: --output is relative to it (hpc/README.md). Needs
# HF_TOKEN (the repo's .env), as stage 1 does: community-1 is gated.
#SBATCH --job-name=bo-community1
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=scratch/logs/%x_%j.log

ARM=community1
ARM_ENV=cw2diar
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/hpc/bakeoff/arm_job.sh"
