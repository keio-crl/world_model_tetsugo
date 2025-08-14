#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --partition=highend
#SBATCH --gres=gpu:rtx3080:1              # GPU 1枚を要求
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=./slurm_logs/%j.out

source .venv/bin/activate
uv run -m scripts.train
deactivate