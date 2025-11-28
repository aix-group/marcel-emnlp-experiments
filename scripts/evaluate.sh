#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_80gb:2
#SBATCH --nodes=1
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    arguments="$@"
else
    # If it's an array job get the task with index SLURM_ARRAY_TASK_ID
    run_path=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" evaluate_tasks.txt)
    arguments="--run_path $run_path $@"
fi

VLLM_MODEL=neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8
source scripts/vllm_serve.sh

pdm run python -m marcel_evaluation.runner $arguments
