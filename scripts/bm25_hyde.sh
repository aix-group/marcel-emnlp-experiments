#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G

RUN_ID=bm25_hyde_gemma-27b
VLLM_MODEL=google/gemma-3-1b-it
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=8192

source experiments/vllm_serve.sh

pdm run python src/marcel/retrievers.py \
    --data_path data/crawls/20250317/data.jsonl \
    --query_path data/queries/20250317-email.json \
    --out_path output/20250317-email/$RUN_ID/ \
    --retrievers bm25 hyde \
    --join_weights 1 1 \
    --hyde_generator_model $VLLM_MODEL \
    --embedding_model sentence-transformers/msmarco-bert-base-dot-v5 \
    --embedding_similarity_function dot
