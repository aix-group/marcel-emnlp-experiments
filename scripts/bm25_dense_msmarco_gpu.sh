#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G

RUN_ID=bm25_dense_msmarco_gpu
pdm run -p backend/ \
    python backend/src/marcel/experiments/retrievers.py \
    --data_path data/crawls/20250317/data.jsonl \
    --query_path data/queries/20250317-email.json \
    --out_path output/20250317-email/$RUN_ID/ \
    --retrievers bm25 dense \
    --join_weights 1 1 \
    --embedding_model sentence-transformers/msmarco-bert-base-dot-v5 \
    --embedding_similarity_function dot_product
