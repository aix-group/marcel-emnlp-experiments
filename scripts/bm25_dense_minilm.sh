#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G

RUN_ID=bm25_dense_minilm
pdm run python src/marcel/retrievers.py \
    --data_path data/crawls/20250317/data.jsonl \
    --query_path data/queries/20250317-email.json \
    --out_path output/20250317-email/$RUN_ID/ \
    --retrievers bm25 dense \
    --join_weights 1 1 \
    --embedding_model all-MiniLM-L6-v2 \
    --embedding_similarity_function cosine
