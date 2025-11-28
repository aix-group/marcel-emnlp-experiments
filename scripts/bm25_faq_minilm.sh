#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G

RUN_ID=bm25_faq_minilm
pdm run -p backend/ \
    python backend/src/marcel/experiments/retrievers.py \
    --data_path data/crawls/20250317/data.jsonl \
    --query_path data/queries/20250317-email.json \
    --faq_path data/queries/20250317-faq.json \
    --out_path output/20250317-email/$RUN_ID/ \
    --retrievers bm25 faq \
    --join_weights 1 1 \
    --faq_embedding_model all-MiniLM-L6-v2 \
    --faq_embedding_similarity_function cosine
