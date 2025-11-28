#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80gb:2
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G

RUN_ID=generation_llama-3.1-70b_w8a8
VLLM_MODEL=neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8

source scripts/vllm_serve.sh

pdm run python src/marcel/retrievers.py \
    --data_path data/crawls/20250317/data.jsonl \
    --query_path data/queries/20250317-email.json \
    --faq_path data/queries/20250317-faq.json \
    --out_path output/20250317-email/$RUN_ID/ \
    --retrievers bm25 faq \
    --join_weights 1 1 \
    --faq_embedding_model all-MiniLM-L6-v2 \
    --faq_embedding_similarity_function cosine \
    --use_generator \
    --generation_model $VLLM_MODEL \
    --generation_temperature 0.7 \
    --no-skip-without-sources \
    --top_k 5
