# Experiments

Every script is to be run from the repository root.

## Retrievers

```sh
sbatch scripts/bm25.sh
sbatch scripts/bm25_dense_minilm_gpu.sh
sbatch scripts/bm25_dense_minilm.sh
sbatch scripts/bm25_dense_msmarco_gpu.sh
sbatch scripts/bm25_dense_msmarco.sh
sbatch scripts/bm25_faq_minilm_gpu.sh
sbatch scripts/bm25_faq_minilm.sh
sbatch scripts/bm25_faq_msmarco_gpu.sh
sbatch scripts/bm25_faq_msmarco.sh
sbatch scripts/bm25_hyde.sh


# BM25 + Dense + Rerank (Model sweep)
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_jina-tiny jinaai/jina-reranker-v1-tiny-en
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_jina-turbo jinaai/jina-reranker-v1-turbo-en
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-xsmall mixedbread-ai/mxbai-rerank-xsmall-v1
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-base mixedbread-ai/mxbai-rerank-base-v1
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-large mixedbread-ai/mxbai-rerank-large-v1
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_minilm-l6 cross-encoder/ms-marco-MiniLM-L6-v2
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_minilm-12 cross-encoder/ms-marco-MiniLM-L12-v2

# Run with best model
sbatch scripts/bm25_dense_rerank.sh
sbatch scripts/bm25_hyde_rerank.sh
sbatch scripts/bm25_faq_rerank.sh

# Evaluation
./scripts/evaluate_task_list.sh
sbatch --array=0-1%20 scripts/evaluate.sh --metrics ReferenceAnswerLength,ContextLength,MeanReciprocalRank,PrecisionAtCutoff,RecallAtCutoff
```

## Generator experiments

```sh
VLLM_PORT=8000 sbatch scripts/generation_llama-3.1-8b.sh
VLLM_PORT=8010 sbatch scripts/generation_llama-3.1-70b.sh
VLLM_PORT=8020 sbatch scripts/generation_llama-3.1-8b_w8a8.sh
VLLM_PORT=8030 sbatch scripts/generation_llama-3.1-70b_w8a8.sh
VLLM_PORT=8040 sbatch scripts/generation_gemma-1b.sh
VLLM_PORT=8050 sbatch scripts/generation_gemma-4b.sh
VLLM_PORT=8060 sbatch scripts/generation_gemma-12b.sh
VLLM_PORT=8070 sbatch scripts/generation_gemma-27b.sh
VLLM_PORT=8080 sbatch scripts/generation_gemma-27b_oracle.sh

# Evaluation: run the command below and follow printed instruction
./scripts/evaluate_task_list.sh # TODO evaluate this: 8585778
```
