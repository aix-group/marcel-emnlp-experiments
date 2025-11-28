# Experiments

Every script is to be run from the repository root.

## Retrievers

```sh
sbatch experiments/bm25.sh  # done
sbatch experiments/bm25_dense_minilm_gpu.sh  # done
sbatch experiments/bm25_dense_minilm.sh  # done
sbatch experiments/bm25_dense_msmarco_gpu.sh  # done
sbatch experiments/bm25_dense_msmarco.sh  # done
sbatch experiments/bm25_faq_minilm_gpu.sh  # done
sbatch experiments/bm25_faq_minilm.sh  # done
sbatch experiments/bm25_faq_msmarco_gpu.sh  # done
sbatch experiments/bm25_faq_msmarco.sh  # done
sbatch experiments/bm25_hyde.sh  # done (1b), done (4b), done (27b)


# BM25 + Dense + Rerank (Model sweep)
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_jina-tiny jinaai/jina-reranker-v1-tiny-en
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_jina-turbo jinaai/jina-reranker-v1-turbo-en
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-xsmall mixedbread-ai/mxbai-rerank-xsmall-v1
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-base mixedbread-ai/mxbai-rerank-base-v1
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-large mixedbread-ai/mxbai-rerank-large-v1
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_minilm-l6 cross-encoder/ms-marco-MiniLM-L6-v2
sbatch experiments/bm25_dense_rerank_sweep.sh bm25_dense_rerank_minilm-12 cross-encoder/ms-marco-MiniLM-L12-v2

# Run with best model
sbatch experiments/bm25_dense_rerank.sh  # done
sbatch experiments/bm25_hyde_rerank.sh  # done
sbatch experiments/bm25_faq_rerank.sh  # done

# Evaluation
cd evaluation
./scripts/evaluate_task_list.sh
sbatch --array=0-1%20 scripts/evaluate.sh --metrics ReferenceAnswerLength,ContextLength,MeanReciprocalRank,PrecisionAtCutoff,RecallAtCutoff
```

## Generator experiments

```sh
VLLM_PORT=8000 sbatch experiments/generation_llama-3.1-8b.sh  # done
VLLM_PORT=8010 sbatch experiments/generation_llama-3.1-70b.sh  #  done
VLLM_PORT=8020 sbatch experiments/generation_llama-3.1-8b_w8a8.sh  # done
VLLM_PORT=8030 sbatch experiments/generation_llama-3.1-70b_w8a8.sh  # done
VLLM_PORT=8040 sbatch experiments/generation_gemma-1b.sh  #  done
VLLM_PORT=8050 sbatch experiments/generation_gemma-4b.sh  # done
VLLM_PORT=8060 sbatch experiments/generation_gemma-12b.sh  # done
VLLM_PORT=8070 sbatch experiments/generation_gemma-27b.sh  # done
VLLM_PORT=8080 sbatch experiments/generation_gemma-27b_oracle.sh  # done

# Evaluation
cd evaluation
./scripts/evaluate_task_list.sh
sbatch --array=0-8%20 scripts/evaluate.sh --model neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8 --tensor_parallel_size 2 --metrics GeneratedAnswerLength,ReferenceAnswerLength,ContextLength,BLEU,ROUGE,BERTScore,AnswerSimilarity,AnswerRelevance,NonAnswerCritic,AnswerFaithfulness  # 7615429_0
```
