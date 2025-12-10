# Marcel Experiments (EMNLP 2025, System Demonstration)

This repository provides the code to reproduce the experiments of following paper:

> Jan Trienes, Anastasiia Derzhanskaia, Roland Schwarzkopf, Markus Mühling, Jörg Schlötterer, and Christin Seifert. 2025. [Marcel: A Lightweight and Open-Source Conversational Agent for University Student Support](https://aclanthology.org/2025.emnlp-demos.13/). In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 181–195, Suzhou, China. Association for Computational Linguistics.

For the demo application itself, please refer to the [Marcel-Chat](https://github.com/aix-group/marcel-chat) repository.


## Environment

Clone this repository:

```sh
git clone https://github.com/aix-group/marcel-emnlp-experiments.git
```

Install dependencies with with [pdm](https://pdm-project.org/en/latest/).

```sh
pdm install
```

## Data

We provide knowledge base, data for FAQ retriever and sample queries in `data/`. The full evaluation data (user queries + model outputs) are shared on request. This is the checksums of data:

```sh
❯ md5sum data/**/*.json*
b92b502be1fe4bb75c82a00b4d1a431b  data/crawls/20250317/data.jsonl           # Knowledge base: scraped HTML converted to Markdown
3f5fd08d8ebb4cf47a4bab341437399f  data/queries/20250317-email.example.json  # Sample queries
cdb09ad24885f75b5e16acd5d8c47dcf  data/queries/20250317-email.json          # User queries for evaluation
7cb12ec7de0975417cf0cac453d50045  data/queries/20250317-faq.json            # Curated data for FAQ retriever
```

Please reach out using contact detail below if you would like to have access to the data.


## Compute Requirements

Every script is to be run from the repository root. We use a Slurm-based execution environment, but each script below can also be executed on a standard Linux host. Minimum requirement: 2 GPUs (e.g., A100 with 80GB VRAM) and sufficient disk space for storing model checkpoints (~ 400 GB).

## Retriever Experiments

Reproducing the main retriever results (Table 2).

```sh
# Retrievers without reranking.
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

# Retrieves with best reranking model
sbatch scripts/bm25_dense_rerank.sh
sbatch scripts/bm25_hyde_rerank.sh
sbatch scripts/bm25_faq_rerank.sh
```

Reproducing the reranker model sweep (Appendix, Table 4).

```
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_jina-tiny jinaai/jina-reranker-v1-tiny-en
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_jina-turbo jinaai/jina-reranker-v1-turbo-en
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-xsmall mixedbread-ai/mxbai-rerank-xsmall-v1
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-base mixedbread-ai/mxbai-rerank-base-v1
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_mxbai-large mixedbread-ai/mxbai-rerank-large-v1
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_minilm-l6 cross-encoder/ms-marco-MiniLM-L6-v2
sbatch scripts/bm25_dense_rerank_sweep.sh bm25_dense_rerank_minilm-12 cross-encoder/ms-marco-MiniLM-L12-v2
```

Evaluate system outputs.

```sh
# Run the command below and follow printed instruction
./scripts/evaluate_task_list.sh
```

Generate LaTeX tables: run [notebooks/retriever-evaluation.ipynb](./notebooks/retriever-evaluation.ipynb).

## Generator Experiments

Reproduce generator experiments (Table 3).

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
```

Evaluate system outputs.


```sh
# Run the command below and follow printed instruction
./scripts/evaluate_task_list.sh
```

Generate LaTeX tables: run [notebooks/generator-evaluation.ipynb](./notebooks/generator-evaluation.ipynb).

## Citation

If you found any of these resources useful, please consider citing the following paper.

```bibtex
@inproceedings{trienes-etal-2025-marcel,
    title = "Marcel: A Lightweight and Open-Source Conversational Agent for University Student Support",
    author = {Trienes, Jan  and
      Derzhanskaia, Anastasiia  and
      Schwarzkopf, Roland  and
      M{\"u}hling, Markus  and
      Schl{\"o}tterer, J{\"o}rg  and
      Seifert, Christin},
    editor = {Habernal, Ivan  and
      Schulam, Peter  and
      Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-demos.13/",
    doi = "10.18653/v1/2025.emnlp-demos.13",
    pages = "181--195",
    ISBN = "979-8-89176-334-0",
}
```

## Contact

Please reach out to <a href="mailto:jan.trienes@uni-marburg.de">Jan Trienes</a> if you have any comments, questions, or suggestions.
