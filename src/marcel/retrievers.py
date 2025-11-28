import argparse
import logging
import os
from pathlib import Path
from typing import List

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

from marcel import data_loader
from marcel.components import (
    ContentLinkNormalizer,
    OpenAIChatGeneratorMultipleSamples,
)
from marcel.experiment_runner import run_experiment
from marcel.faq_retriever import FAQRetriever
from marcel.hyde import HyDE
from marcel.oracle_retriever import BM25RetrieverWithOracle

system_prompt_rag = """
You are a helpful and engaging chatbot called Marcel. If someone asks you, your name is Marcel and you are employed at the Marburg University. You answer questions of students around their studies. Please answer the questions based on the provided documents only. Ignore your own knowledge. Don't say that you are looking at a set of documents. If you cannot find the answer to a given question in the documents you must apologize and say that you don't have any information about the topic (e.g., "Unfortunately, I do not have any knowledge about <rephrase the question>").
""".strip()

user_prompt_template_rag = """

Given these documents, answer the question.

## Documents
{% for doc in documents %}
### {{ doc.meta['og:title'] }}
{{ doc.content | replace("\n", "\\\\n") }}

{% endfor %}


## Question
{{ query }}
""".strip()


def get_pipeline(documents: List[Document], faqs: List[Document], config):
    assert len(config.retrievers) == len(config.join_weights)

    document_store = InMemoryDocumentStore(
        embedding_similarity_function=config.embedding_similarity_function
    )
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(model=config.embedding_model),
    )
    indexing_pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    indexing_pipeline.connect("embedder", "writer")
    indexing_pipeline.run({"documents": documents})
    print("Number of documents in store:", document_store.count_documents())

    pipeline = Pipeline()
    pipeline.add_component(
        "document_joiner",
        DocumentJoiner(join_mode=config.join_mode, top_k=config.top_k),
    )

    if "bm25" in config.retrievers:
        pipeline.add_component(
            "bm25_retriever",
            InMemoryBM25Retriever(
                document_store=document_store,
                top_k=config.bm25_k,
                scale_score=True,
            ),
        )
        pipeline.connect("bm25_retriever.documents", "document_joiner")

    if "oracle" in config.retrievers:
        pipeline.add_component(
            "oracle_retriever",
            BM25RetrieverWithOracle(
                document_store=document_store, top_k=config.bm25_k, mode="oracle"
            ),
        )
        pipeline.connect("oracle_retriever.documents", "document_joiner")

    if "faq" in config.retrievers:
        pipeline.add_component(
            "faq_retriever",
            FAQRetriever(
                documents=documents,
                faqs=faqs,
                top_k=config.faq_k,
                embedding_model=config.faq_embedding_model,
                embedding_similarity_function=config.faq_embedding_similarity_function,
            ),  # type: ignore
        )
        pipeline.connect("faq_retriever.documents", "document_joiner")

    if "dense" in config.retrievers:
        pipeline.add_component(
            "dense_embedder",
            SentenceTransformersTextEmbedder(
                model=config.embedding_model, progress_bar=False
            ),
        )
        pipeline.add_component(
            "dense_retriever",
            InMemoryEmbeddingRetriever(
                document_store=document_store,
                top_k=config.dense_k,
                scale_score=True,
            ),
        )
        pipeline.connect("dense_embedder.embedding", "dense_retriever.query_embedding")
        pipeline.connect("dense_retriever.documents", "document_joiner")

    if "hyde" in config.retrievers:
        pipeline.add_component(
            "hyde_embedder",
            HyDE(
                generator_model=config.hyde_generator_model,
                embedding_model=config.embedding_model,
                n=config.hyde_n,
            ),
        )
        pipeline.add_component(
            "hyde_retriever",
            InMemoryEmbeddingRetriever(
                document_store=document_store, top_k=config.hyde_k, scale_score=True
            ),
        )
        pipeline.connect("hyde_embedder.embedding", "hyde_retriever.query_embedding")
        pipeline.connect("hyde_retriever.documents", "document_joiner")

    if config.use_reranker:
        pipeline.add_component(
            "reranker",
            SentenceTransformersSimilarityRanker(
                model=config.reranker_model,
                top_k=config.top_k,
            ),
        )
        pipeline.connect("document_joiner", "reranker")

    if config.use_generator:
        link_normalizer = ContentLinkNormalizer()
        prompt_builder = ChatPromptBuilder(
            variables=["documents"], required_variables=["documents", "query"]
        )
        llm = OpenAIChatGenerator(
            model=config.generation_model,
            api_base_url=os.environ["LLM_BASE_URL"],
            api_key=Secret.from_env_var("LLM_API_KEY"),
            generation_kwargs={
                "temperature": config.generation_temperature,
                "n": 1,
                "max_tokens": config.generation_max_tokens,
            },
        )
        # with n > 1 we get problems with Gemma on vLLM.
        # Therefore, send n independent requests with this wrapper component.
        multi_llm_wrapper = OpenAIChatGeneratorMultipleSamples(
            llm, n=config.generation_n
        )

        pipeline.add_component("link_normalizer", link_normalizer)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", multi_llm_wrapper)

        if config.use_reranker:
            pipeline.connect("reranker", "link_normalizer")
        else:
            pipeline.connect("document_joiner", "link_normalizer")

        pipeline.connect("link_normalizer", "prompt_builder")
        pipeline.connect("prompt_builder.prompt", "llm.messages")

    return pipeline


def run_pipeline(pipeline, query):
    pipeline_input = {}
    if "hyde_embedder" in pipeline.inputs():
        pipeline_input["hyde_embedder"] = {"text": query["question"]}

    if "dense_embedder" in pipeline.inputs():
        pipeline_input["dense_embedder"] = {"text": query["question"]}

    if "oracle_retriever" in pipeline.inputs():
        pipeline_input["oracle_retriever"] = {
            "query": query["question"],
            "filters": {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.url", "operator": "==", "value": source}
                    for source in query["sources"]
                ],
            },
        }

    if "bm25_retriever" in pipeline.inputs():
        pipeline_input["bm25_retriever"] = {"query": query["question"]}

    if "faq_retriever" in pipeline.inputs():
        pipeline_input["faq_retriever"] = {"text": query["question"]}

    if "reranker" in pipeline.inputs():
        pipeline_input["reranker"] = {"query": query["question"]}
        final_retriever = "reranker"
    else:
        final_retriever = "document_joiner"

    try:
        try:
            pipeline.get_component("llm")
            has_generator = True
        except ValueError:
            has_generator = False

        if has_generator:
            pipeline_input["prompt_builder"] = {
                "template": [ChatMessage.from_system(system_prompt_rag)]
                + [ChatMessage.from_user(user_prompt_template_rag)],
                "template_variables": {"query": query["question"]},
            }
            result = pipeline.run(
                pipeline_input,
                include_outputs_from={final_retriever, "llm"},
            )
            result = {
                "generated_answer": [r.text for r in result["llm"]["replies"]],
                "documents": result[final_retriever]["documents"],
            }
        else:
            result = pipeline.run(
                pipeline_input,
                include_outputs_from={final_retriever},
            )
            result = {
                "generated_answer": "",
                "documents": result[final_retriever]["documents"],
            }
    except Exception:
        logging.exception(f"Failed to generate response for {query['id']}")
        result = {"generated_answer": "", "documents": []}

    return result


def main(args):
    documents = data_loader.load_documents(args.data_path)
    queries = data_loader.load_queries(
        args.query_path, skip_without_sources=args.skip_without_sources
    )

    faqs = []
    if "faq" in args.retrievers:
        faqs = data_loader.load_faqs(args.faq_path)

    print(f"documents = {len(documents)}")
    print(f"queries = {len(queries)}")
    print(f"faqs = {len(faqs)}")

    pipeline = get_pipeline(documents, faqs, args)
    config = vars(args)
    config["run_id"] = Path(args.out_path).name

    run_experiment(
        pipeline,
        run_pipeline,
        queries=queries,
        run_path=args.out_path,
        documents=documents,
        config=vars(args),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    # =======================================
    # Input / output paths
    # =======================================
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL file containing documents.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the JSON file containing queries.")
    parser.add_argument("--faq_path", type=str, required=False, help="Path to FAQs (only for FAQ retriever).")
    parser.add_argument("--out_path", type=str, required=True, help="Path where experiment output will be stored.")
    parser.add_argument("--skip-without-sources", action=argparse.BooleanOptionalAction, default=True, help="Only use queries which have ground-truth sources (useful for retriever-only evaluation).")

    # =======================================
    # General retriever settings
    # =======================================
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--retrievers", type=str, nargs="+", choices=["bm25", "dense", "hyde", "faq", "oracle"], default=["bm25"])
    parser.add_argument("--join_mode", type=str, default="reciprocal_rank_fusion")
    parser.add_argument("--join_weights", type=float, nargs="+", required=False)

    # =======================================
    # BM25, Dense, FAQ, HyDE settings
    # =======================================
    parser.add_argument("--bm25_k", type=int, default=50)
    parser.add_argument("--dense_k", type=int, default=50)
    parser.add_argument("--faq_k", type=int, default=1)
    parser.add_argument("--faq_embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--faq_embedding_similarity_function", type=str, default="cosine")
    parser.add_argument("--hyde_k", type=int, default=50)
    parser.add_argument("--hyde_n", type=int, default=3)
    parser.add_argument("--hyde_generator_model", type=str, default="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8")

    # =======================================
    # Document embedding model (HyDE, Dense)
    # =======================================
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding_similarity_function", type=str, default="cosine")

    # =======================================
    # Reranker
    # =======================================
    parser.add_argument("--use_reranker", action="store_true", default=False)
    parser.add_argument("--reranker_model", type=str)

    # =======================================
    # Generator
    # =======================================
    parser.add_argument("--use_generator", action="store_true", default=False)
    parser.add_argument("--generation_model", type=str, default=None)
    parser.add_argument("--generation_temperature", type=float, default=0.7)
    parser.add_argument("--generation_n", type=int, default=3)
    parser.add_argument("--generation_max_tokens", type=int, default=512)
    # fmt: on

    args = parser.parse_args()

    if not args.join_weights:
        args.join_weights = [1] * len(args.retrievers)  # uniform
    return args


if __name__ == "__main__":
    main(parse_args())
