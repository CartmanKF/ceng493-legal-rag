import json
from pathlib import Path

from .generator import generate_answer
from .io import load_corpus, load_custom_documents
from .retrieval import BM25Retriever, rerank


def load_artifacts(artifacts_dir: Path) -> dict:
    artifacts = {}
    for name in ["retriever_weights.json", "reranker_weights.json"]:
        path = artifacts_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                artifacts[name] = json.load(handle)
        else:
            artifacts[name] = {}
    return artifacts


class RAGPipeline:
    def __init__(
        self,
        documents: list[dict],
        retriever_weights: dict[str, float] | None = None,
        reranker_weights: dict[str, float] | None = None,
        fine_tuned_style: bool = False,
    ):
        self.retriever = BM25Retriever(documents, retriever_weights)
        self.reranker_weights = reranker_weights or {}
        self.fine_tuned_style = fine_tuned_style

    def answer(self, question: str, retrieve_k: int = 20, final_k: int = 5) -> dict:
        retrieved = self.retriever.search(question, top_k=retrieve_k)
        reranked = rerank(question, retrieved, self.reranker_weights, top_k=final_k)
        docs = [item.document for item in reranked]
        generated = generate_answer(question, docs, self.fine_tuned_style)
        return {
            "question": question,
            "answer": generated["answer"],
            "citations": generated["citations"],
            "contexts": [
                {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "citation": (doc.get("metadata") or {}).get("citation_label") or doc.get("id"),
                    "text": doc.get("text", ""),
                }
                for doc in docs
            ],
        }


def build_pipeline(dataset_dir: Path | None, artifacts_dir: Path, mode: str, custom_docs: Path | None = None) -> RAGPipeline:
    documents = load_custom_documents(custom_docs) if custom_docs else load_corpus(dataset_dir)
    artifacts = load_artifacts(artifacts_dir)
    retriever_weights = artifacts["retriever_weights.json"] if mode in {"fine_tuned", "adapted_retriever", "full"} else {}
    reranker_weights = artifacts["reranker_weights.json"] if mode in {"fine_tuned", "adapted_reranker", "full"} else {}
    fine_tuned_style = mode in {"fine_tuned", "adapted_llm", "full"}
    return RAGPipeline(documents, retriever_weights, reranker_weights, fine_tuned_style)
