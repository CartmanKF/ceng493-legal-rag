import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.legal_rag.io import read_jsonl
from src.legal_rag.retrieval import train_reranker_weights, train_retriever_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    embedding_rows = read_jsonl(args.dataset / "embedding.jsonl")
    reranker_rows = read_jsonl(args.dataset / "reranker.jsonl")

    retriever_weights = train_retriever_weights(embedding_rows)
    reranker_weights = train_reranker_weights(reranker_rows)

    (args.out / "retriever_weights.json").write_text(
        json.dumps(retriever_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "reranker_weights.json").write_text(
        json.dumps(reranker_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "embedding_training_rows": len(embedding_rows),
        "reranker_training_rows": len(reranker_rows),
        "retriever_weight_terms": len(retriever_weights),
        "reranker_weight_terms": len(reranker_weights),
    }
    (args.out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
