import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.legal_rag.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/full_finetune_data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    llm_rows = read_jsonl(args.dataset / "llm.jsonl")
    with (args.out / "llm_sft.jsonl").open("w", encoding="utf-8") as handle:
        for row in llm_rows:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    embedding_rows = read_jsonl(args.dataset / "embedding.jsonl")
    with (args.out / "embedding_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in embedding_rows:
            handle.write(
                json.dumps(
                    {
                        "query": row["query"],
                        "positive": row["positive_passage"],
                        "negative": row["negative_passage"],
                        "positive_id": row.get("positive_id"),
                        "negative_id": row.get("negative_id"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    reranker_rows = read_jsonl(args.dataset / "reranker.jsonl")
    with (args.out / "reranker_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in reranker_rows:
            handle.write(
                json.dumps(
                    {"query": row["query"], "passage": row["candidate_passage"], "label": row["label"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Prepared full finetuning data under {args.out}")


if __name__ == "__main__":
    main()
