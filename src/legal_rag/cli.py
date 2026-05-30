import argparse
import json
from pathlib import Path

from .pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Turkish legal RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask")
    ask.add_argument("--dataset", type=Path, default=Path("Datasets_Ceng493_legal_rag"))
    ask.add_argument("--custom-docs", type=Path)
    ask.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ask.add_argument("--mode", choices=["base", "fine_tuned", "adapted_retriever", "adapted_reranker", "adapted_llm"], default="base")
    ask.add_argument("--question", required=True)
    ask.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "ask":
        pipeline = build_pipeline(args.dataset, args.artifacts, args.mode, args.custom_docs)
        result = pipeline.answer(args.question)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["answer"])
            print()
            print("Retrieved contexts:")
            for context in result["contexts"]:
                print(f"- {context['id']}: {context['citation']}")


if __name__ == "__main__":
    main()

