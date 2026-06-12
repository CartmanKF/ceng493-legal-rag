import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.legal_rag.evaluation import evaluate_rag
from src.legal_rag.io import read_json, write_json
from src.legal_rag.pipeline import build_pipeline


MODES = ["base", "adapted_retriever", "adapted_reranker", "adapted_llm", "fine_tuned"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--benchmark", choices=["gold", "rag_eval"], default="gold")
    parser.add_argument("--benchmark-file", type=Path, help="Evaluator-provided JSON benchmark file.")
    parser.add_argument("--custom-docs", type=Path, help="Evaluator-provided document collection.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=Path("reports/evaluation_results.json"))
    args = parser.parse_args()

    benchmark_path = args.benchmark_file or args.dataset / ("gold_benchmark.json" if args.benchmark == "gold" else "rag_eval.json")
    benchmark = read_json(benchmark_path)
    results = {}
    for mode in MODES:
        pipeline = build_pipeline(args.dataset, args.artifacts, mode, args.custom_docs)
        results[mode] = evaluate_rag(pipeline, benchmark, args.limit)
    write_json(args.out, {"benchmark": str(benchmark_path), "custom_docs": str(args.custom_docs) if args.custom_docs else None, "results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
