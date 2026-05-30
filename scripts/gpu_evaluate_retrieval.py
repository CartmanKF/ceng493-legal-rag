import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.legal_rag.io import load_corpus, read_json, write_json


def mean_pool(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_texts(model, tokenizer, texts, prefix, device, max_length):
    encoded = tokenizer([prefix + text for text in texts], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        vectors = mean_pool(model(**encoded), encoded["attention_mask"])
    return F.normalize(vectors, p=2, dim=1).cpu().numpy()


def build_doc_matrix(model, tokenizer, docs, device, batch_size, max_length):
    vectors = []
    loader = DataLoader(docs, batch_size=batch_size, collate_fn=lambda batch: batch)
    for batch in tqdm(loader, desc="encode_docs"):
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in batch]
        vectors.append(encode_texts(model, tokenizer, texts, "passage: ", device, max_length))
    return np.vstack(vectors)


def evaluate(model_path, dataset, benchmark_name, limit, batch_size, max_length):
    docs = load_corpus(dataset)
    benchmark = read_json(dataset / ("gold_benchmark.json" if benchmark_name == "gold" else "rag_eval.json"))
    if limit:
        benchmark = benchmark[:limit]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device = next(model.parameters()).device
    doc_matrix = build_doc_matrix(model, tokenizer, docs, device, batch_size, max_length)
    doc_ids = [doc["id"] for doc in docs]
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal = []
    for row in tqdm(benchmark, desc="queries"):
        question = row.get("question") or row.get("query")
        gold_ids = set(row.get("gold_chunk_ids") or [])
        if not gold_ids:
            gold_ids = {source.get("corpus_row_id") or source.get("source_id") for source in row.get("gold_sources", [])}
        query_vector = encode_texts(model, tokenizer, [question], "query: ", device, max_length)[0]
        scores = doc_matrix @ query_vector
        ranked = np.argsort(-scores)[:10]
        retrieved = [doc_ids[index] for index in ranked]
        for k in hits:
            if gold_ids & set(retrieved[:k]):
                hits[k] += 1
        rank = next((idx + 1 for idx, doc_id in enumerate(retrieved) if doc_id in gold_ids), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    total = max(len(benchmark), 1)
    return {
        "model": str(model_path),
        "benchmark": benchmark_name,
        "n": len(benchmark),
        "recall@1": hits[1] / total,
        "recall@3": hits[3] / total,
        "recall@5": hits[5] / total,
        "recall@10": hits[10] / total,
        "mrr@10": sum(reciprocal) / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--fine-model", default="artifacts/gpu/embedding_model")
    parser.add_argument("--benchmark", choices=["gold", "rag_eval"], default="gold")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("reports/gpu_retrieval_results.json"))
    args = parser.parse_args()
    results = {
        "base_embedding": evaluate(args.base_model, args.dataset, args.benchmark, args.limit, args.batch_size, args.max_length),
        "fine_tuned_embedding": evaluate(args.fine_model, args.dataset, args.benchmark, args.limit, args.batch_size, args.max_length),
    }
    write_json(args.out, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

