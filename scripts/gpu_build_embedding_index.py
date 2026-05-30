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

from src.legal_rag.io import load_corpus, load_custom_documents


def mean_pool(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_batch(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(["passage: " + text for text in texts], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        vectors = mean_pool(model(**encoded), encoded["attention_mask"])
    return F.normalize(vectors, p=2, dim=1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--custom-docs", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    docs = load_custom_documents(args.custom_docs) if args.custom_docs else load_corpus(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device = next(model.parameters()).device
    vectors = []
    loader = DataLoader(docs, batch_size=args.batch_size, collate_fn=lambda batch: batch)
    for batch in tqdm(loader, desc="index"):
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in batch]
        vectors.append(encode_batch(model, tokenizer, texts, device, args.max_length))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", np.vstack(vectors))
    (args.out / "documents.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    (args.out / "index_config.json").write_text(json.dumps({"model": args.model, "count": len(docs)}, indent=2), encoding="utf-8")
    print(f"Saved {len(docs)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
