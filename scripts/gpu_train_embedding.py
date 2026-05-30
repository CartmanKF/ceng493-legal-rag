import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.legal_rag.io import read_jsonl


def mean_pool(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def collate(batch, tokenizer, max_length):
    queries = ["query: " + row["query"] for row in batch]
    positives = ["passage: " + row["positive_passage"] for row in batch]
    negatives = ["passage: " + row["negative_passage"] for row in batch]
    return {
        "query": tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
        "positive": tokenizer(positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
        "negative": tokenizer(negatives, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
    }


def encode(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return F.normalize(mean_pool(model(**inputs), inputs["attention_mask"]), p=2, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--out", type=Path, default=Path("artifacts/gpu/embedding_model"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_jsonl(args.dataset / "embedding.jsonl")
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    losses = []
    step = 0
    progress = tqdm(total=args.max_steps, desc="embedding")
    while step < args.max_steps:
        for batch in loader:
            query = encode(model, batch["query"], device)
            positive = encode(model, batch["positive"], device)
            negative = encode(model, batch["negative"], device)
            pos_scores = (query * positive).sum(dim=1)
            neg_scores = (query * negative).sum(dim=1)
            loss = F.relu(0.2 - pos_scores + neg_scores).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            step += 1
            progress.update(1)
            if step >= args.max_steps:
                break
    progress.close()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    summary = {"model": args.model, "device": str(device), "steps": step, "avg_loss": sum(losses) / max(len(losses), 1)}
    (args.out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

