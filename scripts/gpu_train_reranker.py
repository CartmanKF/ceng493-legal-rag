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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.legal_rag.io import read_jsonl


def collate(batch, tokenizer, max_length):
    queries = [row["query"] for row in batch]
    passages = [row["candidate_passage"] for row in batch]
    labels = torch.tensor([float(row["label"]) for row in batch])
    encoded = tokenizer(queries, passages, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded["labels"] = labels
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--out", type=Path, default=Path("artifacts/gpu/reranker_model"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_jsonl(args.dataset / "reranker.jsonl")
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, trust_remote_code=True).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    losses = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="reranker")
    while step < args.max_steps:
        for micro_step, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels) / args.grad_accum
            loss.backward()
            if (micro_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()) * args.grad_accum)
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

