import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.legal_rag.io import read_jsonl


def format_messages(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def collate(batch, tokenizer, max_length):
    texts = [format_messages(tokenizer, row["messages"]) for row in batch]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("artifacts/gpu/llm_lora"))
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.dataset / "llm.jsonl")
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    losses = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="llm_lora")
    while step < args.max_steps:
        for micro_step, batch in enumerate(loader):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
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
    summary = {"model": args.model, "steps": step, "avg_loss": sum(losses) / max(len(losses), 1)}
    (args.out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

