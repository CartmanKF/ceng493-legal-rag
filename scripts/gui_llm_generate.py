import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def clean_answer(answer: str) -> str:
    for marker in ["\n\nSoru:", "\nSoru:", "\n\nKaynaklar:", "\nKaynaklar:"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()
    if "Cevap:" in answer:
        answer = answer.split("Cevap:", 1)[-1].strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    blocks = []
    for index, context in enumerate(payload["contexts"][:3], start=1):
        blocks.append(
            "[{}] Baslik: {}\nCitation: {}\nMetin: {}".format(
                index,
                context.get("title") or "",
                context.get("citation") or context.get("id"),
                context.get("text") or "",
            )
        )
    messages = [
        {
            "role": "system",
            "content": "Sen bir Turk hukuku soru cevap asistanisin. Sadece verilen kaynaklari ve taslak cevabi kullan. Cevap kisa, dogrudan ve Turkce olsun. Kaynakta olmayan bilgi ekleme.",
        },
        {
            "role": "user",
            "content": "Kaynaklar:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nTaslak cevap:\n{payload.get('draft_answer', '')}\n\nSoru: {payload['question']}\n\nBu taslagi kaynaklara bagli kalarak tek paragraf halinde duzelt ve cevap ver:",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    args.output.write_text(json.dumps({"answer": clean_answer(answer)}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
