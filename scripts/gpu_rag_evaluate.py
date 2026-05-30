import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from src.legal_rag.evaluation import (
    faithfulness_score,
    gold_citation_markers,
    gold_document_ids,
    ndcg_at_k,
    reference_answer,
    unsupported_sentence_rate,
)
from src.legal_rag.io import load_corpus, load_custom_documents, read_json, write_json
from src.legal_rag.text import exact_match, rouge_l, token_f1


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


def build_dense_index(docs, model_path, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device = next(model.parameters()).device
    vectors = []
    loader = DataLoader(docs, batch_size=batch_size, collate_fn=lambda batch: batch)
    for batch in tqdm(loader, desc="encode_docs"):
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in batch]
        vectors.append(encode_texts(model, tokenizer, texts, "passage: ", device, max_length))
    matrix = np.vstack(vectors)
    return tokenizer, model, matrix


def dense_retrieve(question, docs, matrix, tokenizer, model, top_k, max_length):
    device = next(model.parameters()).device
    query = encode_texts(model, tokenizer, [question], "query: ", device, max_length)[0]
    scores = matrix @ query
    order = np.argsort(-scores)[:top_k]
    return [(docs[index], float(scores[index])) for index in order]


def load_reranker(reranker_path):
    if not reranker_path:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(reranker_path)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_path, trust_remote_code=True).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    return tokenizer, model


def rerank(question, candidates, reranker, max_length, top_k):
    tokenizer, model = reranker
    if model is None:
        return candidates[:top_k]
    reranked = []
    for doc, retrieval_score in tqdm(candidates, desc="rerank", leave=False):
        encoded = tokenizer(question, doc.get("text", ""), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
        with torch.no_grad():
            score = float(model(**encoded).logits.view(-1)[0].detach().cpu())
        reranked.append((doc, score, retrieval_score))
    reranked.sort(key=lambda item: item[1], reverse=True)
    return [(doc, score) for doc, score, _ in reranked[:top_k]]


def load_generator(model_name, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question, docs, max_new_tokens):
    context_blocks = []
    citations = []
    for index, doc in enumerate(docs, start=1):
        citation = (doc.get("metadata") or {}).get("citation_label") or doc.get("id")
        citations.append(citation)
        context_blocks.append(f"[{index}] Başlık: {doc.get('title', '')}\nCitation: {citation}\nMetin: {doc.get('text', '')}")
    messages = [
        {
            "role": "system",
            "content": "Sen bir Türk hukuku RAG asistanısın. Yalnızca verilen kaynaklara dayanarak cevap ver. Kaynakta olmayan bilgiyi üretme. Cevabın sonunda kaynak/citation belirt.",
        },
        {"role": "user", "content": "Kaynaklar:\n\n" + "\n\n".join(context_blocks) + f"\n\nSoru: {question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return generated, citations


def evaluate_system(name, docs, benchmark, embed_model, reranker_model, llm_model, adapter_path, args):
    embed_tokenizer, embedder, matrix = build_dense_index(docs, embed_model, args.batch_size, args.embed_max_length)
    retrieval_payload = []
    for row in tqdm(benchmark, desc=f"{name}_retrieve"):
        question = row.get("question") or row.get("query")
        candidates = dense_retrieve(question, docs, matrix, embed_tokenizer, embedder, args.retrieve_k, args.embed_max_length)
        retrieval_payload.append((row, candidates))
    del embedder
    torch.cuda.empty_cache()

    reranker = load_reranker(reranker_model)
    reranked_payload = []
    for row, candidates in retrieval_payload:
        question = row.get("question") or row.get("query")
        final = rerank(question, candidates, reranker, args.rerank_max_length, args.final_k)
        reranked_payload.append((row, [doc for doc, _ in final]))
    if reranker[1] is not None:
        del reranker
        torch.cuda.empty_cache()

    generator_tokenizer = generator_model = None
    if not args.no_generation:
        generator_tokenizer, generator_model = load_generator(llm_model, adapter_path)

    metrics = {
        "recall@1": 0,
        "recall@5": 0,
        "recall@10": 0,
        "mrr@10": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "answer_exact_match": [],
        "answer_token_f1": [],
        "answer_rouge_l": [],
        "citation_hit_rate": 0,
        "faithfulness_token_support": [],
        "unsupported_sentence_rate": [],
    }
    examples = []
    for row, final_docs in tqdm(reranked_payload, desc=f"{name}_generate"):
        question = row.get("question") or row.get("query")
        gold_ids = gold_document_ids(row)
        retrieved_ids = [doc.get("id") for doc in final_docs]
        for k in [1, 5, 10]:
            if gold_ids & set(retrieved_ids[:k]):
                metrics[f"recall@{k}"] += 1
        rank = next((idx + 1 for idx, doc_id in enumerate(retrieved_ids[:10]) if doc_id in gold_ids), None)
        metrics["mrr@10"].append(1.0 / rank if rank else 0.0)
        metrics["ndcg@5"].append(ndcg_at_k(retrieved_ids, gold_ids, 5))
        metrics["ndcg@10"].append(ndcg_at_k(retrieved_ids, gold_ids, 10))
        if args.no_generation:
            answer = ""
            citations = [(doc.get("metadata") or {}).get("citation_label") or doc.get("id") for doc in final_docs]
        else:
            answer, citations = generate_answer(generator_tokenizer, generator_model, question, final_docs[: args.context_k], args.max_new_tokens)
        reference = reference_answer(row)
        metrics["answer_exact_match"].append(exact_match(answer, reference))
        metrics["answer_token_f1"].append(token_f1(answer, reference))
        metrics["answer_rouge_l"].append(rouge_l(answer, reference))
        if any(marker in " ".join(citations) for marker in gold_citation_markers(row)):
            metrics["citation_hit_rate"] += 1
        context_dicts = [{"text": doc.get("text", "")} for doc in final_docs[: args.context_k]]
        metrics["faithfulness_token_support"].append(faithfulness_score(answer, context_dicts) if answer else 0.0)
        metrics["unsupported_sentence_rate"].append(unsupported_sentence_rate(answer, context_dicts) if answer else 0.0)
        if len(examples) < 5:
            examples.append({"question": question, "gold_ids": sorted(gold_ids), "retrieved_ids": retrieved_ids, "answer": answer})
    if generator_model is not None:
        del generator_model
        torch.cuda.empty_cache()
    total = max(len(reranked_payload), 1)
    summary = {"n": len(reranked_payload), "embedding_model": embed_model, "reranker_model": reranker_model, "llm_model": llm_model, "adapter": adapter_path}
    for key, value in metrics.items():
        if isinstance(value, list):
            summary[key] = sum(value) / total
        else:
            summary[key] = value / total
    summary["examples"] = examples
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("Datasets_Ceng493_legal_rag"))
    parser.add_argument("--custom-docs", type=Path)
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--benchmark", choices=["gold", "rag_eval"], default="gold")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--base-embedding", default="intfloat/multilingual-e5-base")
    parser.add_argument("--fine-embedding", default="artifacts/gpu/embedding_model")
    parser.add_argument("--base-reranker", default=None)
    parser.add_argument("--fine-reranker", default="artifacts/gpu/reranker_model")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm-adapter", default="artifacts/gpu/llm_lora")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-max-length", type=int, default=256)
    parser.add_argument("--rerank-max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/gpu_rag_results.json"))
    args = parser.parse_args()

    docs = load_custom_documents(args.custom_docs) if args.custom_docs else load_corpus(args.dataset)
    benchmark_path = args.benchmark_file or args.dataset / ("gold_benchmark.json" if args.benchmark == "gold" else "rag_eval.json")
    benchmark = read_json(benchmark_path)
    if args.limit:
        benchmark = benchmark[: args.limit]
    results = {
        "base_rag": evaluate_system("base", docs, benchmark, args.base_embedding, args.base_reranker, args.llm, None, args),
        "fine_tuned_rag": evaluate_system("fine_tuned", docs, benchmark, args.fine_embedding, args.fine_reranker, args.llm, args.llm_adapter, args),
    }
    write_json(args.out, {"benchmark": str(benchmark_path), "custom_docs": str(args.custom_docs) if args.custom_docs else None, "results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
