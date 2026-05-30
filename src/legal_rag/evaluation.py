import math

from .pipeline import RAGPipeline
from .text import exact_match, rouge_l, sentence_split, token_f1, tokenize


def gold_document_ids(row: dict) -> set[str]:
    gold_ids = set(row.get("gold_chunk_ids") or row.get("relevant_documents") or row.get("relevant_doc_ids") or [])
    if not gold_ids:
        gold_ids = {source.get("corpus_row_id") or source.get("source_id") for source in row.get("gold_sources", [])}
    return {doc_id for doc_id in gold_ids if doc_id}


def gold_citation_markers(row: dict) -> set[str]:
    markers = set(gold_document_ids(row))
    for source in row.get("gold_sources", []):
        markers.update(
            value
            for value in [
                source.get("citation_label"),
                source.get("source_id"),
                source.get("corpus_row_id"),
            ]
            if value
        )
    return markers


def reference_answer(row: dict) -> str:
    return row.get("verified_answer") or row.get("gold_answer_extract") or row.get("answer") or row.get("gold_answer") or ""


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for index, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def faithfulness_score(answer: str, contexts: list[dict]) -> float:
    context_tokens = set(tokenize(" ".join(context.get("text", "") for context in contexts)))
    answer_tokens = [token for token in tokenize(answer) if not token.startswith("kaynak")]
    if not answer_tokens:
        return 0.0
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)


def unsupported_sentence_rate(answer: str, contexts: list[dict]) -> float:
    context_tokens = set(tokenize(" ".join(context.get("text", "") for context in contexts)))
    sentences = sentence_split(answer)
    if not sentences:
        return 0.0
    unsupported = 0
    for sentence in sentences:
        tokens = tokenize(sentence)
        if tokens and sum(1 for token in tokens if token in context_tokens) / len(tokens) < 0.45:
            unsupported += 1
    return unsupported / len(sentences)


def evaluate_rag(pipeline: RAGPipeline, benchmark: list[dict], limit: int | None = None) -> dict:
    rows = benchmark[:limit] if limit else benchmark
    retrieval_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    ndcg_5 = []
    ndcg_10 = []
    answer_f1 = []
    answer_em = []
    answer_rouge_l = []
    citation_hits = 0
    exact_citation_hits = 0
    grounded = 0
    faithfulness = []
    unsupported_rates = []
    examples = []
    for row in rows:
        question = row.get("question") or row.get("query")
        gold_ids = gold_document_ids(row)
        result = pipeline.answer(question, retrieve_k=50, final_k=10)
        retrieved_ids = [context["id"] for context in result["contexts"]]
        for k in retrieval_hits:
            if gold_ids & set(retrieved_ids[:k]):
                retrieval_hits[k] += 1
        rank = next((idx + 1 for idx, doc_id in enumerate(retrieved_ids) if doc_id in gold_ids), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        ndcg_5.append(ndcg_at_k(retrieved_ids, gold_ids, 5))
        ndcg_10.append(ndcg_at_k(retrieved_ids, gold_ids, 10))
        reference = reference_answer(row)
        answer_f1.append(token_f1(result["answer"], reference))
        answer_em.append(exact_match(result["answer"], reference))
        answer_rouge_l.append(rouge_l(result["answer"], reference))
        citation_text = " ".join(result["citations"])
        if any(marker in citation_text for marker in gold_citation_markers(row)):
            citation_hits += 1
        if gold_ids and any(gold_id == context["id"] for gold_id in gold_ids for context in result["contexts"][:1]):
            exact_citation_hits += 1
        if result["citations"] and ("Kaynak" in result["answer"] or "Citation" in result["answer"] or "Citations" in result["answer"]):
            grounded += 1
        faithfulness.append(faithfulness_score(result["answer"], result["contexts"]))
        unsupported_rates.append(unsupported_sentence_rate(result["answer"], result["contexts"]))
        if len(examples) < 5:
            examples.append({"question": question, "gold_ids": sorted(gold_ids), "retrieved_ids": retrieved_ids, "answer": result["answer"]})
    total = max(len(rows), 1)
    return {
        "n": len(rows),
        "recall@1": retrieval_hits[1] / total,
        "recall@3": retrieval_hits[3] / total,
        "recall@5": retrieval_hits[5] / total,
        "recall@10": retrieval_hits[10] / total,
        "mrr@10": sum(reciprocal_ranks) / total,
        "ndcg@5": sum(ndcg_5) / total,
        "ndcg@10": sum(ndcg_10) / total,
        "answer_exact_match": sum(answer_em) / total,
        "answer_token_f1": sum(answer_f1) / total,
        "answer_rouge_l": sum(answer_rouge_l) / total,
        "citation_hit_rate": citation_hits / total,
        "top1_citation_accuracy": exact_citation_hits / total,
        "grounded_answer_rate": grounded / total,
        "faithfulness_token_support": sum(faithfulness) / total,
        "unsupported_sentence_rate": sum(unsupported_rates) / total,
        "examples": examples,
    }
