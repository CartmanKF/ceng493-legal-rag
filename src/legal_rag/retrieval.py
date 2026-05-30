import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .text import token_counts, tokenize


@dataclass
class ScoredDocument:
    document: dict
    score: float


class BM25Retriever:
    def __init__(self, documents: list[dict], term_weights: dict[str, float] | None = None):
        self.documents = documents
        self.term_weights = term_weights or {}
        self.doc_tokens = [token_counts(self._doc_text(doc)) for doc in documents]
        self.doc_lengths = [sum(counts.values()) for counts in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.df = Counter()
        for counts in self.doc_tokens:
            self.df.update(counts.keys())

    def search(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        query_tokens = tokenize(query)
        scores = []
        for index, counts in enumerate(self.doc_tokens):
            score = self._score(query_tokens, counts, self.doc_lengths[index])
            if score > 0:
                scores.append(ScoredDocument(self.documents[index], score))
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]

    def _score(self, query_tokens: list[str], counts: Counter, doc_len: int) -> float:
        k1 = 1.4
        b = 0.72
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = self.df.get(token, 0)
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
            score += idf * (numerator / denominator) * self.term_weights.get(token, 1.0)
        return score

    @staticmethod
    def _doc_text(doc: dict) -> str:
        metadata = doc.get("metadata") or {}
        concepts = " ".join(metadata.get("legal_concepts") or [])
        return f"{doc.get('title', '')} {metadata.get('category', '')} {concepts} {doc.get('text', '')}"


def train_retriever_weights(rows: list[dict]) -> dict[str, float]:
    positive = defaultdict(float)
    negative = defaultdict(float)
    for row in rows:
        query_tokens = set(tokenize(row.get("query", "")))
        pos_tokens = set(tokenize(row.get("positive_passage", "")))
        neg_tokens = set(tokenize(row.get("negative_passage", "")))
        for token in query_tokens & pos_tokens:
            positive[token] += 1.0
        for token in query_tokens & neg_tokens:
            negative[token] += 1.0
    weights = {}
    vocab = set(positive) | set(negative)
    for token in vocab:
        ratio = (positive[token] + 2.0) / (negative[token] + 2.0)
        weights[token] = max(0.75, min(2.5, 1.0 + math.log(ratio) * 0.35))
    return weights


def train_reranker_weights(rows: list[dict]) -> dict[str, float]:
    positive = defaultdict(float)
    negative = defaultdict(float)
    for row in rows:
        query_tokens = set(tokenize(row.get("query", "")))
        passage_tokens = set(tokenize(row.get("candidate_passage", "")))
        overlap = query_tokens & passage_tokens
        target = positive if int(row.get("label", 0)) == 1 else negative
        for token in overlap:
            target[token] += 1.0
    weights = {}
    vocab = set(positive) | set(negative)
    for token in vocab:
        ratio = (positive[token] + 1.5) / (negative[token] + 1.5)
        weights[token] = max(-0.8, min(1.2, math.log(ratio) * 0.4))
    return weights


def rerank(query: str, candidates: list[ScoredDocument], weights: dict[str, float] | None, top_k: int) -> list[ScoredDocument]:
    if not weights:
        return candidates[:top_k]
    query_terms = set(tokenize(query))
    reranked = []
    for item in candidates:
        doc_terms = set(tokenize(BM25Retriever._doc_text(item.document)))
        bonus = sum(weights.get(token, 0.0) for token in query_terms & doc_terms)
        metadata = item.document.get("metadata") or {}
        exact_topic = metadata.get("semantic_topic", "")
        topic_bonus = 0.25 if query_terms & set(tokenize(exact_topic)) else 0.0
        reranked.append(ScoredDocument(item.document, item.score + bonus + topic_bonus))
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:top_k]

