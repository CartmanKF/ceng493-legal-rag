from .text import sentence_split, tokenize


BASE_SYSTEM = "Base RAG answer generator"
FINE_TUNED_SYSTEM = (
    "Sen bir Türk hukuku RAG asistanısın. Yalnızca verilen kaynaklara dayanarak cevap ver. "
    "Kaynakta olmayan bilgiyi üretme ve citation belirt."
)


def generate_answer(question: str, documents: list[dict], fine_tuned_style: bool = False) -> dict:
    if not documents:
        return {"answer": "Verilen dokümanlarda bu soruya cevap oluşturacak kaynak bulunamadı.", "citations": []}
    query_terms = set(tokenize(question))
    selected = []
    citations = []
    for doc in documents[:3]:
        sentences = sentence_split(doc.get("text", ""))
        ranked = sorted(sentences, key=lambda sentence: len(query_terms & set(tokenize(sentence))), reverse=True)
        best = ranked[0] if ranked else doc.get("text", "")
        selected.append(best)
        metadata = doc.get("metadata") or {}
        citations.append(metadata.get("citation_label") or doc.get("id"))
    prefix = "Kaynaga gore: " if fine_tuned_style else ""
    answer = prefix + " ".join(selected)
    if fine_tuned_style:
        answer += "\n\nKaynaklar: " + "; ".join(citations)
    else:
        answer += "\n\nCitations: " + "; ".join(citations)
    return {"answer": answer, "citations": citations}
