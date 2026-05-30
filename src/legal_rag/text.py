import math
import re
import unicodedata
from collections import Counter


TURKISH_STOPWORDS = {
    "acaba", "ama", "ancak", "artık", "aslında", "az", "bazı", "belki",
    "biri", "birkaç", "birşey", "biz", "bu", "çok", "çünkü", "da", "daha",
    "de", "defa", "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi",
    "her", "hiç", "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin",
    "niye", "o", "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya",
    "yani", "bir", "olarak", "olan", "olduğu", "göre", "kapsamında",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return text.replace("ı", "i")


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", normalize(text))
    return [token for token in tokens if len(token) > 1 and token not in TURKISH_STOPWORDS]


def token_counts(text: str) -> Counter:
    return Counter(tokenize(text))


def token_f1(prediction: str, reference: str) -> float:
    pred = token_counts(prediction)
    ref = token_counts(reference)
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(tokenize(prediction)) == " ".join(tokenize(reference)))


def rouge_l(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for pred_token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def safe_log(value: float) -> float:
    return math.log(max(value, 1e-9))
