
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    tokens = [w for w in s.split() if w not in _ARTICLES]
    return " ".join(tokens)

def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

def best_em_f1(pred: str, golds):
    if isinstance(golds, str):
        golds = [golds]
    em = max(1.0 if exact_match(pred, g) else 0.0 for g in golds) if golds else 0.0
    f1 = max(f1_score(pred, g) for g in golds) if golds else 0.0
    return em, f1
