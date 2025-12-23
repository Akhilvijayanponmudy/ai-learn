from typing import List, Dict, Tuple
import re

# In-memory “index” (replace with vector DB / BM25 later)
DOCS = [
    {"doc_id": "refund_policy", "text": "Refunds are available within 14 days of purchase if conditions are met."},
    {"doc_id": "api_keys", "text": "To rotate API keys: create a new key, update clients, verify, then revoke the old key."},
    {"doc_id": "shipping", "text": "Standard shipping takes 3-5 business days depending on location."},
    {"doc_id": "billing", "text": "Billing issues can be resolved by checking payment method and invoice history."},
]

# def _tokenize(s: str) -> set:
#     return set(re.findall(r"[a-zA-Z']+", s.lower()))

def _tokenize(s: str) -> set:
    toks = re.findall(r"[a-zA-Z']+", s.lower())
    norm = []
    for t in toks:
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]  # refunds -> refund
        norm.append(t)
    return set(norm)


def search(query: str, top_k: int) -> List[Dict]:
    q = _tokenize(query)
    hits: List[Tuple[float, Dict]] = []

    for d in DOCS:
        t = _tokenize(d["text"])
        if not t:
            continue
        # simple overlap score -> [0..1]
        overlap = len(q & t) / max(1, len(q))
        if overlap > 0:
            hits.append((overlap, d))

    hits.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, d in hits[:top_k]:
        out.append({
            "doc_id": d["doc_id"],
            "score": float(max(0.0, min(1.0, score))),
            "snippet": d["text"][:200]
        })
    return out

def fetch_contexts(hits: List[Dict]) -> List[str]:
    # Here snippet == context. In real RAG you’d fetch full chunk text by doc_id/chunk_id.
    return [h["snippet"] for h in hits]
