import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"
STORE = Path("../week_3/index_store")  # <-- adjust if needed
EVAL = Path("data/generated/rag_eval.jsonl")
OUT = Path("data/generated/retrieval_metrics.json")

TOP_KS = [1, 3, 5, 10]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def load_store():
    index = faiss.read_index(str(STORE / "docs.index"))
    meta = json.loads((STORE / "meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((STORE / "chunks.json").read_text(encoding="utf-8"))
    return index, meta, chunks

def retrieve_sources(query: str, k: int, index, meta, emb_model) -> List[str]:
    q = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q).astype("float32")
    scores, idxs = index.search(q, k)
    sources = []
    for idx in idxs[0]:
        if idx == -1:
            continue
        sources.append(Path(meta[idx]["source"]).name)
    return sources

def load_eval():
    items = []
    with EVAL.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def recall_at_k(pred_sources: List[str], gold_sources: List[str], k: int) -> float:
    gold = set([Path(s).name for s in gold_sources])
    pred = pred_sources[:k]
    return 1.0 if any(p in gold for p in pred) else 0.0

def mrr_at_k(pred_sources: List[str], gold_sources: List[str], k: int) -> float:
    gold = set([Path(s).name for s in gold_sources])
    for i, p in enumerate(pred_sources[:k], start=1):
        if p in gold:
            return 1.0 / i
    return 0.0

def main():
    index, meta, _ = load_store()
    emb_model = SentenceTransformer(MODEL_NAME)
    items = load_eval()

    totals = {k: {"recall": 0.0, "mrr": 0.0} for k in TOP_KS}

    for it in tqdm(items, desc="Retrieval eval"):
        q = it["question"]
        gold_sources = it.get("gold_sources", [])
        # retrieve max K once
        max_k = max(TOP_KS)
        pred_sources = retrieve_sources(q, max_k, index, meta, emb_model)

        for k in TOP_KS:
            totals[k]["recall"] += recall_at_k(pred_sources, gold_sources, k)
            totals[k]["mrr"] += mrr_at_k(pred_sources, gold_sources, k)

    n = len(items)
    metrics = {"n": n, "top_ks": TOP_KS, "results": {}}
    for k in TOP_KS:
        metrics["results"][f"@{k}"] = {
            "recall": totals[k]["recall"] / n,
            "mrr": totals[k]["mrr"] / n
        }

    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {OUT}")
    for k in TOP_KS:
        r = metrics["results"][f"@{k}"]["recall"]
        m = metrics["results"][f"@{k}"]["mrr"]
        print(f"Recall@{k}={r:.3f}  MRR@{k}={m:.3f}")

if __name__ == "__main__":
    main()
