import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_NAME = "all-MiniLM-L6-v2"
STORE = Path("../week_3/index_store")  # adjust if your week_3 is elsewhere
EVAL = Path("data/generated/rag_eval.jsonl")
OUT = Path("data/generated/rag_answers.jsonl")

TOP_K = 5
MIN_SCORE = 0.45  # gate nonsense queries


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def load_store():
    index = faiss.read_index(str(STORE / "docs.index"))
    meta = json.loads((STORE / "meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((STORE / "chunks.json").read_text(encoding="utf-8"))
    return index, meta, chunks


def retrieve(query: str, k: int, index, meta, chunks, emb_model) -> List[Dict]:
    q = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q).astype("float32")

    scores, idxs = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "source": Path(meta[idx]["source"]).name,
            "chunk_id": meta[idx]["chunk_id"],
            "text": chunks[idx],
        })
    return results


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def sentence_score(sent: str, query: str) -> float:
    # cheap relevance: token overlap with query
    qs = tokenize(query)
    ss = tokenize(sent)
    if not qs or not ss:
        return 0.0
    return len(qs & ss) / (len(qs) + 1e-9)


def build_extractive_answer(question: str, retrieved: List[Dict]) -> str:
    if not retrieved or retrieved[0]["score"] < MIN_SCORE:
        return "I don't know from the provided docs."

    # Collect candidate sentences with citations
    candidates: List[Tuple[float, str]] = []
    for i, r in enumerate(retrieved, start=1):
        tag = f"[C{i}]"
        for s in split_sentences(r["text"]):
            sc = sentence_score(s, question)
            if sc > 0:
                candidates.append((sc, f"{s} {tag}"))

    if not candidates:
        # fallback: return first chunk preview with citation
        return f"{retrieved[0]['text'].strip()[:240]}... [C1]"

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = [c[1] for c in candidates[:3]]  # take top 3 sentences
    return "\n".join(f"- {t}" for t in top)


def load_eval():
    items = []
    with EVAL.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    index, meta, chunks = load_store()
    emb_model = SentenceTransformer(MODEL_NAME)
    eval_items = load_eval()

    with OUT.open("w", encoding="utf-8") as f:
        for it in tqdm(eval_items, desc="Generate extractive answers"):
            q = it["question"]
            retrieved = retrieve(q, TOP_K, index, meta, chunks, emb_model)
            answer = build_extractive_answer(q, retrieved)

            f.write(json.dumps({
                "id": it["id"],
                "question": q,
                "answer": answer,
                "retrieved": retrieved,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
