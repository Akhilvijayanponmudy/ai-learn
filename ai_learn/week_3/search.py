import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
STORE = Path("index_store")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def load_store():
    index = faiss.read_index(str(STORE / "docs.index"))
    meta = json.loads((STORE / "meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((STORE / "chunks.json").read_text(encoding="utf-8"))
    return index, meta, chunks


def search(query: str, k: int = 5) -> List[Tuple[float, dict, str]]:
    index, meta, chunks = load_store()
    model = SentenceTransformer(MODEL_NAME)

    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q).astype("float32")

    scores, idxs = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append((float(score), meta[idx], chunks[idx]))
    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 search.py "your query here" [k]')
        raise SystemExit(1)

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    results = search(query, k=k)
    print(f"\nQuery: {query}\nTop {k} results:\n")

    for i, (score, m, chunk) in enumerate(results, start=1):
        print(f"#{i}  score={score:.4f}")
        print(f"source: {m['source']}  chunk_id: {m['chunk_id']}  len: {m['char_len']}")
        print("-" * 80)
        print(chunk[:800].strip())  # preview
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
