import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


DOCS_DIR = Path("data/docs")
OUT_DIR = Path("index_store")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"  # strong default, fast
CHUNK_SIZE = 1200               # characters (simple, works well to start)
OVERLAP = 200                   # characters overlap
BATCH_SIZE = 64


def read_text_files(docs_dir: Path) -> List[Dict]:
    items = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            items.append({"path": str(p), "text": text})
    return items


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def main():
    docs = read_text_files(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No .txt/.md files found in {DOCS_DIR.resolve()}")

    # Build chunk list + metadata
    chunks: List[str] = []
    meta: List[Dict] = []

    for d in docs:
        doc_chunks = chunk_text(d["text"], CHUNK_SIZE, OVERLAP)
        for i, ch in enumerate(doc_chunks):
            chunks.append(ch)
            meta.append({
                "source": d["path"],
                "chunk_id": i,
                "char_len": len(ch),
            })

    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    # Embed
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    # Cosine similarity = inner product on normalized vectors
    embeddings = l2_normalize(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP + normalized => cosine
    index.add(embeddings)

    # Save index + metadata + chunks
    faiss.write_index(index, str(OUT_DIR / "docs.index"))
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print("Saved:")
    print(" - index_store/docs.index")
    print(" - index_store/meta.json")
    print(" - index_store/chunks.json")


if __name__ == "__main__":
    main()
