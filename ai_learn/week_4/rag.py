import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env vars from .env file, overriding any existing shell variables
load_dotenv(override=True)

# ---------- Config ----------
MODEL_NAME = "all-MiniLM-L6-v2"
STORE = Path("index_store")
TOP_K = 6

# OpenAI (set env var: OPENAI_API_KEY)
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o-mini"  # good quality/cost
# ---------------------------


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def load_store():
    index = faiss.read_index(str(STORE / "docs.index"))
    meta = json.loads((STORE / "meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((STORE / "chunks.json").read_text(encoding="utf-8"))
    return index, meta, chunks


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    index, meta, chunks = load_store()
    emb_model = SentenceTransformer(MODEL_NAME)

    q = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q).astype("float32")

    scores, idxs = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "source": meta[idx]["source"],
            "chunk_id": meta[idx]["chunk_id"],
            "text": chunks[idx]
        })
    return results


def build_context(chunks: List[Dict]) -> str:
    # keep context structured + citeable
    parts = []
    for i, c in enumerate(chunks, start=1):
        tag = f"[C{i} | {Path(c['source']).name}#chunk{c['chunk_id']}]"
        snippet = c["text"].strip()
        parts.append(f"{tag}\n{snippet}")
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, context: str):
    system = (
        "You are a documentation support assistant. "
        "You must answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know and ask for what doc/section is missing. "
        "Do not use outside knowledge. "
        "Always include citations for every answer section using the chunk tags like [C2]. "
        "If multiple chunks support a point, cite multiple tags."
    )

    user = f"""Question:
{question}

Context:
{context}

Instructions:
- Answer concisely.
- Use bullet points when helpful.
- Add citations like [C1], [C2] next to the sentences they support.
- If not answerable from context, refuse with: "I don't know from the provided docs." and suggest what to check."""
    return system, user


def call_openai(system: str, user: str) -> str:
    # OpenAI Python SDK v1.x style
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def main():
    question = input("Ask a question: ").strip()
    chunks = retrieve(question, k=TOP_K)

    if not chunks:
        print("\nI don't know from the provided docs. (No chunks retrieved.)")
        return

    context = build_context(chunks)
    system, user = build_messages(question, context)

    if USE_OPENAI:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY in your environment.")
        answer = call_openai(system, user)
        print("\n--- Answer ---\n")
        print(answer)
        print("\n--- Retrieved Chunks ---\n")
        for i, c in enumerate(chunks, start=1):
            print(f"[C{i}] score={c['score']:.4f} source={c['source']} chunk_id={c['chunk_id']}")
    else:
        print("\nContext built, but USE_OPENAI=False. Plug in your local LLM call here.")


if __name__ == "__main__":
    main()
