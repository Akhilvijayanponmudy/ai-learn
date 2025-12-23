from typing import List, Dict
from .search import search, fetch_contexts

def rag_answer(query: str, top_k: int) -> Dict:
    hits = search(query, top_k=top_k)
    contexts = fetch_contexts(hits)

    # Very simple generation: stitch top contexts.
    # Replace with LLM call later.
    if not contexts:
        answer = "I couldn't find relevant information in the knowledge base."
    else:
        answer = f"Based on the available docs: {contexts[0]}"

    return {"answer": answer, "hits": hits, "contexts": contexts}
