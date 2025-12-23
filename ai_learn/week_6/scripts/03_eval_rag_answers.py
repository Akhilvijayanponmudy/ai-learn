import json
import re
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

EVAL = Path("data/generated/rag_eval.jsonl")
ANSWERS = Path("data/generated/rag_answers.jsonl")   # you can create later from your rag.py output
OUT = Path("data/generated/rag_answer_metrics.json")

def split_sentences(text: str) -> List[str]:
    # simple split
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def has_citation(sent: str) -> bool:
    return bool(re.search(r"\[C\d+\]", sent))

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def load_jsonl(p: Path) -> List[Dict]:
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def baseline_answers_from_eval(eval_items: List[Dict]) -> List[Dict]:
    # creates placeholder answers so the pipeline runs without LLM
    out = []
    for it in eval_items:
        out.append({
            "id": it["id"],
            "question": it["question"],
            "answer": "I don't know from the provided docs.",
            "retrieved": []
        })
    return out

def main():
    eval_items = load_jsonl(EVAL)

    if ANSWERS.exists():
        answers = {a["id"]: a for a in load_jsonl(ANSWERS)}
    else:
        # run with baseline placeholders
        answers = {a["id"]: a for a in baseline_answers_from_eval(eval_items)}

    covs = []
    overlaps = []

    for it in tqdm(eval_items, desc="Answer eval"):
        a = answers.get(it["id"])
        if not a:
            continue

        answer = a.get("answer", "")
        sents = split_sentences(answer)
        if not sents:
            continue

        citation_cov = sum(1 for s in sents if has_citation(s)) / len(sents)
        covs.append(citation_cov)

        # Build context string from retrieved chunks if present
        ctx = ""
        retrieved = a.get("retrieved", [])
        if isinstance(retrieved, list) and retrieved:
            ctx = "\n".join([r.get("text", "") for r in retrieved if isinstance(r, dict)])

        overlap = jaccard(tokenize(answer), tokenize(ctx)) if ctx else 0.0
        overlaps.append(overlap)

    metrics = {
        "n": len(covs),
        "citation_coverage_avg": sum(covs) / len(covs) if covs else 0.0,
        "context_overlap_avg": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "notes": [
            "citation_coverage is a groundedness proxy: sentences containing [C#].",
            "context_overlap is a weak faithfulness proxy: token overlap(answer, retrieved_context).",
            "For real evaluation, generate answers with your RAG bot and write rag_answers.jsonl."
        ]
    }

    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {OUT}")
    print(f"citation_coverage_avg={metrics['citation_coverage_avg']:.3f}")
    print(f"context_overlap_avg={metrics['context_overlap_avg']:.3f}")

if __name__ == "__main__":
    main()
