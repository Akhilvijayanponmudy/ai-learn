from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import json
import re
import urllib.request

from .common import normalize_text


@dataclass
class JudgeResult:
    faithfulness: float   # answer supported by context
    groundedness: float   # answer stays within context (penalize hallucinated entities/claims)
    notes: str = ""


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]


def overlap_judge(answer: str, contexts: List[str], min_n: int = 3, max_n: int = 5) -> JudgeResult:
    a = normalize_text(answer)
    ctx = normalize_text(" ".join(contexts or []))
    a_toks = [t for t in a.split() if t]
    c_toks = [t for t in ctx.split() if t]

    if not a_toks or not c_toks:
        return JudgeResult(faithfulness=0.0, groundedness=0.0, notes="empty answer or context")

    ctx_set = set(c_toks)
    token_support = sum(1 for t in a_toks if t in ctx_set)
    token_cov = token_support / max(1, len(a_toks))

    # n-gram support: proportion of answer ngrams found in context ngrams
    ctx_ngr = set()
    for n in range(min_n, max_n + 1):
        ctx_ngr.update(_ngrams(c_toks, n))

    a_ngr_total = 0
    a_ngr_hit = 0
    for n in range(min_n, max_n + 1):
        a_ngr = _ngrams(a_toks, n)
        a_ngr_total += len(a_ngr)
        a_ngr_hit += sum(1 for g in a_ngr if g in ctx_ngr)

    ngram_cov = (a_ngr_hit / max(1, a_ngr_total))

    # Faithfulness: more weight to n-grams (claim-level), fallback to tokens
    faith = 0.7 * ngram_cov + 0.3 * token_cov

    # Groundedness: penalize “new” tokens not in context (very rough)
    new_tokens = sum(1 for t in a_toks if t not in ctx_set)
    novelty = new_tokens / max(1, len(a_toks))
    grounded = max(0.0, 1.0 - novelty)

    notes = f"token_cov={token_cov:.3f}, ngram_cov={ngram_cov:.3f}, novelty={novelty:.3f}"
    return JudgeResult(faithfulness=float(faith), groundedness=float(grounded), notes=notes)


class LLMJudge:
    """
    OpenAI-compatible judge via HTTP.
    Env vars:
      - OPENAI_BASE_URL (e.g. https://api.openai.com/v1)
      - OPENAI_API_KEY
      - OPENAI_MODEL (e.g. gpt-4.1-mini)
    """
    def __init__(self):
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def judge(self, answer: str, contexts: List[str]) -> JudgeResult:
        prompt = {
            "role": "user",
            "content": (
                "You are an evaluator for RAG answers.\n"
                "Given CONTEXT and ANSWER, score:\n"
                "1) faithfulness: how well ANSWER is supported by CONTEXT (0..1)\n"
                "2) groundedness: does ANSWER avoid adding unsupported claims (0..1)\n"
                "Return STRICT JSON: {\"faithfulness\": float, \"groundedness\": float, \"notes\": string}\n\n"
                f"CONTEXT:\n{chr(10).join(contexts or [])}\n\n"
                f"ANSWER:\n{answer}\n"
            )
        }

        body = json.dumps({
            "model": self.model,
            "messages": [prompt],
            "temperature": 0
        }).encode("utf-8")

        req = urllib.request.Request(
            url=f"{self.base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")

        data = json.loads(raw)
        text = data["choices"][0]["message"]["content"].strip()

        # extract JSON robustly
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return JudgeResult(0.0, 0.0, notes=f"could_not_parse: {text[:200]}")
        obj = json.loads(m.group(0))
        return JudgeResult(
            faithfulness=float(obj.get("faithfulness", 0.0)),
            groundedness=float(obj.get("groundedness", 0.0)),
            notes=str(obj.get("notes", "")),
        )
