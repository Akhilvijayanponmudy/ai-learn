from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio as fuzzy_ratio

from .common import normalize_text, tokenize, safe_div
from .judges import overlap_judge, LLMJudge, JudgeResult


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_set = {}
    for t in p:
        p_set[t] = p_set.get(t, 0) + 1
    g_set = {}
    for t in g:
        g_set[t] = g_set.get(t, 0) + 1
    common = 0
    for t, c in p_set.items():
        common += min(c, g_set.get(t, 0))
    prec = common / max(1, len(p))
    rec = common / max(1, len(g))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mrr_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold or [])
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold or [])
    if not gold_set:
        return float("nan")
    hit = any(doc_id in gold_set for doc_id in retrieved[:k])
    return 1.0 if hit else 0.0


@dataclass
class RAGResults:
    retrieval: pd.DataFrame
    answers: pd.DataFrame
    aggregated: Dict[str, float]


def evaluate_rag(rows: List[dict], ks: List[int], answer_metrics: List[str],
                 judge_mode: str, overlap_cfg: dict) -> RAGResults:
    # Setup judge
    llm = None
    if judge_mode == "llm":
        llm = LLMJudge()

    retr_rows = []
    ans_rows = []

    for r in rows:
        qid = r.get("id")
        gold_docs = r.get("gold_doc_ids") or []
        retrieved_items = r.get("retrieved") or []
        retrieved_doc_ids = [it.get("doc_id") if isinstance(it, dict) else str(it) for it in retrieved_items]

        # retrieval metrics
        recs = {"id": qid}
        for k in ks:
            recs[f"recall@{k}"] = recall_at_k(retrieved_doc_ids, gold_docs, k)
            recs[f"mrr@{k}"] = mrr_at_k(retrieved_doc_ids, gold_docs, k)
        retr_rows.append(recs)

        # answer metrics
        pred = r.get("pred_answer", "") or ""
        gold = r.get("gold_answer", "") or ""
        contexts = r.get("contexts") or []

        a = {"id": qid}
        if "exact_match" in answer_metrics:
            a["exact_match"] = exact_match(pred, gold)
        if "token_f1" in answer_metrics:
            a["token_f1"] = token_f1(pred, gold)
        if "fuzzy_ratio" in answer_metrics:
            a["fuzzy_ratio"] = float(fuzzy_ratio(normalize_text(pred), normalize_text(gold))) / 100.0

        # faithfulness/groundedness
        if judge_mode == "llm" and llm is not None:
            jr = llm.judge(pred, contexts)
        else:
            jr = overlap_judge(pred, contexts,
                               min_n=int(overlap_cfg.get("min_ngram", 3)),
                               max_n=int(overlap_cfg.get("max_ngram", 5)))
        a["faithfulness"] = jr.faithfulness
        a["groundedness"] = jr.groundedness
        a["judge_notes"] = jr.notes

        ans_rows.append(a)

    retr_df = pd.DataFrame(retr_rows)
    ans_df = pd.DataFrame(ans_rows)

    # aggregate
    agg = {}
    for col in retr_df.columns:
        if col == "id":
            continue
        agg[col] = float(np.nanmean(retr_df[col].to_numpy(dtype=float)))
    for col in ans_df.columns:
        if col in ("id", "judge_notes"):
            continue
        agg[col] = float(np.nanmean(ans_df[col].to_numpy(dtype=float)))

    return RAGResults(retrieval=retr_df, answers=ans_df, aggregated=agg)
