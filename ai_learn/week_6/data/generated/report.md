# Week 6 Evaluation Report

## Classifier

- Train size: **180**, Test size: **60**
- Macro F1: **1.0000**
- ROC-AUC (OvR, macro): **1.0000**
- ECE (calibration): **0.2365**

### Threshold tuning (OvR best-F1)

- billing: threshold=0.15, best_f1=1.000
- bug: threshold=0.15, best_f1=1.000
- feature_request: threshold=0.15, best_f1=1.000
- how-to: threshold=0.20, best_f1=1.000

## RAG Retrieval

- Eval set size: **11**

- Recall@1: **0.909**, MRR@1: **0.909**
- Recall@3: **1.000**, MRR@3: **0.955**
- Recall@5: **1.000**, MRR@5: **0.955**
- Recall@10: **1.000**, MRR@10: **0.955**

## RAG Answer Heuristics

- Answer samples evaluated: **11**
- Citation coverage avg: **0.614**
- Context overlap avg: **0.284**

Notes:
- citation_coverage is a groundedness proxy: sentences containing [C#].
- context_overlap is a weak faithfulness proxy: token overlap(answer, retrieved_context).
- For real evaluation, generate answers with your RAG bot and write rag_answers.jsonl.

## Before/After template

Fill this after you improve chunking / prompts / embeddings:

| Area | Before | After | What changed |
|---|---:|---:|---|
| Classifier Macro F1 |  |  |  |
| Retrieval Recall@5 |  |  |  |
| Answer citation coverage |  |  |  |
