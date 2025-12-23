# week6_eval — Before vs After

This report compares **before vs after** on:
- Classifier: F1, ROC-AUC, calibration (Brier), threshold tuning
- RAG: Recall@k/MRR@k + answer quality + faithfulness/groundedness


## Classifier Metrics

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| `best_threshold` | 0.3300 | 0.3900 | 0.0600 |
| `brier_score` | 0.2166 | 0.0512 | -0.1654 |
| `conf_fn` | 0.0000 | 0.0000 | 0.0000 |
| `conf_fp` | 1.0000 | 0.0000 | -1.0000 |
| `conf_tn` | 2.0000 | 3.0000 | 1.0000 |
| `conf_tp` | 3.0000 | 3.0000 | 0.0000 |
| `f1_at_best_threshold` | 0.8571 | 1.0000 | 0.1429 |
| `precision_at_best_threshold` | 0.7500 | 1.0000 | 0.2500 |
| `recall_at_best_threshold` | 1.0000 | 1.0000 | 0.0000 |
| `roc_auc` | 0.6667 | 1.0000 | 0.3333 |


## RAG Metrics

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| `exact_match` | 0.0000 | 0.5000 | 0.5000 |
| `faithfulness` | 0.3337 | 0.4203 | 0.0866 |
| `fuzzy_ratio` | 0.4863 | 0.9138 | 0.4275 |
| `groundedness` | 0.6458 | 0.6062 | -0.0396 |
| `mrr@1` | 0.5000 | 1.0000 | 0.5000 |
| `mrr@10` | 0.7500 | 1.0000 | 0.2500 |
| `mrr@3` | 0.7500 | 1.0000 | 0.2500 |
| `mrr@5` | 0.7500 | 1.0000 | 0.2500 |
| `recall@1` | 0.5000 | 1.0000 | 0.5000 |
| `recall@10` | 1.0000 | 1.0000 | 0.0000 |
| `recall@3` | 1.0000 | 1.0000 | 0.0000 |
| `recall@5` | 1.0000 | 1.0000 | 0.0000 |
| `token_f1` | 0.1678 | 0.9074 | 0.7396 |


## Plots

### classifier_before

- roc: `roc_before.png`
- calibration: `calibration_before.png`
- threshold_f1: `threshold_f1_before.png`

### classifier_after

- roc: `roc_after.png`
- calibration: `calibration_after.png`
- threshold_f1: `threshold_f1_after.png`


