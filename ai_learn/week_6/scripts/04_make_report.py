import json
from pathlib import Path

OUT = Path("data/generated/report.md")

CLS = Path("data/generated/classifier_metrics.json")
RET = Path("data/generated/retrieval_metrics.json")
ANS = Path("data/generated/rag_answer_metrics.json")

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def main():
    cls = load_json(CLS)
    ret = load_json(RET)
    ans = load_json(ANS)

    lines = []
    lines.append("# Week 6 Evaluation Report\n")

    lines.append("## Classifier\n")
    if cls:
        lines.append(f"- Train size: **{cls['n_train']}**, Test size: **{cls['n_test']}**")
        lines.append(f"- Macro F1: **{cls['macro_f1']:.4f}**")
        lines.append(f"- ROC-AUC (OvR, macro): **{cls['roc_auc_ovr_macro']:.4f}**")
        lines.append(f"- ECE (calibration): **{cls['ece']:.4f}**\n")
        lines.append("### Threshold tuning (OvR best-F1)\n")
        for k, v in cls["thresholds_ovr_best_f1"].items():
            lines.append(f"- {k}: threshold={v:.2f}, best_f1={cls['thresholds_ovr_best_f1_scores'][k]:.3f}")
    else:
        lines.append("- (No classifier metrics found.)")

    lines.append("\n## RAG Retrieval\n")
    if ret:
        lines.append(f"- Eval set size: **{ret['n']}**\n")
        for k in ret["top_ks"]:
            r = ret["results"][f"@{k}"]["recall"]
            m = ret["results"][f"@{k}"]["mrr"]
            lines.append(f"- Recall@{k}: **{r:.3f}**, MRR@{k}: **{m:.3f}**")
    else:
        lines.append("- (No retrieval metrics found.)")

    lines.append("\n## RAG Answer Heuristics\n")
    if ans:
        lines.append(f"- Answer samples evaluated: **{ans['n']}**")
        lines.append(f"- Citation coverage avg: **{ans['citation_coverage_avg']:.3f}**")
        lines.append(f"- Context overlap avg: **{ans['context_overlap_avg']:.3f}**")
        lines.append("\nNotes:")
        for n in ans.get("notes", []):
            lines.append(f"- {n}")
    else:
        lines.append("- (No answer metrics found.)")

    lines.append("\n## Before/After template\n")
    lines.append("Fill this after you improve chunking / prompts / embeddings:\n")
    lines.append("| Area | Before | After | What changed |\n|---|---:|---:|---|\n| Classifier Macro F1 |  |  |  |\n| Retrieval Recall@5 |  |  |  |\n| Answer citation coverage |  |  |  |\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {OUT}")

if __name__ == "__main__":
    main()
