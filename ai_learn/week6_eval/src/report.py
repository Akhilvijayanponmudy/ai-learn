from __future__ import annotations
from typing import Dict, Optional
import os
import pandas as pd

from .common import ensure_dir


def _fmt(x: float) -> str:
    if x is None:
        return "NA"
    try:
        if x != x:  # nan
            return "NA"
        return f"{x:.4f}"
    except Exception:
        return str(x)


def make_markdown_report(
    out_path: str,
    title: str,
    classifier_before: Optional[Dict[str, float]],
    classifier_after: Optional[Dict[str, float]],
    rag_before: Optional[Dict[str, float]],
    rag_after: Optional[Dict[str, float]],
    assets: Dict[str, Dict[str, str]],
) -> None:
    ensure_dir(os.path.dirname(out_path))

    def section(name: str, before: Optional[Dict[str, float]], after: Optional[Dict[str, float]]) -> str:
        if not before and not after:
            return f"## {name}\n\nNo data.\n\n"
        keys = sorted(set((before or {}).keys()) | set((after or {}).keys()))
        lines = [f"## {name}\n", "| Metric | Before | After | Δ |", "|---|---:|---:|---:|"]
        for k in keys:
            b = (before or {}).get(k)
            a = (after or {}).get(k)
            delta = None
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and (b == b) and (a == a):
                delta = a - b
            lines.append(f"| `{k}` | {_fmt(b)} | {_fmt(a)} | {_fmt(delta) if delta is not None else 'NA'} |")
        return "\n".join(lines) + "\n\n"

    md = []
    md.append(f"# {title}\n")
    md.append("This report compares **before vs after** on:\n"
              "- Classifier: F1, ROC-AUC, calibration (Brier), threshold tuning\n"
              "- RAG: Recall@k/MRR@k + answer quality + faithfulness/groundedness\n\n")

    md.append(section("Classifier Metrics", classifier_before, classifier_after))
    md.append(section("RAG Metrics", rag_before, rag_after))

    # link plots if present
    md.append("## Plots\n")
    for group, amap in (assets or {}).items():
        if not amap:
            continue
        md.append(f"### {group}\n")
        for name, path in amap.items():
            rel = os.path.relpath(path, os.path.dirname(out_path))
            md.append(f"- {name}: `{rel}`")
        md.append("")
    md.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
