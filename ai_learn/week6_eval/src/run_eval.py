from __future__ import annotations
import argparse
import os
import pandas as pd

from .common import load_yaml, ensure_dir, now_ts, read_jsonl, write_json
from .classifier_eval import evaluate_classifier, save_classifier_plots
from .rag_eval import evaluate_rag
from .report import make_markdown_report


def _parse_thr_grid(cfg: dict):
    g = cfg["threshold_grid"]
    return float(g["start"]), float(g["stop"]), float(g["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--classifier_before", default=None, help="CSV with columns id,y_true,y_prob")
    ap.add_argument("--classifier_after", default=None, help="CSV with columns id,y_true,y_prob")
    ap.add_argument("--rag_before", default=None, help="JSONL rag eval file")
    ap.add_argument("--rag_after", default=None, help="JSONL rag eval file")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = cfg.get("outputs_dir", "outputs")
    run_id = now_ts()
    out_dir = os.path.join(out_root, run_id)
    ensure_dir(out_dir)

    summary = {"run_id": run_id, "out_dir": out_dir}
    assets = {"classifier_before": {}, "classifier_after": {}}

    # ---- Classifier
    c_cfg = cfg["classifier"]
    thr_grid = _parse_thr_grid(c_cfg)

    c_before_metrics = None
    c_after_metrics = None

    if args.classifier_before:
        dfb = pd.read_csv(args.classifier_before)
        rb = evaluate_classifier(
            dfb,
            label_col=c_cfg["label_col"],
            prob_col=c_cfg["prob_col"],
            tune_objective=c_cfg.get("tune_objective", "f1"),
            threshold_grid=thr_grid,
            calibration_bins=int(c_cfg.get("calibration_bins", 10)),
        )
        c_before_metrics = dict(rb.metrics)
        c_before_metrics["best_threshold"] = rb.threshold
        c_before_metrics.update({f"conf_{k}": v for k, v in rb.confusion.items()})

        # save tables
        rb.threshold_curve.to_csv(os.path.join(out_dir, "classifier_before_threshold_curve.csv"), index=False)
        rb.calibration.to_csv(os.path.join(out_dir, "classifier_before_calibration.csv"), index=False)
        rb.roc.to_csv(os.path.join(out_dir, "classifier_before_roc.csv"), index=False)

        assets["classifier_before"] = save_classifier_plots(rb, out_dir, "before")

    if args.classifier_after:
        dfa = pd.read_csv(args.classifier_after)
        ra = evaluate_classifier(
            dfa,
            label_col=c_cfg["label_col"],
            prob_col=c_cfg["prob_col"],
            tune_objective=c_cfg.get("tune_objective", "f1"),
            threshold_grid=thr_grid,
            calibration_bins=int(c_cfg.get("calibration_bins", 10)),
        )
        c_after_metrics = dict(ra.metrics)
        c_after_metrics["best_threshold"] = ra.threshold
        c_after_metrics.update({f"conf_{k}": v for k, v in ra.confusion.items()})

        ra.threshold_curve.to_csv(os.path.join(out_dir, "classifier_after_threshold_curve.csv"), index=False)
        ra.calibration.to_csv(os.path.join(out_dir, "classifier_after_calibration.csv"), index=False)
        ra.roc.to_csv(os.path.join(out_dir, "classifier_after_roc.csv"), index=False)

        assets["classifier_after"] = save_classifier_plots(ra, out_dir, "after")

    # ---- RAG
    r_cfg = cfg["rag"]
    rag_before_metrics = None
    rag_after_metrics = None

    if args.rag_before:
        rows_b = read_jsonl(args.rag_before)
        rb = evaluate_rag(
            rows_b,
            ks=list(map(int, r_cfg.get("ks", [1, 3, 5, 10]))),
            answer_metrics=r_cfg.get("answer_metrics", ["exact_match", "token_f1"]),
            judge_mode=r_cfg.get("judge_mode", "overlap"),
            overlap_cfg=r_cfg.get("overlap", {}),
        )
        rag_before_metrics = dict(rb.aggregated)
        rb.retrieval.to_csv(os.path.join(out_dir, "rag_before_retrieval.csv"), index=False)
        rb.answers.to_csv(os.path.join(out_dir, "rag_before_answers.csv"), index=False)

    if args.rag_after:
        rows_a = read_jsonl(args.rag_after)
        ra = evaluate_rag(
            rows_a,
            ks=list(map(int, r_cfg.get("ks", [1, 3, 5, 10]))),
            answer_metrics=r_cfg.get("answer_metrics", ["exact_match", "token_f1"]),
            judge_mode=r_cfg.get("judge_mode", "overlap"),
            overlap_cfg=r_cfg.get("overlap", {}),
        )
        rag_after_metrics = dict(ra.aggregated)
        ra.retrieval.to_csv(os.path.join(out_dir, "rag_after_retrieval.csv"), index=False)
        ra.answers.to_csv(os.path.join(out_dir, "rag_after_answers.csv"), index=False)

    # ---- Report
    report_path = os.path.join(out_dir, "report.md")
    make_markdown_report(
        out_path=report_path,
        title=f"{cfg.get('project_name','Eval')} — Before vs After",
        classifier_before=c_before_metrics,
        classifier_after=c_after_metrics,
        rag_before=rag_before_metrics,
        rag_after=rag_after_metrics,
        assets=assets,
    )

    summary.update({
        "classifier_before": c_before_metrics,
        "classifier_after": c_after_metrics,
        "rag_before": rag_before_metrics,
        "rag_after": rag_after_metrics,
        "report_path": report_path,
        "assets": assets,
    })
    write_json(os.path.join(out_dir, "summary.json"), summary)

    print(f"\nSaved outputs to: {out_dir}")
    print(f"Report: {report_path}\n")


if __name__ == "__main__":
    main()
