import json
import random
import re
from pathlib import Path

import pandas as pd

random.seed(42)

DOCS_DIR = Path("data/docs")
OUT_DIR = Path("data/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTENTS = ["bug", "how-to", "billing", "feature_request"]

# ---------- Helpers ----------
def read_docs(docs_dir: Path):
    docs = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".md", ".txt"]:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            docs.append((p.name, txt))
    return docs

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_headings(md: str):
    # capture lines like "# Title" "## Section"
    heads = []
    for line in md.splitlines():
        line = line.strip()
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            title = normalize_ws(m.group(2))
            if title:
                heads.append((level, title))
    return heads

def extract_bullets(md: str):
    bullets = []
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("- "):
            item = normalize_ws(line[2:])
            if item:
                bullets.append(item)
    return bullets

def label_from_question(q: str) -> str:
    ql = q.lower()
    if any(x in ql for x in ["price", "billing", "payment", "invoice", "plan", "quota"]):
        return "billing"
    if any(x in ql for x in ["feature request", "can you add", "support", "i want", "wish", "roadmap"]):
        return "feature_request"
    if any(x in ql for x in ["error", "failed", "fatal", "invalid", "empty export", "permission denied", "timeout"]):
        return "bug"
    return "how-to"

# ---------- 1) Build synthetic classifier dataset ----------
def make_classifier_dataset():
    # Mix of realistic support queries
    howto_templates = [
        "How do I export orders to CSV?",
        "How to import products from a CSV file?",
        "Where do I configure FTP/SFTP settings?",
        "How do I speed up export for large stores?",
        "How can I update existing users during import?",
        "How do I find WooCommerce logs?",
    ]
    bug_templates = [
        "Export is empty even though orders exist.",
        "Import fails with invalid CSV error.",
        "Fatal error during import on large files.",
        "SFTP connection timeout when exporting.",
        "Permission denied while uploading export file via SFTP.",
        "Authentication failed when connecting to SFTP.",
    ]
    billing_templates = [
        "My billing didn’t update after purchase.",
        "Where can I download an invoice?",
        "How do I change my billing email?",
        "I was charged twice—what should I do?",
        "What plan includes scheduled exports?",
        "Why does my quota show exceeded?",
    ]
    feature_templates = [
        "Feature request: export one row per line item.",
        "Can you add a filter by SKU for export?",
        "I want a dashboard for export history.",
        "Please add support for SSH key based SFTP.",
        "Can we export membership data in a single row?",
        "Feature request: webhook on export completion.",
    ]

    rows = []
    def add_items(items, label, n_repeat=20):
        for _ in range(n_repeat):
            t = random.choice(items)
            # Add noise/variation
            variants = [
                t,
                t.replace("How do I", "How can I"),
                t.replace("CSV", "csv"),
                t + " in WooCommerce?",
                "Hi, " + t.lower(),
            ]
            rows.append({"text": random.choice(variants), "label": label})

    add_items(howto_templates, "how-to", 60)
    add_items(bug_templates, "bug", 60)
    add_items(billing_templates, "billing", 60)
    add_items(feature_templates, "feature_request", 60)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    out = OUT_DIR / "classifier.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

# ---------- 2) Build RAG eval set ----------
def make_rag_eval_set(target_n=60):
    docs = read_docs(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No docs found in {DOCS_DIR.resolve()} (.md/.txt). Add docs first.")

    eval_items = []
    idx = 0

    # Auto-generate Q/A from docs headings and bullets (simple but useful)
    for fname, txt in docs:
        heads = extract_headings(txt)
        bullets = extract_bullets(txt)

        # Generate "where/how/what" questions from known topics
        for level, h in heads[:12]:
            q = None
            a = None

            hl = h.lower()
            if "ftp" in hl or "sftp" in hl:
                q = "Where do I configure FTP/SFTP settings?"
                a = "Go to WooCommerce → Import Export → Settings → FTP/SFTP."
            elif "troubleshooting" in hl:
                q = "What are common troubleshooting steps for slow export/import?"
                a = "Try smaller batches, check PHP memory limit and max execution time, and verify FTP/SFTP connectivity."
            elif "filters" in hl and "export" in hl:
                q = "What filters are supported for orders export?"
                a = "Orders export supports filters like date range, order status, payment method, customer email, and optional order total min/max."
            elif "faq" in hl:
                q = "What should I check if export is empty?"
                a = "Check filters like date range/status, verify matching records exist, and try removing filters."

            if q and a:
                eval_items.append({
                    "id": f"rag_{idx:04d}",
                    "question": q,
                    "gold_answer": a,
                    "gold_sources": [fname],  # expected doc(s)
                })
                idx += 1

        # Generate questions from bullets
        for b in bullets[:20]:
            if len(eval_items) >= target_n:
                break
            if "Port" in b or "default 22" in b:
                eval_items.append({
                    "id": f"rag_{idx:04d}",
                    "question": "What is the default port for SFTP?",
                    "gold_answer": "The default port for SFTP is 22.",
                    "gold_sources": [fname],
                })
                idx += 1
            if "Export is empty" in txt:
                eval_items.append({
                    "id": f"rag_{idx:04d}",
                    "question": "Export is empty. What should I do?",
                    "gold_answer": "Check filters like date range/status, verify matching records exist, and try removing filters.",
                    "gold_sources": ["faq.md"],
                })
                idx += 1
                break

        if len(eval_items) >= target_n:
            break

    # Add some fixed eval cases to ensure coverage
    fixed = [
        ("How do I configure SFTP for exports?", "Configure it under WooCommerce → Import Export → Settings → FTP/SFTP.", ["ftp_sftp.md"]),
        ("Import fails with invalid CSV. What should I check?", "Ensure it is comma-separated, UTF-8 encoded, and has a header row.", ["faq.md"]),
        ("How can I speed up export?", "Use smaller date ranges, export in batches, and avoid expensive filters when possible.", ["faq.md", "orders_export.md"]),
        ("What to do if SFTP shows authentication failed?", "Recheck username/password, confirm password login is allowed, and confirm port 22.", ["ftp_sftp.md"]),
        ("Why are line items not in separate rows?", "Some exports generate one row per order; line items may be combined unless line-item mode is enabled.", ["orders_export.md"]),
    ]
    for q, a, srcs in fixed:
        eval_items.append({
            "id": f"rag_{idx:04d}",
            "question": q,
            "gold_answer": a,
            "gold_sources": srcs,
        })
        idx += 1

    # Deduplicate by question
    seen = set()
    uniq = []
    for it in eval_items:
        key = it["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    out = OUT_DIR / "rag_eval.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it in uniq[:max(target_n, len(fixed))]:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Wrote {out} ({len(uniq[:max(target_n, len(fixed))])} items)")

def main():
    make_classifier_dataset()
    make_rag_eval_set(target_n=60)

if __name__ == "__main__":
    main()
