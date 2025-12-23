# Week 6 Eval: Classifier + RAG

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


python -m src.run_eval \
  --config configs/eval.yaml \
  --classifier_before data/classifier_before.csv \
  --classifier_after  data/classifier_after.csv \
  --rag_before data/rag_before.jsonl \
  --rag_after  data/rag_after.jsonl
