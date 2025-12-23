from __future__ import annotations
import os
import json
import yaml
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    # simple whitespace tokenizer; replace with spaCy later if you want
    return [t for t in s.split(" ") if t]
