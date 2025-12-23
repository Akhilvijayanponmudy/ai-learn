from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conint, confloat

# ---------- /health ----------
class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str

# ---------- /classify ----------
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    # optional threshold for label decision
    threshold: confloat(ge=0.0, le=1.0) = 0.5

class ClassifyResponse(BaseModel):
    label: Literal["positive", "negative"]
    score: confloat(ge=0.0, le=1.0)
    threshold: confloat(ge=0.0, le=1.0)

# ---------- /search ----------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: conint(ge=1, le=20) = 5

class SearchHit(BaseModel):
    doc_id: str
    score: confloat(ge=0.0, le=1.0)
    snippet: str

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHit]

# ---------- /rag ----------
class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: conint(ge=1, le=20) = 5
    # "simple" keeps it deterministic and offline
    mode: Literal["simple"] = "simple"

class RagResponse(BaseModel):
    query: str
    answer: str
    contexts: List[str]
    retrieved: List[SearchHit]
