from fastapi import APIRouter, Request
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.config import settings
from app.core.rate_limit import limiter
from app.api.schemas import (
    HealthResponse,
    ClassifyRequest, ClassifyResponse,
    SearchRequest, SearchResponse, SearchHit,
    RagRequest, RagResponse,
)
from app.services.classifier import classify_text
from app.services.search import search as search_fn
from app.services.rag import rag_answer

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", service=settings.APP_NAME, version="1.0.0")

@router.post("/classify", response_model=ClassifyResponse)
@limiter.limit(settings.RATE_LIMIT_CLASSIFY)
def classify(request: Request, payload: ClassifyRequest):
    score = classify_text(payload.text)
    label = "positive" if score >= payload.threshold else "negative"
    return ClassifyResponse(label=label, score=score, threshold=payload.threshold)

@router.post("/search", response_model=SearchResponse)
@limiter.limit(settings.RATE_LIMIT_SEARCH)
def search(request: Request, payload: SearchRequest):
    top_k = min(payload.top_k, settings.MAX_TOP_K)
    hits = search_fn(payload.query, top_k=top_k)
    return SearchResponse(query=payload.query, top_k=top_k, hits=[SearchHit(**h) for h in hits])

@router.post("/rag", response_model=RagResponse)
@limiter.limit(settings.RATE_LIMIT_RAG)
def rag(request: Request, payload: RagRequest):
    top_k = min(payload.top_k, settings.MAX_TOP_K)
    out = rag_answer(payload.query, top_k=top_k)
    return RagResponse(
        query=payload.query,
        answer=out["answer"],
        contexts=out["contexts"],
        retrieved=[SearchHit(**h) for h in out["hits"]],
    )
