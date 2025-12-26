from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import uuid
import time
import logging
from contextvars import ContextVar

from app.logging_config import setup_logging, request_id_ctx_var
from app.models import QueryRequest, QueryResponse, FeedbackRequest
from app.services.classifier import Classifier
from app.services.search import SearchService

# Setup logging immediately
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Support Assistant API", version="1.0.0")

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# Initialize Services
classifier = Classifier()
search_service = SearchService()

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request_id_ctx_var.set(request_id)
    
    start_time = time.time()
    
    logger.info(f"Incoming request: {request.method} {request.url.path}", extra={
        "method": request.method,
        "path": request.url.path,
        "request_id": request_id
    })
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    
    logger.info(f"Request completed", extra={
        "status_code": response.status_code,
        "duration": process_time,
        "request_id": request_id
    })
    
    return response

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    # 1. Classify
    category = classifier.predict(request.text)
    logger.info(f"Classified query: {request.text[:20]}... as {category}", extra={"category": category})

    # 2. Search
    docs = search_service.search(request.text)
    
    # 3. Generate Answer (Simulated RAG)
    answer = search_service.generate_answer(request.text, docs)
    
    return QueryResponse(
        query_id=request_id_ctx_var.get(),
        category=category,
        answer=answer,
        sources=docs
    )

@app.post("/api/v1/feedback")
async def feedback_endpoint(feedback: FeedbackRequest):
    logger.info("Received feedback", extra={
        "query_id": feedback.query_id, 
        "rating": feedback.rating, 
        "comment": feedback.comment
    })
    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
