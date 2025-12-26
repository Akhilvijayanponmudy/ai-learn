from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    text: str

class Document(BaseModel):
    id: str
    content: str
    metadata: dict = {}

class QueryResponse(BaseModel):
    query_id: str
    category: str
    answer: str
    sources: List[Document]

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int # 1-5
    comment: Optional[str] = None
