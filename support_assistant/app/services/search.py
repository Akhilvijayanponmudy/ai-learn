from typing import List
from app.models import Document
import logging

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        # Mock knowledge base
        self.knowledge_base = [
            Document(id="1", content="To reset your password, go to Settings > Security > Reset Password.", metadata={"topic": "password"}),
            Document(id="2", content="Billing cycles are monthly starting from the day you signed up.", metadata={"topic": "billing"}),
            Document(id="3", content="If you see Error 500, please contact support with the request ID.", metadata={"topic": "errors"}),
        ]
    
    def search(self, query: str, top_k: int = 2) -> List[Document]:
        # Dumb keyword search for demonstration
        results = []
        words = query.lower().split()
        for doc in self.knowledge_base:
            if any(word in doc.content.lower() for word in words):
                results.append(doc)
        
        # Fallback if no match
        if not results:
            results.append(self.knowledge_base[0])
            
        return results[:top_k]

    def generate_answer(self, query: str, context: List[Document]) -> str:
        # Mock LLM generation
        context_str = "\n".join([d.content for d in context])
        logger.info(f"Generating answer for query: {query} with context length: {len(context_str)}")
        return f"Based on the docs, here is the info: {context[0].content}"
