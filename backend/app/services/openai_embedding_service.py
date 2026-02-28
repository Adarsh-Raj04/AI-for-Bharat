"""
OpenAI Embedding Service - Uses OpenAI's text-embedding models
"""
from openai import OpenAI
from typing import List
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    """
    OpenAI embedding service
    Drop-in replacement for sentence-transformers
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string
        
        Args:
            query: The query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        
        Args:
            documents: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # OpenAI allows batch embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=documents
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        # text-embedding-3-small: 1536 dimensions
        # text-embedding-3-large: 3072 dimensions
        # text-embedding-ada-002: 1536 dimensions
        if "large" in self.model:
            return 3072
        return 1536


# Singleton instance
_openai_embedding_service = None


def get_openai_embedding_service() -> OpenAIEmbeddingService:
    """Get or create the OpenAI embedding service singleton"""
    global _openai_embedding_service
    if _openai_embedding_service is None:
        _openai_embedding_service = OpenAIEmbeddingService()
    return _openai_embedding_service
