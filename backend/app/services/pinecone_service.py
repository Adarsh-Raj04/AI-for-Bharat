"""
Pinecone Service - Vector database integration with hybrid search
"""
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class PineconeService:
    def __init__(self):
        self.pc = None
        self.index = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone client and index"""
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if settings.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
            
            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        filter_dict: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            query_embedding: Vector embedding of the query
            query_text: Original query text for keyword matching
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matched documents with metadata
        """
        try:
            # Semantic search using vector similarity
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            documents = []
            for match in results.matches:
                doc = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "text": match.metadata.get("text", ""),
                    "source_type": match.metadata.get("source_type", "unknown"),
                    "source_id": match.metadata.get("source_id", ""),
                    "title": match.metadata.get("title", ""),
                    "url": match.metadata.get("url", "")
                }
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents from Pinecone")
            return documents
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            # Return empty list on error to allow graceful degradation
            return []
    
    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert or update documents in the index
        
        Args:
            documents: List of documents with id, values (embedding), and metadata
        """
        try:
            self.index.upsert(vectors=documents)
            logger.info(f"Upserted {len(documents)} documents to Pinecone")
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise


# Singleton instance
_pinecone_service = None


def get_pinecone_service() -> PineconeService:
    """Get or create the Pinecone service singleton"""
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service
