"""
Reranker Service - Re-rank retrieved documents by relevance
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        """Initialize reranker - using simple scoring for hackathon"""
        pass
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents by relevance to query
        
        For hackathon: Simple keyword-based boosting + vector score
        Production: Use cross-encoder model for better reranking
        
        Args:
            query: The search query
            documents: List of documents with scores
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        try:
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            
            # Calculate relevance scores
            for doc in documents:
                text_lower = doc.get("text", "").lower()
                title_lower = doc.get("title", "").lower()
                
                # Base score from vector similarity
                base_score = doc.get("score", 0.0)
                
                # Keyword matching boost
                keyword_boost = 0.0
                for term in query_terms:
                    if term in title_lower:
                        keyword_boost += 0.1  # Title match is important
                    if term in text_lower:
                        keyword_boost += 0.05  # Content match
                
                # Source type boost (prefer certain sources)
                source_boost = 0.0
                source_type = doc.get("source_type", "")
                if source_type == "pubmed":
                    source_boost = 0.05
                elif source_type == "clinical_trial":
                    source_boost = 0.04
                elif source_type == "fda":
                    source_boost = 0.03
                
                # Combined relevance score
                doc["relevance_score"] = min(1.0, base_score + keyword_boost + source_boost)
            
            # Sort by relevance score
            reranked = sorted(
                documents,
                key=lambda x: x.get("relevance_score", 0.0),
                reverse=True
            )
            
            # Return top k
            result = reranked[:top_k]
            logger.info(f"Reranked {len(documents)} documents, returning top {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original documents on error
            return documents[:top_k]


# Singleton instance
_reranker_service = None


def get_reranker_service() -> RerankerService:
    """Get or create the reranker service singleton"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
