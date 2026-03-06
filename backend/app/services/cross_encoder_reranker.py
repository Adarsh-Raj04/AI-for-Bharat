"""
Cross-Encoder Reranker - Advanced reranking using cross-encoder models
Phase 2 Feature #8
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Advanced reranking using cross-encoder models.
    Falls back to keyword-based reranking if model not available.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Initialized cross-encoder model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback reranking")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load cross-encoder model: {e}, using fallback")
            self.model = None
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder model
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        if not self.model:
            return self._fallback_rerank(query, documents, top_k)
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                text = doc.get('text', '')
                title = doc.get('title', '')
                # Combine title and text for better relevance
                doc_text = f"{title} {text}"[:512]  # Limit length
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc['cross_encoder_score'] = float(score)
                doc['relevance_score'] = float(score)  # Use as main relevance score
            
            # Sort by cross-encoder score
            reranked = sorted(
                documents,
                key=lambda x: x.get('cross_encoder_score', 0.0),
                reverse=True
            )
            
            result = reranked[:top_k]
            logger.info(f"Reranked {len(documents)} documents using cross-encoder, returning top {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking error: {e}")
            return self._fallback_rerank(query, documents, top_k)
    
    def _fallback_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fallback reranking using keyword matching and original scores
        
        Args:
            query: Search query
            documents: List of documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for doc in documents:
            text_lower = doc.get('text', '').lower()
            title_lower = doc.get('title', '').lower()
            
            # Base score from vector similarity
            base_score = doc.get('score', 0.0)
            
            # Keyword matching boost
            keyword_boost = 0.0
            for term in query_terms:
                if term in title_lower:
                    keyword_boost += 0.1
                if term in text_lower:
                    keyword_boost += 0.05
            
            # Source type boost
            source_boost = 0.0
            source_type = doc.get('source_type', '')
            if source_type == 'pubmed':
                source_boost = 0.05
            elif source_type == 'clinical_trial':
                source_boost = 0.04
            
            # Combined relevance score
            doc['relevance_score'] = min(1.0, base_score + keyword_boost + source_boost)
        
        # Sort by relevance score
        reranked = sorted(
            documents,
            key=lambda x: x.get('relevance_score', 0.0),
            reverse=True
        )
        
        result = reranked[:top_k]
        logger.info(f"Reranked {len(documents)} documents using fallback method, returning top {len(result)}")
        return result
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        use_cross_encoder: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents (main interface)
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            use_cross_encoder: Whether to use cross-encoder (if available)
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        if use_cross_encoder and self.model:
            return self.rerank_with_cross_encoder(query, documents, top_k)
        else:
            return self._fallback_rerank(query, documents, top_k)
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Dict[str, Any]]],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-document sets in batch
        
        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            top_k: Number of top documents per query
            
        Returns:
            List of reranked document lists
        """
        results = []
        for query, documents in zip(queries, document_lists):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        
        return results


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Create a cross-encoder reranker instance"""
    return CrossEncoderReranker()
