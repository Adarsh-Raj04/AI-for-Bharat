"""
Hybrid Retriever - Enhanced hybrid search with better ranking
Phase 2 Feature #4 (Simplified - uses existing Pinecone)
"""

from typing import List, Dict, Any
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Enhanced hybrid retriever combining semantic and keyword search.
    Uses Reciprocal Rank Fusion (RRF) for better result merging.
    """
    
    def __init__(self, pinecone_service, embedding_service):
        self.pinecone_service = pinecone_service
        self.embedding_service = embedding_service
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
        self.rrf_k = 60  # RRF constant
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion
        
        RRF formula: score = sum(1 / (k + rank))
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            k: RRF constant (default 60)
            
        Returns:
            Merged and reranked results
        """
        # Build score dictionary
        scores = {}
        
        # Add semantic scores
        for rank, doc in enumerate(semantic_results, 1):
            doc_id = doc.get('id', doc.get('source_id', str(rank)))
            scores[doc_id] = {
                'doc': doc,
                'score': 1 / (k + rank),
                'semantic_rank': rank,
                'keyword_rank': None
            }
        
        # Add keyword scores
        for rank, doc in enumerate(keyword_results, 1):
            doc_id = doc.get('id', doc.get('source_id', str(rank)))
            if doc_id in scores:
                scores[doc_id]['score'] += 1 / (k + rank)
                scores[doc_id]['keyword_rank'] = rank
            else:
                scores[doc_id] = {
                    'doc': doc,
                    'score': 1 / (k + rank),
                    'semantic_rank': None,
                    'keyword_rank': rank
                }
        
        # Sort by RRF score
        sorted_docs = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Return documents with updated scores
        results = []
        for item in sorted_docs:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['score']
            doc['semantic_rank'] = item['semantic_rank']
            doc['keyword_rank'] = item['keyword_rank']
            results.append(doc)
        
        return results
    
    def keyword_boost(
        self,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Boost documents based on keyword matching
        
        Args:
            documents: List of documents
            query: Query string
            
        Returns:
            Documents with keyword boost scores
        """
        query_terms = set(query.lower().split())
        
        for doc in documents:
            text = doc.get('text', '').lower()
            title = doc.get('title', '').lower()
            
            # Count keyword matches
            title_matches = sum(1 for term in query_terms if term in title)
            text_matches = sum(1 for term in query_terms if term in text)
            
            # Calculate boost
            keyword_score = (title_matches * 0.3) + (text_matches * 0.1)
            doc['keyword_score'] = keyword_score
            
            # Boost original score
            original_score = doc.get('score', 0.0)
            doc['boosted_score'] = original_score + keyword_score
        
        return documents
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        use_rrf: bool = True
    ) -> List[Dict]:
        """
        Perform enhanced hybrid search
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            use_rrf: Whether to use Reciprocal Rank Fusion
            
        Returns:
            Hybrid search results
        """
        try:
            # Get semantic results (more results for RRF)
            semantic_results = self.pinecone_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k * 2  # Get more for better fusion
            )
            
            if not semantic_results:
                return []
            
            # Apply keyword boosting
            keyword_boosted = self.keyword_boost(semantic_results, query)
            
            if use_rrf:
                # Simulate keyword-only results by sorting by keyword score
                keyword_results = sorted(
                    keyword_boosted,
                    key=lambda x: x.get('keyword_score', 0.0),
                    reverse=True
                )[:top_k * 2]
                
                # Apply RRF
                fused_results = self.reciprocal_rank_fusion(
                    semantic_results,
                    keyword_results,
                    k=self.rrf_k
                )
                
                # Return top k
                results = fused_results[:top_k]
                logger.info(f"Hybrid search with RRF returned {len(results)} results")
                return results
            else:
                # Simple weighted combination
                for doc in keyword_boosted:
                    semantic_score = doc.get('score', 0.0)
                    keyword_score = doc.get('keyword_score', 0.0)
                    doc['hybrid_score'] = (
                        semantic_score * self.semantic_weight +
                        keyword_score * self.keyword_weight
                    )
                
                # Sort by hybrid score
                sorted_results = sorted(
                    keyword_boosted,
                    key=lambda x: x.get('hybrid_score', 0.0),
                    reverse=True
                )
                
                results = sorted_results[:top_k]
                logger.info(f"Hybrid search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to semantic only
            return semantic_results[:top_k] if semantic_results else []
    
    def to_langchain_documents(
        self,
        results: List[Dict]
    ) -> List[Document]:
        """
        Convert results to LangChain Document objects
        
        Args:
            results: Search results
            
        Returns:
            List of LangChain Documents
        """
        documents = []
        for result in results:
            metadata = {
                'score': result.get('score', 0.0),
                'hybrid_score': result.get('hybrid_score', result.get('rrf_score', 0.0)),
                'source_type': result.get('source_type', 'unknown'),
                'source_id': result.get('source_id', ''),
                'title': result.get('title', 'Untitled'),
                'url': result.get('url', ''),
            }
            
            doc = Document(
                page_content=result.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)
        
        return documents


def get_hybrid_retriever(pinecone_service, embedding_service) -> HybridRetriever:
    """Create a hybrid retriever instance"""
    return HybridRetriever(pinecone_service, embedding_service)
