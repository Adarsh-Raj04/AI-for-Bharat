"""
RAG Pipeline - Main orchestration of retrieval and generation
"""

from typing import Dict, Any, List, Iterator, Optional
import logging
import time

from app.services.cache_service import get_cache_service
from app.services.pinecone_service import get_pinecone_service
from app.services.reranker_service import get_reranker_service
from app.services.intent_classifier import get_intent_classifier
from app.services.bias_detector import get_bias_detector
from app.services.safety_guardrails import get_safety_guardrails
from langsmith import traceable
from app.core.config import settings


logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self):

        # Embeddings
        if settings.USE_OPENAI:
            from app.services.openai_embedding_service import (
                get_openai_embedding_service,
            )

            self.embedding_service = get_openai_embedding_service()
        else:
            from app.services.embedding_service import get_embedding_service

            self.embedding_service = get_embedding_service()

        # LLM
        if settings.USE_OPENAI:
            from app.services.openai_service import get_openai_service

            self.claude_service = get_openai_service()
        elif settings.USE_AWS_BEDROCK:
            from app.services.bedrock_service import get_bedrock_service

            self.claude_service = get_bedrock_service()

        self.cache_service = get_cache_service()
        self.pinecone_service = get_pinecone_service()
        self.reranker_service = get_reranker_service()

        self.intent_classifier = get_intent_classifier()
        self.bias_detector = get_bias_detector()
        self.safety_guardrails = get_safety_guardrails()

    @traceable(name="safety_check", run_type="chain")
    def _run_safety(self, query: str):
        logger.info("Running safety check")
        return self.safety_guardrails.check_query(query)

    @traceable(name="intent_classification", run_type="chain")
    def _run_intent(self, query: str):
        logger.info("Classifying intent")
        return self.intent_classifier.classify(query)

    @traceable(name="embedding", run_type="embedding")
    def _run_embedding(self, query: str, use_cache: bool):
        logger.info("Generating embedding")
        return self._get_or_generate_embedding(query, use_cache)

    @traceable(name="retrieval", run_type="retriever")
    def _run_retrieval(self, query, query_embedding, top_k):
        logger.info(f"Retrieving docs top_k={top_k}")
        docs = self.pinecone_service.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
        )
        logger.info(f"Retrieved {len(docs)} documents")
        return docs

    @traceable(name="rerank", run_type="parser")
    def _run_rerank(self, query, docs, top_k):
        logger.info(f"Reranking {len(docs)} docs")
        return self.reranker_service.rerank(
            query=query,
            documents=docs,
            top_k=top_k,
        )

    @traceable(name="bias_analysis", run_type="tool")
    def _run_bias(self, docs, query):
        logger.info("Running bias analysis")
        return self.bias_detector.analyze_sources(docs, query)

    # --------------------------------------------------------
    # MAIN QUERY PIPELINE
    # --------------------------------------------------------

    @traceable(name="rag_pipeline_process_query")
    def process_query(
        self,
        query: str,
        use_cache: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start_time = time.time()
        run_id = None

        try:

            # Safety
            is_safe, block_reason, safety_metadata = self._run_safety(query)

            if not is_safe:
                return self._fallback_response(query, reason="blocked")

            # Intent
            intent, intent_confidence, _ = self._run_intent(query)
            routing_config = self.intent_classifier.get_routing_config(intent)

            # Cache
            if use_cache:
                cached = self.cache_service.get_query_results(query)
                if cached:
                    return cached

            # Embedding
            query_embedding = self._run_embedding(query, use_cache)

            # Retrieval
            retrieved_docs = self._run_retrieval(
                query,
                query_embedding,
                routing_config["top_k"],
            )

            if not retrieved_docs:
                return self._fallback_response(query, reason="no_documents")

            # Rerank
            reranked_docs = self._run_rerank(
                query,
                retrieved_docs,
                routing_config["rerank_top_k"],
            )

            # Bias
            bias_analysis = self._run_bias(reranked_docs, query)

            # Generation
            response = self.claude_service.generate_response(
                query=query,
                context_documents=reranked_docs,
                max_tokens=routing_config["max_tokens"],
            )

            citations = self._build_citations(reranked_docs)

            confidence = self._calculate_confidence_score(
                reranked_docs, intent_confidence, bias_analysis
            )

            result = {
                "text": response["text"],
                "citations": citations,
                "confidence": confidence,
                "intent": intent.value,
                "tokens_used": response["tokens_used"],
                "model": response["model"],
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "bias_analysis": bias_analysis,
            }

            return result

        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            return self._fallback_response(query, error=str(e))

    # --------------------------------------------------------
    # STREAMING
    # --------------------------------------------------------

    @traceable(name="rag_pipeline_process_query_stream")
    def process_query_stream(
        self, query: str, use_cache: bool = True
    ) -> Iterator[Dict[str, Any]]:

        try:
            query_embedding = self._get_or_generate_embedding(query, use_cache)

            docs = self.pinecone_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=settings.TOP_K_RESULTS,
            )

            if not docs:
                yield {"type": "error", "data": "No relevant documents"}
                return

            reranked = self.reranker_service.rerank(
                query=query, documents=docs, top_k=5
            )

            yield {"type": "citations", "data": self._build_citations(reranked)}

            for chunk in self.claude_service.generate_response_stream(
                query=query, context_documents=reranked
            ):
                yield {"type": "text", "data": chunk}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "data": str(e)}

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _get_or_generate_embedding(self, query: str, use_cache: bool) -> List[float]:

        if use_cache:
            cached = self.cache_service.get_query_embedding(query)
            if cached:
                return cached

        embedding = self.embedding_service.embed_query(query)

        if use_cache:
            self.cache_service.set_query_embedding(query, embedding)

        return embedding

    def _build_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "number": i,
                "title": doc.get("title", "Untitled"),
                "url": doc.get("url", ""),
                "relevance_score": doc.get("relevance_score", 0.0),
            }
            for i, doc in enumerate(documents, 1)
        ]

    def _calculate_confidence_score(
        self, documents, intent_confidence, bias_analysis
    ) -> float:

        if not documents:
            return 0.0

        relevance = sum(d.get("relevance_score", 0.0) for d in documents[:3]) / min(
            3, len(documents)
        )

        bias_penalty = bias_analysis.get("bias_score", 0.0) * 0.2

        confidence = (
            relevance * 0.5 + intent_confidence * 0.3 + (1 - bias_penalty) * 0.2
        )

        return max(0.0, min(0.95, confidence))

    def _fallback_response(self, query: str, error: str = None, reason: str = None):
        return {
            "text": "Unable to process query right now.",
            "citations": [],
            "confidence": 0.0,
            "intent": "error",
            "error": error,
            "reason": reason,
        }


# Singleton
_rag_pipeline = None


def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
