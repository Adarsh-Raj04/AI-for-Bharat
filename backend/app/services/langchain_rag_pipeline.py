"""
LangChain RAG Pipeline - Phase 1 Implementation
Replaces direct API calls with LangChain abstractions for better observability and maintainability
"""

from __future__ import annotations

import logging
import os
import time
from functools import cached_property
from typing import Any, Dict, Iterator, List, Optional

from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.core.config import settings
from app.services.bias_detector import get_bias_detector
from app.services.cache_service import get_cache_service
from app.services.intent_classifier import get_intent_classifier, QueryIntent
from app.services.reranker_service import get_reranker_service
from app.services.safety_guardrails import get_safety_guardrails
from app.services.intent_router import get_intent_router
from app.services.summarization_chain import get_summarization_chain
from app.services.comparison_chain import get_comparison_chain
from app.services.regulatory_chain import get_regulatory_chain
# Phase 2 Enhanced Services
from app.services.langchain_memory import get_conversational_memory
from app.services.hybrid_retriever import get_hybrid_retriever
from app.services.citation_tracker import get_citation_tracker
from app.services.cross_encoder_reranker import get_cross_encoder_reranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    """Serialise retrieved documents into numbered context blocks."""
    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[{i}] {meta.get('source_type', 'unknown').upper()}: {meta.get('source_id', '')}\n"
            f"Title: {meta.get('title', 'Untitled')}\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n".join(parts)


def _docs_to_rerank_payload(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert LangChain Documents to the shape expected by the reranker."""
    return [
        {
            "text": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            # Flatten metadata to top level for reranker
            "source_type": doc.metadata.get("source_type", "unknown"),
            "source_id": doc.metadata.get("source_id", ""),
            "title": doc.metadata.get("title", "Untitled"),
            "url": doc.metadata.get("url", ""),
            "metadata": doc.metadata,  # Keep full metadata too
        }
        for doc in docs
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LangChainRAGPipeline:
    """
    LangChain-based RAG Pipeline with full observability via LangSmith.

    Features:
    - Unified LLM interface (OpenAI, Bedrock)
    - LangSmith tracing for all operations
    - Prompt template versioning
    - LCEL chain composition
    - Error handling and fallbacks
    """

    def __init__(self) -> None:
        self._configure_langsmith()
        self._initialize_services()
        self._initialize_llm()
        self._initialize_vectorstore()
        self._initialize_prompts()
        self._initialize_chains()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _configure_langsmith(self) -> None:
        """Set LangSmith environment variables when tracing is enabled."""
        if not settings.LANGSMITH_TRACING:
            return
        os.environ.update(
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": settings.LANGSMITH_API_KEY,
                "LANGCHAIN_PROJECT": settings.LANGSMITH_PROJECT or "medresearch-prod",
                "LANGCHAIN_ENDPOINT": settings.LANGSMITH_ENDPOINT,
            }
        )
        logger.info("LangSmith tracing enabled")

    def _initialize_services(self) -> None:
        self.cache_service = get_cache_service()
        self.reranker_service = get_reranker_service()
        self.intent_classifier = get_intent_classifier()
        self.bias_detector = get_bias_detector()
        self.safety_guardrails = get_safety_guardrails()
        
        # Phase 2: Intent routing and summarization
        self.intent_router = None  # Initialized after LLM
        self.summarization_chain = None  # Initialized after LLM
        
        # Phase 3: Specialized chains
        self.comparison_chain = None  # Initialized after LLM
        self.regulatory_chain = None  # Initialized after LLM
        
        # Phase 2 Enhanced Services (initialized early, some need LLM later)
        self.memory = None  # Initialized after LLM
        self.citation_tracker = get_citation_tracker()
        self.cross_encoder_reranker = get_cross_encoder_reranker()
        self.hybrid_retriever = None  # Initialized after vectorstore

    def _initialize_llm(self) -> None:
        try:
            if settings.USE_OPENAI:
                self.llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.1,
                    streaming=True,
                )
                logger.info("Initialized OpenAI LLM: %s", settings.OPENAI_MODEL)
            else:
                self.llm = ChatBedrock(
                    model_id=settings.BEDROCK_MODEL_ID,
                    region_name=settings.AWS_REGION,
                    credentials_profile_name=None,
                    model_kwargs={"temperature": 0.1},
                )
                logger.info("Initialized Bedrock LLM: %s", settings.BEDROCK_MODEL_ID)
        except Exception:
            logger.exception("Failed to initialize LLM")
            raise

    def _initialize_vectorstore(self) -> None:
        try:
            if settings.USE_OPENAI:
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                )
            else:
                from langchain_huggingface import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
            self.embeddings = embeddings

            self.vectorstore = PineconeVectorStore(
                index=self.pinecone_index,
                embedding=embeddings,
                text_key="text",
            )
            
            # Phase 2: Initialize hybrid retriever after vectorstore
            from app.services.pinecone_service import get_pinecone_service
            from app.services.embedding_service import get_embedding_service
            pinecone_service = get_pinecone_service()
            embedding_service = get_embedding_service()
            self.hybrid_retriever = get_hybrid_retriever(pinecone_service, embedding_service)
            
            logger.info("Initialized Pinecone vector store with LangChain and hybrid retriever")
        except Exception:
            logger.exception("Failed to initialize vector store")
            raise

    def _initialize_prompts(self) -> None:
        system_base = (
            "You are MedResearch AI, an expert medical research assistant. "
            "Your role is to provide accurate, evidence-based answers to medical "
            "and pharmaceutical research questions.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the question using ONLY the information from the context documents provided\n"
            "2. Cite sources using [number] notation (e.g., [1], [2]) corresponding to the context documents\n"
            "3. If the context doesn't contain enough information, acknowledge this limitation\n"
            "4. Use clear, professional medical language\n"
            "5. Format your response with proper markdown for readability:\n"
            "   - Use ## for headers\n"
            "   - Use - or * for bullet lists\n"
            "   - Use **bold** for emphasis\n"
            "   - For tables, use proper markdown syntax: | Column | Column |\n"
            "                                              |--------|--------|\n"
            "                                              | Data   | Data   |\n"
            "6. Include relevant statistics, findings, or data points from the sources\n"
            "7. Be concise but comprehensive\n\n"
            "CONTEXT DOCUMENTS:\n{context}"
        )

        self.rag_prompt = ChatPromptTemplate.from_messages(
            [("system", system_base), ("human", "{question}")]
        )

        self.conversational_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_base.replace(
                        "2. Cite sources using [number] notation (e.g., [1], [2]) "
                        "corresponding to the context documents",
                        "2. Consider the conversation history for context and "
                        "coreference resolution\n"
                        "3. Cite sources using [number] notation (e.g., [1], [2])",
                    ).replace(
                        "3. If the context",
                        "4. If the context",
                    ).replace(
                        "4. Use clear",
                        "5. Use clear",
                    ).replace(
                        "5. Format",
                        "6. Format",
                    ).replace(
                        "6. Include",
                        "7. Include",  # keep numbering consistent
                    ).replace(
                        "7. Be concise",
                        "8. Be concise",
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        logger.info("Initialized prompt templates")

    def _initialize_chains(self) -> None:
        # Chains are built dynamically in process_query to avoid duplicate retrieval
        # Phase 2: Initialize intent router and summarization chain
        self.intent_router = get_intent_router(self.llm)
        self.summarization_chain = get_summarization_chain(self.llm)
        
        # Phase 3: Initialize specialized chains
        self.comparison_chain = get_comparison_chain(self.llm)
        self.regulatory_chain = get_regulatory_chain(self.llm)
        
        # Phase 2: Initialize conversational memory with LLM
        self.memory = get_conversational_memory(self.llm)
        
        logger.info("Chain initialization deferred to query time, Phase 2 & 3 features initialized (including memory)")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _retrieve_documents(self, query: str, top_k: int = None, use_enhanced: bool = True) -> List[Document]:
        """
        Retrieve documents with Phase 2 enhanced hybrid retrieval and cross-encoder reranking.
        Falls back to legacy method if enhanced retrieval fails.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_enhanced: Whether to use Phase 2 enhanced retrieval
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        try:
            # Generate embedding
            query_embedding = self.embeddings.embed_query(query)
            
            if use_enhanced and self.hybrid_retriever:
                # Phase 2: Enhanced hybrid retrieval with RRF
                results = self.hybrid_retriever.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k * 2,  # Get more for cross-encoder reranking
                    use_rrf=True
                )
                
                if results:
                    # Phase 2: Cross-encoder reranking
                    reranked_results = self.cross_encoder_reranker.rerank(
                        query=query,
                        documents=results,
                        top_k=top_k,
                        use_cross_encoder=True
                    )
                    
                    # Convert to LangChain Documents
                    documents = self.hybrid_retriever.to_langchain_documents(reranked_results)
                    
                    logger.info(f"Retrieved {len(documents)} documents (Phase 2: hybrid + cross-encoder)")
                    return documents
            
            # Fallback to legacy method
            from app.services.pinecone_service import get_pinecone_service
            pinecone_service = get_pinecone_service()
            
            docs_dict = pinecone_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k
            )
            
            # Convert to LangChain Documents
            documents = []
            for doc_dict in docs_dict:
                metadata = {
                    "score": doc_dict.get("score", 0.0),
                    "id": doc_dict.get("id", ""),
                    "source_type": doc_dict.get("source_type", "unknown"),
                    "source_id": doc_dict.get("source_id", ""),
                    "title": doc_dict.get("title", "Untitled"),
                    "url": doc_dict.get("url", ""),
                }
                if "metadata" in doc_dict:
                    metadata.update(doc_dict["metadata"])
                
                doc = Document(
                    page_content=doc_dict.get("text", ""),
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents (legacy fallback)")
            return documents
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}", exc_info=True)
            return []

    @cached_property
    def _model_name(self) -> str:
        if settings.USE_OPENAI:
            return settings.OPENAI_MODEL
        if settings.USE_AWS_BEDROCK:
            return settings.BEDROCK_MODEL_ID
        return settings.CLAUDE_MODEL

    def _get_model_name(self) -> str:
        return self._model_name

    def _format_chat_history(self, chat_history: List[Dict]) -> List:
        """Format chat history with Phase 2 memory management"""
        if not chat_history:
            return []
        
        # Phase 2: Use conversational memory to manage context window
        if self.memory:
            managed_history = self.memory.manage_context_window(
                chat_history,
                max_tokens=4000
            )
            formatted = self.memory.format_messages_for_langchain(managed_history)
            logger.info(f"Formatted {len(formatted)} messages (managed from {len(chat_history)} original)")
            return formatted
        
        # Fallback to simple formatting
        messages = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _build_citations(self, documents: List[Dict[str, Any]], response_text: str = None) -> List[Dict[str, Any]]:
        """Build citation list from reranked documents with Phase 2 enhanced tracking"""
        citations = []
        for i, doc in enumerate(documents, 1):
            # Handle both dict and Document objects
            if isinstance(doc, dict):
                citation = {
                    "number": i,
                    "title": doc.get("title", "Untitled"),
                    "url": doc.get("url", ""),
                    "relevance_score": doc.get("relevance_score", 0.0),
                    "source_type": doc.get("source_type", "unknown"),
                    "source_id": doc.get("source_id", ""),
                }
            else:
                # Document object
                metadata = getattr(doc, 'metadata', {})
                citation = {
                    "number": i,
                    "title": metadata.get("title", "Untitled"),
                    "url": metadata.get("url", ""),
                    "relevance_score": metadata.get("relevance_score", 0.0),
                    "source_type": metadata.get("source_type", "unknown"),
                    "source_id": metadata.get("source_id", ""),
                }
            
            citations.append(citation)
        
        # Phase 2: Enrich citations with usage tracking
        if response_text and self.citation_tracker:
            citations = self.citation_tracker.enrich_citations(citations, response_text)
            logger.debug(f"Enriched {len(citations)} citations with usage tracking")
        
        return citations

    def _calculate_confidence_score(
        self,
        documents: List,
        intent_confidence: float,
        bias_analysis: Dict,
    ) -> float:
        if not documents:
            return 0.0

        top_docs = documents[:3]
        relevance = sum(d.get("relevance_score", 0.0) for d in top_docs) / len(top_docs)
        bias_penalty = bias_analysis.get("bias_score", 0.0) * 0.2
        confidence = relevance * 0.5 + intent_confidence * 0.3 + (1 - bias_penalty) * 0.2
        return max(0.0, min(0.95, confidence))

    def _fallback_response(
        self,
        query: str,  # noqa: ARG002 — kept for potential future use / logging
        error: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = {
            "blocked": "This query cannot be processed due to safety guidelines.",
            "no_documents": "No relevant research documents found for your query.",
        }
        return {
            "text": messages.get(reason, "Unable to process your query at this time."),
            "citations": [],
            "confidence": 0.0,
            "intent": "error",
            "model": self._get_model_name(),
            "error": error,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Singleton
    # ---------------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        use_cache: bool = True,  # noqa: ARG002
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query through the LangChain RAG pipeline.

        Args:
            query: User's question.
            chat_history: Previous conversation messages.
            use_cache: Whether to use cached embeddings (reserved for future use).
            user_id: User identifier for tracing.
            session_id: Session identifier for tracing.

        Returns:
            Response dict with text, citations, and metadata.
        """
        start_time = time.time()
        chat_history = chat_history or []

        try:
            # Safety check
            is_safe, block_reason, _safety_meta = self.safety_guardrails.check_query(query)
            if not is_safe:
                logger.warning("Query blocked: %s", block_reason)
                return self._fallback_response(query, reason="blocked")

            # Intent classification
            intent, intent_confidence, _ = self.intent_classifier.classify(query)
            routing_config = self.intent_classifier.get_routing_config(intent)

            # Retrieve documents (single call, no duplicate traces)
            retrieved_docs = self._retrieve_documents(query, top_k=routing_config.get("top_k", settings.TOP_K_RESULTS))
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return self._fallback_response(query, reason="no_documents")

            # Rerank
            reranked_docs = self.reranker_service.rerank(
                query=query,
                documents=_docs_to_rerank_payload(retrieved_docs),
                top_k=routing_config["rerank_top_k"],
            )

            # Bias analysis
            bias_analysis = self.bias_detector.analyze_sources(reranked_docs, query)

            # Phase 2 & 3: Route to specialized chains based on intent
            formatted_context = _format_docs(retrieved_docs)
            
            if intent == QueryIntent.SUMMARIZATION:
                # Summarization chain
                pmid = self.summarization_chain.extract_pmid_from_query(query)
                if pmid:
                    response_text = self.summarization_chain.summarize_by_pmid(pmid, reranked_docs)
                elif len(reranked_docs) == 1:
                    response_text = self.summarization_chain.summarize_single_document(reranked_docs[0])
                else:
                    response_text = self.summarization_chain.summarize_multiple_documents(reranked_docs)
                    
            elif intent == QueryIntent.COMPARISON:
                # Phase 3: Comparison chain
                response_text = self.comparison_chain.compare(query, formatted_context)
                
            elif intent == QueryIntent.REGULATORY_COMPLIANCE:
                # Phase 3: Regulatory chain
                response_text = self.regulatory_chain.get_regulatory_info(query, formatted_context)
                
            else:
                # Use intent-routed chain for other intents
                if chat_history:
                    formatted_history = self._format_chat_history(chat_history)
                    # For conversational queries, use conversational prompt
                    chain = (
                        {
                            "context": lambda _: formatted_context,
                            "chat_history": lambda _: formatted_history,
                            "question": RunnablePassthrough(),
                        }
                        | self.conversational_prompt
                        | self.llm
                        | StrOutputParser()
                    )
                else:
                    # Use intent-specific chain
                    chain = self.intent_router.build_routed_chain(intent, formatted_context)
                
                # Generate response
                response_text = chain.invoke(query)

            # Phase 2: Build citations with enhanced tracking
            citations = self._build_citations(reranked_docs, response_text)
            confidence = self._calculate_confidence_score(
                reranked_docs, intent_confidence, bias_analysis
            )
            
            # Phase 2: Generate citation report
            citation_report = None
            if self.citation_tracker:
                citation_report = self.citation_tracker.generate_citation_report(
                    response_text, citations
                )
                logger.info(f"Citation accuracy: {citation_report['validation']['accuracy']:.2%}")

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info("Query processed successfully in %dms", elapsed_ms)

            result = {
                "text": response_text,
                "citations": citations,
                "confidence": confidence,
                "intent": intent.value,
                "model": self._get_model_name(),
                "processing_time_ms": elapsed_ms,
                "bias_analysis": bias_analysis,
            }
            
            # Add citation report if available
            if citation_report:
                result["citation_report"] = citation_report
            
            return result

        except Exception:
            logger.exception("RAG pipeline error")
            return self._fallback_response(query, error="Internal pipeline error")

    def process_query_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        use_cache: bool = True,  # noqa: ARG002
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Process a query with a streaming response.

        Yields dicts with keys ``type`` (``"citations"``, ``"text"``, ``"error"``)
        and ``data``.
        """
        chat_history = chat_history or []

        try:
            # Safety check
            is_safe, _block_reason, _safety_meta = self.safety_guardrails.check_query(query)
            if not is_safe:
                yield {"type": "error", "data": "Query blocked by safety guardrails"}
                return

            # Intent classification
            intent, _intent_confidence, _ = self.intent_classifier.classify(query)

            # Retrieve documents (single call)
            retrieved_docs = self._retrieve_documents(query, top_k=settings.TOP_K_RESULTS)
            
            if not retrieved_docs:
                yield {"type": "error", "data": "No relevant documents found"}
                return

            # Rerank
            reranked_docs = self.reranker_service.rerank(
                query=query,
                documents=_docs_to_rerank_payload(retrieved_docs),
                top_k=5,
            )

            # Send citations first
            yield {"type": "citations", "data": self._build_citations(reranked_docs)}

            # Build and stream the appropriate chain
            if chat_history:
                formatted_history = self._format_chat_history(chat_history)
                chain = (
                    {
                        "context": lambda _: _format_docs(retrieved_docs),
                        "chat_history": lambda _: formatted_history,
                        "question": RunnablePassthrough(),
                    }
                    | self.conversational_prompt
                    | self.llm
                )
            else:
                chain = (
                    {
                        "context": lambda _: _format_docs(retrieved_docs),
                        "question": RunnablePassthrough(),
                    }
                    | self.rag_prompt
                    | self.llm
                )

            for chunk in chain.stream(query):
                # Handle different chunk types from LangChain
                if hasattr(chunk, 'content'):
                    # AIMessageChunk with content attribute
                    if chunk.content:
                        yield {"type": "text", "data": chunk.content}
                elif isinstance(chunk, str):
                    # String chunk
                    yield {"type": "text", "data": chunk}
                else:
                    # Other chunk types - try to extract content
                    content = str(chunk) if chunk else ""
                    if content:
                        yield {"type": "text", "data": content}

        except Exception:
            logger.exception("Streaming error")
            yield {"type": "error", "data": "Internal streaming error"}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_langchain_rag_pipeline: Optional[LangChainRAGPipeline] = None


_langchain_rag_pipeline = None

def get_langchain_rag_pipeline():
    global _langchain_rag_pipeline

    if _langchain_rag_pipeline is None:
        _langchain_rag_pipeline = LangChainRAGPipeline()

    return _langchain_rag_pipeline