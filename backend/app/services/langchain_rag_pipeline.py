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
from app.services.timeline_service import build_research_timeline
from app.services.diversity_filter import apply_diversity_filter

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
            "source_type": doc.metadata.get("source_type", "unknown"),
            "source_id": doc.metadata.get("source_id", ""),
            "title": doc.metadata.get("title", "Untitled"),
            "url": doc.metadata.get("url", ""),
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class LangChainRAGPipeline:
    """
    LangChain-based RAG Pipeline with full observability via LangSmith.
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
        self.intent_router = None
        self.summarization_chain = None
        self.comparison_chain = None
        self.regulatory_chain = None
        self.memory = None
        self.citation_tracker = get_citation_tracker()
        self.cross_encoder_reranker = get_cross_encoder_reranker()
        self.hybrid_retriever = None

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

            from app.services.pinecone_service import get_pinecone_service
            from app.services.embedding_service import get_embedding_service

            pinecone_service = get_pinecone_service()
            embedding_service = get_embedding_service()
            self.hybrid_retriever = get_hybrid_retriever(
                pinecone_service, embedding_service
            )

            logger.info(
                "Initialized Pinecone vector store with LangChain and hybrid retriever"
            )
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
            "8. If the document is not relevant to medical research, state 'This document is not relevant to medical research.'\n\n"
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
                    )
                    .replace("3. If the context", "4. If the context")
                    .replace("4. Use clear", "5. Use clear")
                    .replace("5. Format", "6. Format")
                    .replace("6. Include", "7. Include")
                    .replace("7. Be concise", "8. Be concise"),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        logger.info("Initialized prompt templates")

    def _initialize_chains(self) -> None:
        self.intent_router = get_intent_router(self.llm)
        self.summarization_chain = get_summarization_chain(self.llm)
        self.comparison_chain = get_comparison_chain(self.llm)
        self.regulatory_chain = get_regulatory_chain(self.llm)
        self.memory = get_conversational_memory(self.llm)
        logger.info(
            "Chain initialization deferred to query time, Phase 2 & 3 features initialized"
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _fetch_by_pmid(self, pmid: str) -> list:
        """Fetch all chunks for a specific PMID via Pinecone metadata filter."""
        try:
            print("Fetching by PMID from Pinecone metadata filter:", pmid)
            dummy_vector = [0.0] * 1536

            results = self.pinecone_index.query(
                vector=dummy_vector,
                top_k=10,
                filter={
                    "$or": [
                        {"pmid": {"$eq": pmid}},
                        {"source_id": {"$eq": f"PMID:{pmid}"}},
                    ]
                },
                include_metadata=True,
            )

            if not results.matches:
                logger.warning("PMID:%s — no matches in Pinecone metadata filter", pmid)
                return []

            docs = []
            for match in results.matches:
                meta = match.metadata or {}
                docs.append(
                    {
                        "text": meta.get("text", ""),
                        "score": match.score,
                        "relevance_score": match.score,
                        "source_type": meta.get("source_type", "pubmed"),
                        "source_id": meta.get("source_id", f"PMID:{pmid}"),
                        "pmid": meta.get("pmid", pmid),
                        "title": meta.get("title", "Untitled"),
                        "url": meta.get(
                            "url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        ),
                        "publication_date": meta.get("publication_date", ""),
                        "journal": meta.get("journal", ""),
                        "authors": meta.get("authors", ""),
                        "metadata": meta,
                    }
                )

            logger.info(
                "PMID:%s — fetched %d chunks via metadata filter", pmid, len(docs)
            )
            return docs

        except Exception as e:
            logger.error("PMID metadata fetch failed for %s: %s", pmid, e)
            return []

    def _retrieve_documents(
        self,
        query: str,
        top_k: int = None,
        use_enhanced: bool = True,
        source_filter: str = None,
    ) -> List[Document]:
        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        filter_dict = {"source_id": {"$eq": source_filter}} if source_filter else None

        try:
            query_embedding = self.embeddings.embed_query(query)

            if use_enhanced and self.hybrid_retriever:
                results = self.hybrid_retriever.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k * 2,
                    use_rrf=True,
                    filter_dict=filter_dict,
                )

                if results:
                    reranked_results = self.cross_encoder_reranker.rerank(
                        query=query,
                        documents=results,
                        top_k=top_k,
                        use_cross_encoder=True,
                    )
                    documents = self.hybrid_retriever.to_langchain_documents(
                        reranked_results
                    )
                    logger.info(
                        "Retrieved %d documents (hybrid + cross-encoder, filter=%s)",
                        len(documents),
                        source_filter,
                    )
                    return documents

            # Fallback legacy path
            from app.services.pinecone_service import get_pinecone_service

            pinecone_service = get_pinecone_service()

            docs_dict = pinecone_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            documents = []
            for doc_dict in docs_dict:
                metadata = {
                    "score": doc_dict.get("score", 0.0),
                    "id": doc_dict.get("id", ""),
                    "source_type": doc_dict.get("source_type", "unknown"),
                    "source_id": doc_dict.get("source_id", ""),
                    "title": doc_dict.get("title", "Untitled"),
                    "url": doc_dict.get("url", ""),
                    "publication_date": doc_dict.get("publication_date", ""),
                }
                if "metadata" in doc_dict:
                    metadata.update(doc_dict["metadata"])
                doc = Document(page_content=doc_dict.get("text", ""), metadata=metadata)
                documents.append(doc)

            logger.info(
                "Retrieved %d documents (legacy fallback, filter=%s)",
                len(documents),
                source_filter,
            )
            return documents

        except Exception as e:
            logger.error("Document retrieval failed: %s", e, exc_info=True)
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
        """Format chat history with Phase 2 memory management."""
        if not chat_history:
            return []

        if self.memory:
            managed_history = self.memory.manage_context_window(
                chat_history, max_tokens=4000
            )
            formatted = self.memory.format_messages_for_langchain(managed_history)
            logger.info(
                "Formatted %d messages (managed from %d original)",
                len(formatted),
                len(chat_history),
            )
            return formatted

        # Fallback simple formatting
        messages = []
        for msg in chat_history:
            role = (
                msg.get("role")
                if isinstance(msg, dict)
                else (msg[0] if isinstance(msg, (list, tuple)) else None)
            )
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else (msg[1] if isinstance(msg, (list, tuple)) else "")
            )
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))
        return messages

    def _chat_history_to_lc_messages(self, chat_history) -> List:
        """
        Convert chat_history in ANY format to proper LangChain message objects.
        Handles: list of dicts, list of tuples, list of LangChain messages.
        This prevents MESSAGE_COERCION_FAILURE when streaming.
        """
        messages = []
        for item in chat_history or []:
            # Already a LangChain message object
            if isinstance(item, (HumanMessage, AIMessage)):
                messages.append(item)
                continue

            # Tuple or list: ("user"/"human"/"assistant", "content")
            if isinstance(item, (list, tuple)) and len(item) == 2:
                role, content = item[0], item[1]
            # Dict: {"role": "user", "content": "..."}
            elif isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                continue

            if role in ("user", "human"):
                messages.append(HumanMessage(content=str(content)))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=str(content)))
            # skip system/unknown

        return messages

    def _build_citations(
        self, documents: List[Dict[str, Any]], response_text: str = None
    ) -> List[Dict[str, Any]]:
        citations = []
        seen_source_ids = set()

        for doc in documents:
            if isinstance(doc, dict):
                source_id = doc.get("source_id", "")
                title = doc.get("title", "Untitled")
                url = doc.get("url", "")
                relevance = doc.get("relevance_score", 0.0)
                source_type = doc.get("source_type", "unknown")
                pub_date = doc.get("publication_date") or None
            else:
                metadata = getattr(doc, "metadata", {})
                source_id = metadata.get("source_id", "")
                title = metadata.get("title", "Untitled")
                url = metadata.get("url", "")
                relevance = getattr(doc, "relevance_score", None) or metadata.get(
                    "relevance_score", 0.0
                )
                source_type = metadata.get("source_type", "unknown")
                pub_date = metadata.get("publication_date") or None

            dedup_key = source_id or title
            if dedup_key and dedup_key in seen_source_ids:
                continue
            seen_source_ids.add(dedup_key)

            citations.append(
                {
                    "number": len(citations) + 1,
                    "title": title,
                    "url": url,
                    "relevance_score": relevance,
                    "source_type": source_type,
                    "source_id": source_id,
                    "publication_date": pub_date,
                    "times_cited": 0,
                    "is_cited": False,
                }
            )

        if response_text and self.citation_tracker:
            citations = self.citation_tracker.enrich_citations(citations, response_text)
            logger.debug("Enriched %d citations with usage tracking", len(citations))

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
        confidence = (
            relevance * 0.5 + intent_confidence * 0.3 + (1 - bias_penalty) * 0.2
        )
        return max(0.0, min(0.95, confidence))

    def _fallback_response(
        self, query: str, error: Optional[str] = None, reason: Optional[str] = None
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
    # Compare branch — fixed MESSAGE_COERCION_FAILURE
    # ------------------------------------------------------------------

    def _compare_branch(
        self,
        query,
        compare_filters,
        chat_history,
        user_id,
        session_id,
        compare_titles=None,
    ):
        """
        Retrieve from two source_ids independently, then stream a structured
        comparison from the LLM using real document titles.
        """
        sid_a, sid_b = compare_filters[0], compare_filters[1]

        docs_a = self._retrieve_documents(query=query, source_filter=sid_a)
        docs_b = self._retrieve_documents(query=query, source_filter=sid_b)

        # Resolve human-readable names — prefer passed titles, fallback to metadata
        def _doc_title(docs, passed_title, sid):
            if passed_title and passed_title.strip():
                return passed_title.strip()
            for d in docs:
                meta = d.metadata if hasattr(d, "metadata") else d.get("metadata", {})
                t = meta.get("title", "")
                if t and t != "Untitled":
                    return t
            return sid  # last resort: show source_id

        passed_titles = compare_titles or []
        title_a = _doc_title(
            docs_a, passed_titles[0] if len(passed_titles) > 0 else None, sid_a
        )
        title_b = _doc_title(
            docs_b, passed_titles[1] if len(passed_titles) > 1 else None, sid_b
        )

        def _extract_text(docs, max_chars=3000):
            texts = []
            for d in docs:
                if hasattr(d, "page_content"):
                    texts.append(d.page_content)
                elif isinstance(d, dict):
                    texts.append(d.get("page_content") or d.get("text", ""))
            return "\n\n".join(texts)[:max_chars]

        def _make_citations(docs, number_start):
            seen, out = set(), []
            for d in docs:
                meta = d.metadata if hasattr(d, "metadata") else d.get("metadata", {})
                sid = meta.get("source_id", "")
                if sid in seen:
                    continue
                seen.add(sid)
                out.append(
                    {
                        "number": number_start + len(out),
                        "title": meta.get("title", sid),
                        "url": meta.get("url", ""),
                        "relevance_score": meta.get("score", 1.0),
                        "source_type": meta.get("source_type", "document"),
                        "source_id": sid,
                    }
                )
            return out

        citations = _make_citations(docs_a, 1) + _make_citations(
            docs_b, len(docs_a) + 1
        )
        yield {"type": "citations", "data": citations}

        context_a = _extract_text(docs_a)
        context_b = _extract_text(docs_b)

        compare_prompt = f"""You are a medical research assistant comparing two research documents.

**{title_a}** — relevant content:
{context_a or "(no relevant content found)"}

**{title_b}** — relevant content:
{context_b or "(no relevant content found)"}

User question: {query}

Provide a structured comparison using the actual document names above (not "Document A/B"):
1. **Key Findings** — what each document says about the topic
2. **Agreements** — where they align
3. **Differences** — methodology, conclusions, patient populations, etc.
4. **Clinical Relevance** — which findings are more applicable and why

Always refer to documents by their actual names: **{title_a}** and **{title_b}**."""

        # ── THE FIX: convert history to proper LangChain message objects ──
        # Old code did: for role, content in chat_history → dict → llm.stream()
        # That caused: MESSAGE_COERCION_FAILURE (unexpected message type 'role')
        lc_messages = self._chat_history_to_lc_messages(chat_history)
        lc_messages.append(HumanMessage(content=compare_prompt))

        try:
            for chunk in self.llm.stream(lc_messages):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text:
                    yield {"type": "text", "data": text}
        except Exception as e:
            logger.exception("Compare streaming error")
            yield {"type": "error", "data": f"Compare failed: {e}"}
            return

        yield {"type": "metadata", "data": {"confidence": 0.9, "intent": "comparison"}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_query(
        self,
        query,
        chat_history=None,
        use_cache=True,
        user_id=None,
        session_id=None,
        source_filter=None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        chat_history = chat_history or []

        try:
            is_safe, block_reason, _safety_meta = self.safety_guardrails.check_query(
                query
            )
            if not is_safe:
                logger.warning("Query blocked: %s", block_reason)
                return self._fallback_response(query, reason="blocked")

            intent, intent_confidence, _ = self.intent_classifier.classify(query)
            routing_config = self.intent_classifier.get_routing_config(intent)

            retrieved_docs = self._retrieve_documents(
                query,
                top_k=routing_config.get("top_k", settings.TOP_K_RESULTS),
                source_filter=source_filter,
            )

            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return self._fallback_response(query, reason="no_documents")

            reranked_docs = self.reranker_service.rerank(
                query=query,
                documents=_docs_to_rerank_payload(retrieved_docs),
                top_k=routing_config["rerank_top_k"],
            )
            reranked_docs = apply_diversity_filter(
                documents=reranked_docs,
                max_chunks_per_source=2,
                target_k=routing_config["rerank_top_k"],
            )

            bias_analysis = self.bias_detector.analyze_sources(reranked_docs, query)
            formatted_context = _format_docs(retrieved_docs)

            if intent == QueryIntent.SUMMARIZATION:
                pmid = self.summarization_chain.extract_pmid_from_query(query)
                if pmid:
                    pmid_docs = self._fetch_by_pmid(pmid)
                    if pmid_docs:
                        response_text = self.summarization_chain.summarize_by_pmid(
                            pmid, pmid_docs
                        )
                    else:
                        response_text = self.summarization_chain.summarize_by_pmid(
                            pmid, reranked_docs
                        )
                elif len(reranked_docs) == 1:
                    response_text = self.summarization_chain.summarize_single_document(
                        reranked_docs[0]
                    )
                else:
                    response_text = (
                        self.summarization_chain.summarize_multiple_documents(
                            reranked_docs
                        )
                    )

            elif intent == QueryIntent.COMPARISON:
                response_text = self.comparison_chain.compare(query, formatted_context)

            elif intent == QueryIntent.REGULATORY_COMPLIANCE:
                response_text = self.regulatory_chain.get_regulatory_info(
                    query, formatted_context
                )

            else:
                if chat_history:
                    formatted_history = self._format_chat_history(chat_history)
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
                    chain = self.intent_router.build_routed_chain(
                        intent, formatted_context
                    )
                response_text = chain.invoke(query)

            citations = self._build_citations(reranked_docs, response_text)
            confidence = self._calculate_confidence_score(
                reranked_docs, intent_confidence, bias_analysis
            )

            citation_report = None
            if self.citation_tracker:
                citation_report = self.citation_tracker.generate_citation_report(
                    response_text, citations
                )
                logger.info(
                    "Citation accuracy: %.2f%%",
                    citation_report["validation"]["accuracy"] * 100,
                )

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info("Query processed successfully in %dms", elapsed_ms)

            timeline = build_research_timeline(citations)

            result = {
                "text": response_text,
                "citations": citations,
                "confidence": confidence,
                "intent": intent.value,
                "model": self._get_model_name(),
                "processing_time_ms": elapsed_ms,
                "bias_analysis": bias_analysis,
                "timeline": timeline,
            }

            if citation_report:
                result["citation_report"] = citation_report

            return result

        except Exception:
            logger.exception("RAG pipeline error")
            return self._fallback_response(query, error="Internal pipeline error")

    def process_query_stream(
        self,
        query: str,
        chat_history=None,
        use_cache: bool = True,
        user_id=None,
        session_id=None,
        source_filter=None,
        compare_filters=None,
        compare_titles=None,  # ← human-readable titles for each compare_filter
    ):
        import time

        start_time = time.time()
        chat_history = chat_history or []

        try:
            # ── Compare mode — short-circuits everything else ──────────────
            if compare_filters and len(compare_filters) == 2:
                yield from self._compare_branch(
                    query,
                    compare_filters,
                    chat_history,
                    user_id,
                    session_id,
                    compare_titles=compare_titles,
                )
                return

            # ── Safety check ───────────────────────────────────────────────
            is_safe, _block_reason, _safety_meta = self.safety_guardrails.check_query(
                query
            )
            if not is_safe:
                yield {"type": "error", "data": "Query blocked by safety guardrails"}
                return

            # ── Intent classification ──────────────────────────────────────
            intent, intent_confidence, _ = self.intent_classifier.classify(query)
            routing_config = self.intent_classifier.get_routing_config(intent)
            logger.info(
                "STREAM INTENT: %s (confidence %.2f) | query: '%s'",
                intent,
                intent_confidence,
                query[:80],
            )

            # ── Retrieve ───────────────────────────────────────────────────
            retrieved_docs = self._retrieve_documents(
                query,
                top_k=routing_config.get("top_k", settings.TOP_K_RESULTS),
                source_filter=source_filter,
            )
            if not retrieved_docs:
                yield {"type": "error", "data": "No relevant documents found"}
                return

            # ── Rerank + diversity filter ──────────────────────────────────
            reranked_docs = self.reranker_service.rerank(
                query=query,
                documents=_docs_to_rerank_payload(retrieved_docs),
                top_k=routing_config["rerank_top_k"],
            )
            reranked_docs = apply_diversity_filter(
                documents=reranked_docs,
                max_chunks_per_source=2,
                target_k=routing_config["rerank_top_k"],
            )

            bias_analysis = self.bias_detector.analyze_sources(reranked_docs, query)
            citations = self._build_citations(reranked_docs)
            confidence = self._calculate_confidence_score(
                reranked_docs, intent_confidence, bias_analysis
            )
            formatted_context = _format_docs(retrieved_docs)

            # ── Intent routing ─────────────────────────────────────────────
            if intent == QueryIntent.SUMMARIZATION:
                pmid = self.summarization_chain.extract_pmid_from_query(query)
                if pmid:
                    print("Routing to PMID-specific summarization for PMID:", pmid)
                    pmid_docs = self._fetch_by_pmid(pmid)
                    if pmid_docs:
                        logger.info(
                            "PMID fetch returned %d docs for %s", len(pmid_docs), pmid
                        )
                        citations = self._build_citations(pmid_docs)
                        yield {"type": "citations", "data": citations}
                        response_text = self.summarization_chain.summarize_by_pmid(
                            pmid, pmid_docs
                        )
                    else:
                        logger.warning(
                            "PMID fetch empty, falling back to reranked docs"
                        )
                        yield {"type": "citations", "data": citations}
                        response_text = self.summarization_chain.summarize_by_pmid(
                            pmid, reranked_docs
                        )
                elif len(reranked_docs) == 1:
                    print(
                        "Single document retrieved, using single-document summarization"
                    )
                    response_text = self.summarization_chain.summarize_single_document(
                        reranked_docs[0]
                    )
                else:
                    print(
                        "Multiple documents retrieved, using multi-document summarization"
                    )
                    yield {"type": "citations", "data": citations}
                    response_text = (
                        self.summarization_chain.summarize_multiple_documents(
                            reranked_docs
                        )
                    )
                yield {"type": "text", "data": response_text}

            elif intent == QueryIntent.COMPARISON:
                print("Comparison query received")
                yield {"type": "citations", "data": citations}
                response_text = self.comparison_chain.compare(query, formatted_context)
                yield {"type": "text", "data": response_text}

            elif intent == QueryIntent.REGULATORY_COMPLIANCE:
                print("Regulatory compliance query received")
                yield {"type": "citations", "data": citations}
                response_text = self.regulatory_chain.get_regulatory_info(
                    query, formatted_context
                )
                yield {"type": "text", "data": response_text}

            else:
                print("General QA query received")
                # General QA — stream token by token
                yield {"type": "citations", "data": citations}

                if chat_history:
                    formatted_history = self._format_chat_history(chat_history)
                    chain = (
                        {
                            "context": lambda _: formatted_context,
                            "chat_history": lambda _: formatted_history,
                            "question": RunnablePassthrough(),
                        }
                        | self.conversational_prompt
                        | self.llm
                    )
                else:
                    prompt = self.intent_router.get_prompt_for_intent(intent)
                    chain = (
                        {
                            "context": lambda _: formatted_context,
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | self.llm
                    )

                for chunk in chain.stream(query):
                    if hasattr(chunk, "content"):
                        if chunk.content:
                            yield {"type": "text", "data": chunk.content}
                    elif isinstance(chunk, str):
                        yield {"type": "text", "data": chunk}
                    else:
                        content = str(chunk) if chunk else ""
                        if content:
                            yield {"type": "text", "data": content}

            # ── Final metadata ─────────────────────────────────────────────
            elapsed_ms = int((time.time() - start_time) * 1000)
            yield {
                "type": "metadata",
                "data": {
                    "confidence": confidence,
                    "intent": intent.value,
                    "processing_time_ms": elapsed_ms,
                    "bias_analysis": bias_analysis,
                },
            }

        except Exception:
            logger.exception("Streaming error")
            yield {"type": "error", "data": "Internal streaming error"}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_langchain_rag_pipeline: Optional[LangChainRAGPipeline] = None


def get_langchain_rag_pipeline():
    global _langchain_rag_pipeline
    if _langchain_rag_pipeline is None:
        _langchain_rag_pipeline = LangChainRAGPipeline()
    return _langchain_rag_pipeline
