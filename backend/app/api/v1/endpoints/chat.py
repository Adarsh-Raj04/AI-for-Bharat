from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import uuid
import json
from typing import Optional, Dict
from fastapi.responses import Response
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.core.config import settings
from app.core.auth import get_current_user
from app.core.rate_limit import limiter
from app.schemas.chat import ChatRequest, ChatResponse, MessageResponse, Citation
from app.models import User, Session as ChatSession, Message
from app.services.rag_pipeline import get_rag_pipeline
from app.services.langchain_rag_pipeline import get_langchain_rag_pipeline
from app.services.chat_memory import load_chat_history, append_message_to_redis
from app.services.export_service import get_export_service
from langsmith import traceable
from app.services.cache_service import get_redis_client
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/query", response_model=ChatResponse)
@limiter.limit("20/minute")
async def query(
    request: Request,
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Process query through RAG pipeline (always uses RAG, no mock fallback)
    This is the main endpoint for production use
    """

    # Get or create user
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    if not user:
        user = User(
            auth0_id=current_user["auth0_id"],
            email=current_user["email"],
            name=current_user.get("name"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Get or create session
    if chat_request.session_id:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == chat_request.session_id,
                ChatSession.user_id == user.id,
            )
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = ChatSession(
            user_id=user.id,
            session_name=f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # Save user message
    user_message = Message(
        session_id=session.id, role="user", content=chat_request.message
    )
    db.add(user_message)
    db.commit()

    # Generate AI response using RAG pipeline
    # Use LangChain pipeline if LANGSMITH_TRACING is enabled, otherwise use legacy pipeline
    try:
        if settings.LANGSMITH_TRACING:
            rag_pipeline = get_langchain_rag_pipeline()
        else:
            rag_pipeline = get_rag_pipeline()
            
        ai_response = rag_pipeline.process_query(
            query=chat_request.message, user_id=str(user.id), session_id=str(session.id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

    # Save assistant message
    assistant_message = Message(
        session_id=session.id,
        role="assistant",
        content=ai_response["text"],
        intent=ai_response.get("intent", "general_qa"),
        confidence=ai_response.get("confidence", 0.0),
        citations=ai_response.get("citations", []),
        tokens_used=ai_response.get("tokens_used", 0),
    )
    db.add(assistant_message)

    # Update session
    session.total_messages += 2
    session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(assistant_message)

    # Build response
    response = ChatResponse(
        message_id=assistant_message.id,
        session_id=session.id,
        response={
            "text": ai_response["text"],
            "citations": ai_response.get("citations", []),
            "confidence": ai_response.get("confidence", 0.0),
            "intent": ai_response.get("intent", "general_qa"),
            "sources_used": ai_response.get("sources_used", 0),
            "requires_human_review": ai_response.get("requires_human_review", False),
            "safety_blocked": ai_response.get("safety_blocked", False),
        },
        metadata={
            "processing_time_ms": ai_response.get("processing_time_ms", 0),
            "tokens_used": ai_response.get("tokens_used", 0),
            "model": ai_response.get("model", "unknown"),
        },
        timestamp=datetime.utcnow(),
        retrieved_documents=ai_response.get("retrieved_documents", []),
        bias_analysis=ai_response.get("bias_analysis"),
        requires_human_review=ai_response.get("requires_human_review", False),
        safety_blocked=ai_response.get("safety_blocked", False),
    )

    return response


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Conversational RAG chat endpoint
    Redis-first memory + DB persistence
    """

    redis_client = get_redis_client()

    # -----------------------------
    # Get or create user
    # -----------------------------
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()


    if not user:
        user = User(
            auth0_id=current_user["auth0_id"],
            email=current_user["email"],
            name=current_user.get("name"),
        )
        db.add(user)
        db.flush()

    # -----------------------------
    # Get or create session
    # -----------------------------
    if chat_request.session_id:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == chat_request.session_id,
                ChatSession.user_id == user.id,
            )
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = ChatSession(
            user_id=user.id,
            session_name=f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        )
        db.add(session)
        db.flush()

    # -----------------------------
    # Load chat history (REDIS FIRST)
    # -----------------------------
    chat_history = load_chat_history(
        redis_client=redis_client,
        db=db,
        session_id=session.id,
    )

    # -----------------------------
    # Save user message
    # -----------------------------
    user_message = Message(
        session_id=session.id,
        role="user",
        content=chat_request.message,
    )
    db.add(user_message)

    append_message_to_redis(
        redis_client,
        session.id,
        "user",
        chat_request.message,
    )

    # -----------------------------
    # Generate AI response
    # -----------------------------
    if settings.ENABLE_RAG:
        try:
            # Use LangChain pipeline if LANGSMITH_TRACING is enabled
            if settings.LANGSMITH_TRACING:
                rag_pipeline = get_langchain_rag_pipeline()
            else:
                rag_pipeline = get_rag_pipeline()

            ai_response = rag_pipeline.process_query(
                query=chat_request.message,
                chat_history=chat_history,  # ⭐ IMPORTANT
                user_id=str(user.id),
                session_id=str(session.id),
            )
        except Exception as e:
            logger.exception("RAG pipeline failed")
            ai_response = generate_mock_response(chat_request.message)
            ai_response["error"] = str(e)
    else:
        logger.info("Rag is not enabled, will be giving mock response.")
        ai_response = generate_mock_response(chat_request.message)

    # -----------------------------
    # Save assistant message
    # -----------------------------
    assistant_message = Message(
        session_id=session.id,
        role="assistant",
        content=ai_response["text"],
        intent=ai_response.get("intent", "general_qa"),
        confidence=ai_response.get("confidence", 0.0),
        citations=ai_response.get("citations", []),
        tokens_used=ai_response.get("tokens_used", 0),
    )
    db.add(assistant_message)

    append_message_to_redis(
        redis_client,
        session.id,
        "assistant",
        ai_response["text"],
    )

    # -----------------------------
    # Update session
    # -----------------------------
    session.total_messages += 2
    session.updated_at = datetime.utcnow()

    # ⭐ SINGLE COMMIT
    db.commit()
    db.refresh(assistant_message)

    # -----------------------------
    # Build response
    # -----------------------------
    response = ChatResponse(
        message_id=assistant_message.id,
        session_id=session.id,
        response={
            "text": ai_response["text"],
            "citations": ai_response.get("citations", []),
            "confidence": ai_response.get("confidence", 0.0),
            "intent": ai_response.get("intent", "general_qa"),
            "sources_used": ai_response.get("sources_used", 0),
            "requires_human_review": ai_response.get("requires_human_review", False),
            "safety_blocked": ai_response.get("safety_blocked", False),
        },
        metadata={
            "processing_time_ms": ai_response.get("processing_time_ms", 0),
            "tokens_used": ai_response.get("tokens_used", 0),
            "model": ai_response.get("model", "mock-model"),
        },
        timestamp=datetime.now(),
        retrieved_documents=ai_response.get("retrieved_documents", []),
        bias_analysis=ai_response.get("bias_analysis"),
        requires_human_review=ai_response.get("requires_human_review", False),
        safety_blocked=ai_response.get("safety_blocked", False),
    )

    return response


def generate_mock_response(query: str) -> Dict:
    """
    Generate mock response for testing
    Will be replaced with actual AI integration
    """

    # Simple intent detection
    query_lower = query.lower()

    if "summarize" in query_lower or "pmid" in query_lower:
        intent = "summarization"
        text = """This is a mock summary of a research paper. 

**Objective:** The study aimed to evaluate the efficacy and safety of Drug X in patients with Condition Y.

**Methods:** This was a randomized, double-blind, placebo-controlled trial involving 450 participants across 25 centers.

**Results:** The primary endpoint was met with a 35% improvement in the treatment group compared to placebo (p<0.001). Common adverse events included headache (12%) and nausea (8%).

**Conclusions:** Drug X demonstrated significant efficacy with an acceptable safety profile for treating Condition Y [1].

*Note: This is a mock response for demonstration purposes.*"""

        citations = [
            {
                "number": 1,
                "source_id": "PMID:12345678",
                "source_type": "pubmed",
                "title": "Efficacy of Drug X in Treatment of Condition Y: A Randomized Trial",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 0.95,
            }
        ]
        confidence = 0.92

    elif "compare" in query_lower:
        intent = "comparison"
        text = """Here's a comparison of the two drugs:

| Metric | Drug A | Drug B |
|--------|--------|--------|
| Response Rate | 45% [1] | 38% [2] |
| Median PFS | 8.2 months [1] | 6.9 months [2] |
| Grade 3+ AEs | 22% [1] | 28% [2] |

**Key Findings:**
- Drug A showed superior response rates and progression-free survival
- Drug B had a slightly higher rate of severe adverse events
- Both drugs demonstrated acceptable safety profiles [1,2]

*Note: This is a mock comparison for demonstration purposes.*"""

        citations = [
            {
                "number": 1,
                "source_id": "NCT04280705",
                "source_type": "clinical_trial",
                "title": "Phase 3 Study of Drug A in Advanced Disease",
                "url": "https://clinicaltrials.gov/study/NCT04280705",
                "relevance_score": 0.89,
            },
            {
                "number": 2,
                "source_id": "NCT03456789",
                "source_type": "clinical_trial",
                "title": "Efficacy and Safety of Drug B: A Multicenter Trial",
                "url": "https://clinicaltrials.gov/study/NCT03456789",
                "relevance_score": 0.87,
            },
        ]
        confidence = 0.88

    elif "fda" in query_lower or "regulatory" in query_lower:
        intent = "compliance_regulatory"
        text = """**FDA Accelerated Approval Requirements:**

The FDA's accelerated approval pathway allows earlier approval of drugs for serious conditions based on surrogate endpoints [1].

**Key Criteria:**
1. Drug treats a serious or life-threatening condition
2. Provides meaningful advantage over existing treatments
3. Demonstrates effect on surrogate endpoint reasonably likely to predict clinical benefit

**Post-Approval Requirements:**
- Confirmatory trials must verify clinical benefit
- Failure to conduct trials may result in withdrawal of approval

**Recent Updates:** FDA issued updated guidance in 2023 clarifying expectations for post-marketing studies [1].

*Note: This is a mock response. Always consult official FDA guidance for regulatory decisions.*"""

        citations = [
            {
                "number": 1,
                "source_id": "FDA-2023-D-1234",
                "source_type": "fda",
                "title": "Accelerated Approval Program Guidance",
                "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents",
                "relevance_score": 0.94,
            }
        ]
        confidence = 0.91

    else:
        intent = "general_qa"
        text = f"""Thank you for your question: "{query}"

This is a mock response from MedResearch AI. In the production version, I would:

1. Search through PubMed, ClinicalTrials.gov, and FDA databases
2. Retrieve relevant research papers and clinical trials
3. Use AI to synthesize information from multiple sources
4. Provide a comprehensive answer with citations

**Example capabilities:**
- Summarize research papers by PMID
- Compare drug efficacy across clinical trials
- Explain FDA regulatory requirements
- Generate literature reviews

*Note: This is a demonstration response. The full AI integration is coming soon!*"""

        citations = []
        confidence = 0.75

    return {
        "text": text,
        "citations": citations,
        "confidence": confidence,
        "intent": intent,
    }


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Process chat message with streaming response
    Returns Server-Sent Events (SSE) stream
    """

    # Get or create user
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    if not user:
        user = User(
            auth0_id=current_user["auth0_id"],
            email=current_user["email"],
            name=current_user.get("name"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Get or create session
    if chat_request.session_id:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == chat_request.session_id,
                ChatSession.user_id == user.id,
            )
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = ChatSession(
            user_id=user.id,
            session_name=f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # Save user message
    user_message = Message(
        session_id=session.id, role="user", content=chat_request.message
    )
    db.add(user_message)
    db.commit()

    async def event_generator():
        """Generate SSE events"""
        if not settings.ENABLE_RAG:
            # Mock streaming for development
            mock_response = generate_mock_response(chat_request.message)

            # Send citations
            yield f"data: {json.dumps({'type': 'citations', 'data': mock_response['citations']})}\n\n"

            # Stream text in chunks
            text = mock_response["text"]
            chunk_size = 50
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                yield f"data: {json.dumps({'type': 'text', 'data': chunk})}\n\n"

            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'data': {'confidence': mock_response['confidence'], 'intent': mock_response['intent'], 'sources_used': len(mock_response['citations'])}})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': {'session_id': str(session.id)}})}\n\n"
        else:
            # Use RAG pipeline streaming
            try:
                # Use LangChain pipeline if LANGSMITH_TRACING is enabled
                if settings.LANGSMITH_TRACING:
                    rag_pipeline = get_langchain_rag_pipeline()
                else:
                    rag_pipeline = get_rag_pipeline()
                    
                full_text = ""

                for chunk in rag_pipeline.process_query_stream(chat_request.message):
                    if chunk["type"] == "text":
                        full_text += chunk["data"]
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Save assistant message after streaming completes
                assistant_message = Message(
                    session_id=session.id,
                    role="assistant",
                    content=full_text,
                    intent="general_qa",
                    confidence=0.0,
                    citations=[],
                    tokens_used=0,
                )
                db.add(assistant_message)

                # Update session
                session.total_messages += 2
                session.updated_at = datetime.utcnow()
                db.commit()

                yield f"data: {json.dumps({'type': 'done', 'data': {'session_id': str(session.id)}})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/export")
@limiter.limit("10/minute")
async def export_message(
    request: Request,
    message_id: Optional[str] = None,
    session_id: Optional[str] = None,
    format: str = "markdown",
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Export chat responses.

    Behavior:
    - If message_id is provided → export that assistant response
    - If session_id is provided → export full chat session
    """

    # Validate input
    if not message_id and not session_id:
        raise HTTPException(
            status_code=400,
            detail="Provide either message_id or session_id"
        )

    user = db.query(User).filter(
        User.auth0_id == current_user["auth0_id"]
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # --------------------------------------
    # CASE 1: Export a single assistant message
    # --------------------------------------
    if message_id:

        message = db.query(Message).filter(
            Message.id == message_id
        ).first()

        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        session = db.query(ChatSession).filter(
            ChatSession.id == message.session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get user query before this assistant message
        user_message = (
            db.query(Message)
            .filter(
                Message.session_id == message.session_id,
                Message.role == "user",
                Message.created_at < message.created_at
            )
            .order_by(Message.created_at.desc())
            .first()
        )

        query = user_message.content if user_message else "Query not found"
        response_text = message.content
        citations = message.citations or []

        metadata = {
            "intent": message.intent,
            "confidence": message.confidence,
            "model": "MedResearch AI",
            "tokens_used": message.tokens_used,
        }

    # --------------------------------------
    # CASE 2: Export entire session
    # --------------------------------------
    else:

        session = db.query(ChatSession).filter(
            ChatSession.id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        messages = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at)
            .all()
        )

        if not messages:
            raise HTTPException(status_code=404, detail="No messages found")

        conversation = ""

        for m in messages:
            role = "User" if m.role == "user" else "Assistant"
            conversation += f"\n\n### {role}\n{m.content}"

        query = "Full conversation export"
        response_text = conversation
        citations = []

        metadata = {
    "intent": "conversation_export",
    "confidence": 0.0,
    "model": "MedResearch AI",
    "tokens_used": sum([m.tokens_used or 0 for m in messages]),
}

    # --------------------------------------
    # Generate export file
    # --------------------------------------

    try:

        export_service = get_export_service()

        file_bytes = export_service.export(
            format=format,
            query=query,
            response=response_text,
            citations=citations,
            metadata=metadata
        )

        content_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "markdown": "text/markdown",
            "md": "text/markdown"
        }

        extensions = {
            "pdf": "pdf",
            "docx": "docx",
            "markdown": "md",
            "md": "md"
        }

        content_type = content_types.get(format.lower(), "application/octet-stream")
        extension = extensions.get(format.lower(), "txt")

        filename = f"medresearch_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{extension}"

        return Response(
            content=file_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/citation-report/{message_id}")
async def get_citation_report(
    request: Request,
    message_id: int,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get detailed citation report for a message
    
    Phase 2 Feature #6 - Enhanced Citations
    
    Args:
        message_id: ID of the assistant message
    
    Returns:
        Citation report with validation, enrichment, and statistics
    """
    # Get the message
    message = db.query(Message).filter(Message.id == message_id).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Verify user owns this message's session
    session = db.query(ChatSession).filter(ChatSession.id == message.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Generate citation report
    try:
        from app.services.citation_tracker import get_citation_tracker
        
        citation_tracker = get_citation_tracker()
        response_text = message.content
        citations = message.citations or []
        
        report = citation_tracker.generate_citation_report(response_text, citations)
        
        return {
            "message_id": message_id,
            "report": report,
            "formatted_citations": {
                "apa": citation_tracker.format_citations(citations, style="apa"),
                "mla": citation_tracker.format_citations(citations, style="mla"),
            }
        }
        
    except Exception as e:
        logger.error(f"Citation report error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate citation report: {str(e)}")
