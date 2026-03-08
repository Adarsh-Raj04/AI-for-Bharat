from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import uuid
import json
from typing import Dict
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
from app.services.timeline_service import build_research_timeline
from app.services.session_namer import generate_session_name
import logging


router = APIRouter()
logger = logging.getLogger(__name__)


def _get_or_create_user(db: Session, auth0_id: str, email: str, name: str = None):
    user = db.query(User).filter(User.auth0_id == auth0_id).first()
    if not user:
        user = User(auth0_id=auth0_id, email=email, name=name)
        db.add(user)
        db.flush()
    return user


def _get_or_create_session(db: Session, user_id: str, session_id: str = None):
    if session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = ChatSession(
            user_id=user_id,
            session_name=f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        )
        db.add(session)
        db.flush()
    return session


def _persist_assistant_message(db: Session, redis_client, session, ai_response: dict):
    assistant_message = Message(
        session_id=session.id,
        role="assistant",
        content=ai_response["text"],
        intent=ai_response.get("intent", "general_qa"),
        confidence=ai_response.get("confidence", 0.0),
        citations=ai_response.get("citations", []),
        timeline=ai_response.get("timeline", []),
        tokens_used=ai_response.get("tokens_used", 0),
    )
    db.add(assistant_message)
    append_message_to_redis(redis_client, session.id, "assistant", ai_response["text"])
    session.total_messages += 1
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(assistant_message)
    return assistant_message


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_generator(chat_request: ChatRequest, current_user: Dict, db: Session):
    redis_client = get_redis_client()

    user = _get_or_create_user(
        db,
        auth0_id=current_user["auth0_id"],
        email=current_user["email"],
        name=current_user.get("name"),
    )

    try:
        session = _get_or_create_session(db, user.id, chat_request.session_id)
    except HTTPException as e:
        yield _sse("error", {"message": e.detail})
        return

    # Store compare_docs so ChatMessage can render "Doc A vs Doc B" on reload
    compare_docs_payload = None
    if chat_request.compare_filters and chat_request.compare_titles:
        compare_docs_payload = [
            {"source_id": sid, "title": title}
            for sid, title in zip(
                chat_request.compare_filters, chat_request.compare_titles
            )
        ]
    user_msg = Message(
        session_id=session.id,
        role="user",
        content=chat_request.message,
    )
    # compare_docs requires the column to exist in the Message model.
    # Add it with: compare_docs = Column(JSON, nullable=True)
    # and run: alembic revision --autogenerate -m "add_compare_docs_to_messages"
    #          alembic upgrade head
    if compare_docs_payload and hasattr(Message, "compare_docs"):
        user_msg.compare_docs = compare_docs_payload
    db.add(user_msg)
    append_message_to_redis(redis_client, session.id, "user", chat_request.message)
    session.total_messages += 1
    db.commit()

    yield _sse("session", {"session_id": str(session.id)})

    chat_history = load_chat_history(
        redis_client=redis_client, db=db, session_id=session.id
    )
    pipeline = get_langchain_rag_pipeline()

    accumulated_text = ""
    citations = []
    timeline = []
    confidence = 0.0
    intent = "general_qa"

    try:
        for chunk in pipeline.process_query_stream(
            query=chat_request.message,
            chat_history=chat_history,
            user_id=str(user.id),
            session_id=str(session.id),
            source_filter=chat_request.source_filter,
            compare_filters=chat_request.compare_filters,
            compare_titles=chat_request.compare_titles,  # ← real doc names for prompt
        ):
            chunk_type = chunk.get("type")
            chunk_data = chunk.get("data")

            if chunk_type == "citations":
                citations = chunk_data
                timeline = build_research_timeline(citations)
                yield _sse("citations", {"citations": citations, "timeline": timeline})

            elif chunk_type == "text":
                accumulated_text += chunk_data
                yield _sse("text", {"text": chunk_data})

            elif chunk_type == "metadata":
                confidence = chunk_data.get("confidence", confidence)
                intent = chunk_data.get("intent", intent)

            elif chunk_type == "error":
                yield _sse("error", {"message": chunk_data})
                return

    except Exception as e:
        logger.exception("Streaming pipeline error")
        yield _sse("error", {"message": "Internal streaming error"})
        return

    ai_response = {
        "text": accumulated_text,
        "citations": citations,
        "timeline": timeline,
        "confidence": confidence,
        "intent": intent,
    }

    assistant_msg = _persist_assistant_message(db, redis_client, session, ai_response)

    yield _sse(
        "done",
        {
            "message_id": str(assistant_msg.id),
            "session_id": str(session.id),
            "confidence": confidence,
            "intent": intent,
        },
    )

    # ── Auto-rename on first exchange ─────────────────────────────────────
    # Only fires when session still has the default timestamp name
    import re as _re

    if session.total_messages <= 2 and _re.match(
        r"^Chat \d{4}-\d{2}-\d{2}", session.session_name or ""
    ):
        try:
            new_name = generate_session_name(chat_request.message)
            session.session_name = new_name
            db.commit()
            yield _sse(
                "rename",
                {
                    "session_id": str(session.id),
                    "session_name": new_name,
                },
            )
            logger.info("Session %s renamed to %r", session.id, new_name)
        except Exception:
            logger.warning(
                "Auto-rename failed for session %s", session.id, exc_info=True
            )


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if chat_request.stream:
        return StreamingResponse(
            _stream_generator(chat_request, current_user, db),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    redis_client = get_redis_client()
    user = _get_or_create_user(
        db,
        auth0_id=current_user["auth0_id"],
        email=current_user["email"],
        name=current_user.get("name"),
    )
    session = _get_or_create_session(db, user.id, chat_request.session_id)
    chat_history = load_chat_history(
        redis_client=redis_client, db=db, session_id=session.id
    )

    user_message = Message(
        session_id=session.id, role="user", content=chat_request.message
    )
    db.add(user_message)
    append_message_to_redis(redis_client, session.id, "user", chat_request.message)

    pipeline = get_langchain_rag_pipeline()

    try:
        ai_response = pipeline.process_query(
            query=chat_request.message,
            chat_history=chat_history,
            user_id=str(user.id),
            session_id=str(session.id),
            source_filter=chat_request.source_filter,  # ← threads document filter
        )
    except Exception as e:
        logger.exception("RAG pipeline failed")
        ai_response = {
            "text": "Unable to process your query at this time.",
            "citations": [],
            "confidence": 0.0,
            "intent": "error",
            "error": str(e),
        }

    assistant_msg = _persist_assistant_message(db, redis_client, session, ai_response)

    return ChatResponse(
        message_id=assistant_msg.id,
        session_id=session.id,
        response={
            "text": ai_response["text"],
            "citations": ai_response.get("citations", []),
            "confidence": ai_response.get("confidence", 0.0),
            "intent": ai_response.get("intent", "general_qa"),
            "sources_used": len(ai_response.get("citations", [])),
            "requires_human_review": ai_response.get("requires_human_review", False),
            "safety_blocked": ai_response.get("safety_blocked", False),
            "timeline": ai_response.get("timeline", []),
        },
        metadata={
            "processing_time_ms": ai_response.get("processing_time_ms", 0),
            "tokens_used": ai_response.get("tokens_used", 0),
            "model": ai_response.get("model", "unknown"),
        },
        timestamp=datetime.now(),
        retrieved_documents=ai_response.get("retrieved_documents", []),
        bias_analysis=ai_response.get("bias_analysis"),
        requires_human_review=ai_response.get("requires_human_review", False),
        safety_blocked=ai_response.get("safety_blocked", False),
    )


def generate_mock_response(query: str) -> Dict:
    query_lower = query.lower()

    if "summarize" in query_lower or "pmid" in query_lower:
        intent = "summarization"
        text = """This is a mock summary of a research paper.\n\n**Objective:** The study aimed to evaluate the efficacy and safety of Drug X in patients with Condition Y.\n\n**Methods:** Randomized, double-blind, placebo-controlled trial involving 450 participants.\n\n**Results:** 35% improvement vs placebo (p<0.001).\n\n**Conclusions:** Drug X demonstrated significant efficacy [1].\n\n*Mock response.*"""
        citations = [
            {
                "number": 1,
                "source_id": "PMID:12345678",
                "source_type": "pubmed",
                "title": "Efficacy of Drug X",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 0.95,
            }
        ]
        confidence = 0.92
    elif "compare" in query_lower:
        intent = "comparison"
        text = "Mock comparison response."
        citations = []
        confidence = 0.88
    elif "fda" in query_lower or "regulatory" in query_lower:
        intent = "compliance_regulatory"
        text = "Mock regulatory response."
        citations = []
        confidence = 0.91
    else:
        intent = "general_qa"
        text = f'Mock response for: "{query}"'
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

    user_message = Message(
        session_id=session.id, role="user", content=chat_request.message
    )
    db.add(user_message)
    db.commit()

    async def event_generator():
        if not settings.ENABLE_RAG:
            mock_response = generate_mock_response(chat_request.message)
            yield f"data: {json.dumps({'type': 'citations', 'data': mock_response['citations']})}\n\n"
            text = mock_response["text"]
            for i in range(0, len(text), 50):
                yield f"data: {json.dumps({'type': 'text', 'data': text[i:i+50]})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': {'session_id': str(session.id)}})}\n\n"
        else:
            try:
                rag_pipeline = (
                    get_langchain_rag_pipeline()
                    if settings.LANGSMITH_TRACING
                    else get_rag_pipeline()
                )
                full_text = ""
                for chunk in rag_pipeline.process_query_stream(chat_request.message):
                    if chunk["type"] == "text":
                        full_text += chunk["data"]
                    yield f"data: {json.dumps(chunk)}\n\n"
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
                session.total_messages += 2
                session.updated_at = datetime.utcnow()
                db.commit()
                yield f"data: {json.dumps({'type': 'done', 'data': {'session_id': str(session.id)}})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
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
    if not message_id and not session_id:
        raise HTTPException(
            status_code=400, detail="Provide either message_id or session_id"
        )

    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if message_id:
        message = db.query(Message).filter(Message.id == message_id).first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        session = (
            db.query(ChatSession).filter(ChatSession.id == message.session_id).first()
        )
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        user_message = (
            db.query(Message)
            .filter(
                Message.session_id == message.session_id,
                Message.role == "user",
                Message.created_at < message.created_at,
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
    else:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session or session.user_id != user.id:
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

    try:
        export_service = get_export_service()
        file_bytes = export_service.export(
            format=format,
            query=query,
            response=response_text,
            citations=citations,
            metadata=metadata,
        )
        content_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "markdown": "text/markdown",
            "md": "text/markdown",
        }
        extensions = {"pdf": "pdf", "docx": "docx", "markdown": "md", "md": "md"}
        content_type = content_types.get(format.lower(), "application/octet-stream")
        extension = extensions.get(format.lower(), "txt")
        filename = f"medresearch_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{extension}"
        return Response(
            content=file_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
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
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    session = db.query(ChatSession).filter(ChatSession.id == message.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        from app.services.citation_tracker import get_citation_tracker

        citation_tracker = get_citation_tracker()
        report = citation_tracker.generate_citation_report(
            message.content, message.citations or []
        )
        return {
            "message_id": message_id,
            "report": report,
            "formatted_citations": {
                "apa": citation_tracker.format_citations(
                    message.citations or [], style="apa"
                ),
                "mla": citation_tracker.format_citations(
                    message.citations or [], style="mla"
                ),
            },
        }
    except Exception as e:
        logger.error(f"Citation report error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate citation report: {str(e)}"
        )
