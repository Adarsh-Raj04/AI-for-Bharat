"""
Ingest Endpoint — POST /api/v1/ingest/document
Accepts an uploaded PDF/DOCX, indexes to Pinecone, returns summary + session info.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.v1.endpoints.auth import get_current_user
from app.core.database import get_db
from app.models import User
from app.models import Session as ChatSession
from app.models.message import Message
from app.services.ingest_service import get_ingest_service
from app.services.langchain_rag_pipeline import get_langchain_rag_pipeline
from app.services.session_namer import generate_session_name
from app.services.url_fetcher import get_document_parser

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB

_DEFAULT_SESSION_NAME = "Document Summary"


@router.post("/document")
async def ingest_document(
    file: UploadFile = File(...),
    # BUG FIX 1: was Form(...) — required, caused 422 when frontend sends empty string
    # Now Optional with default "" so new chats (no session yet) work fine
    session_id: Optional[str] = Form(default=""),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ── Validate file before doing any DB work ────────────────────────────
    content_type = file.content_type or ""
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower()

    if content_type not in ALLOWED_MIME_TYPES and ext not in ("pdf", "docx", "doc"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or DOCX.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 5 MB.",
        )

    # ── Resolve user ──────────────────────────────────────────────────────
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

    # ── Resolve or create session ─────────────────────────────────────────
    is_new_session = False
    session = None

    if session_id:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user.id,
            )
            .first()
        )

    if not session:
        # New session — placeholder name, will be improved after parsing
        is_new_session = True
        session = ChatSession(
            user_id=user.id,
            session_name=_DEFAULT_SESSION_NAME,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # ── Parse document ────────────────────────────────────────────────────
    parser = get_document_parser()
    doc = parser.parse(file_bytes, filename)

    if doc.get("error"):
        raise HTTPException(
            status_code=422, detail=f"Could not parse file: {doc['error']}"
        )
    if not doc.get("text"):
        raise HTTPException(
            status_code=422, detail="No readable text found in this file."
        )

    doc.setdefault("url", "")
    doc.setdefault("source_type", "uploaded_document")

    # ── Ingest + summarize ────────────────────────────────────────────────
    pipeline = get_langchain_rag_pipeline()
    ingest_svc = get_ingest_service(
        pinecone_index=pipeline.pinecone_index,
        embedding_model=pipeline.embeddings,
        summarization_chain=pipeline.summarization_chain,
    )
    result = ingest_svc.ingest_and_summarize(doc)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # ── BUG FIX 2+3: Rename new session using doc title, not summary text ─
    # Old code: `if not session_id:` → always False since session exists by now
    # Old code: passed summary[:300] to generate_session_name → garbage name
    # Fix: check is_new_session flag; use doc title or filename as the input
    if is_new_session:
        doc_title = (
            doc.get("title")  # parsed metadata title (PubMed, etc.)
            or result.get("citation", {}).get("title")  # citation title from ingest
            or filename  # fallback: original filename
        )
        ai_name = generate_session_name(doc_title)
        session.session_name = ai_name or _DEFAULT_SESSION_NAME
        db.commit()
        db.refresh(session)

    # ── Persist summary message ───────────────────────────────────────────
    message = Message(
        session_id=session.id,
        role="assistant",
        content=result["summary"],
        intent="summarization",
        confidence=1.0,
        citations=[result["citation"]] if result.get("citation") else [],
        source_id=result["source_id"],
    )
    db.add(message)
    session.total_messages = (session.total_messages or 0) + 1
    session.updated_at = datetime.now()
    db.commit()
    db.refresh(message)

    return JSONResponse(
        {
            "session_id": str(session.id),
            "session_name": session.session_name,
            "summary": result["summary"],
            "source_id": result["source_id"],
            "chunks_indexed": result["chunks_indexed"],
            "already_existed": result["already_existed"],
            "citation": result["citation"],
        }
    )
