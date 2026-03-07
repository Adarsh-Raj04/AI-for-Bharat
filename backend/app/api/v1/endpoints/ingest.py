"""
Ingest Endpoint — POST /api/v1/ingest
Accepts a URL or uploaded PDF/DOCX, fetches/parses it,
indexes to Pinecone, and returns a summary.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from app.api.v1.endpoints.auth import get_current_user
from app.services.url_fetcher import get_document_parser, get_url_fetcher
from app.services.ingest_service import get_ingest_service
from app.services.langchain_rag_pipeline import get_langchain_rag_pipeline
from app.services.session_namer import generate_session_name
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.message import Message
from app.models import User, Session as ChatSession, Message
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------------------------
# POST /ingest/document
# ---------------------------------------------------------------------------


@router.post("/document")
async def ingest_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):

    # ---------------------------------------------------------------------------
    # Resolve user
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Resolve or create session
    # ---------------------------------------------------------------------------
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

        session_name = "Document Summary"

        session = ChatSession(
            user_id=user.id,
            session_name=session_name,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    """Upload a PDF or DOCX, index it to Pinecone, return summary."""
    # Validate file type
    content_type = file.content_type or ""
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower()

    if content_type not in ALLOWED_MIME_TYPES and ext not in ("pdf", "docx", "doc"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or DOCX.",
        )

    # Read and size-check
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 5 MB."
        )

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

    # Add URL placeholder so ingest_service can build a source_id
    doc.setdefault("url", "")
    doc.setdefault("source_type", "uploaded_document")

    pipeline = get_langchain_rag_pipeline()
    ingest_svc = get_ingest_service(
        pinecone_index=pipeline.pinecone_index,
        embedding_model=pipeline.embeddings,
        summarization_chain=pipeline.summarization_chain,
    )

    result = ingest_svc.ingest_and_summarize(doc)

    if not session_id:
        ai_name = generate_session_name(result["summary"][:300])
        if ai_name:
            session.session_name = ai_name

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

    session.total_messages += 1
    session.updated_at = datetime.now()

    db.commit()
    db.refresh(message)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

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
