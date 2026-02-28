from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List

from app.core.database import get_db
from app.core.config import settings

# Use optional auth for hackathon/development
if settings.SKIP_AUTH:
    from app.core.auth_optional import get_current_user
else:
    from app.core.auth import get_current_user

from app.models import User, Session as ChatSession, Message
from app.schemas.session import SessionResponse, SessionListResponse, SessionCreateRequest

router = APIRouter()


@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all sessions for current user"""
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    
    if not user:
        return {
            "sessions": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.updated_at.desc()).offset(offset).limit(limit).all()
    
    total = db.query(ChatSession).filter(ChatSession.user_id == user.id).count()
    
    return {
        "sessions": sessions,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new chat session"""
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    session = ChatSession(
        user_id=user.id,
        session_name=request.session_name or "New Chat"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get session details"""
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all messages for a session"""
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.asc()).all()
    
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations or [],
                "confidence": msg.confidence,
                "intent": msg.intent,
                "created_at": msg.created_at
            }
            for msg in messages
        ]
    }


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a session"""
    user = db.query(User).filter(User.auth0_id == current_user["auth0_id"]).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    
    return {"message": "Session deleted successfully"}
