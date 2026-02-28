from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class SessionCreateRequest(BaseModel):
    session_name: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    user_id: int  # Changed from str to int
    session_name: Optional[str]
    total_messages: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    limit: int
    offset: int
