from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class DisclaimerRequest(BaseModel):
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class DisclaimerResponse(BaseModel):
    accepted: bool
    version: int = 1


class UserResponse(BaseModel):
    id: int  # Changed from str
    email: EmailStr
    name: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True
