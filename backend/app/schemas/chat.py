from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False
    options: Optional[Dict] = None


class Citation(BaseModel):
    number: int
    source_id: str
    source_type: str
    title: str
    url: str
    relevance_score: float
    doi: Optional[str] = ""
    publication_date: Optional[str] = ""


class RetrievedDocument(BaseModel):
    id: str
    title: str
    source_type: str
    url: str
    relevance_score: float
    doi: Optional[str] = ""
    publication_date: Optional[str] = ""


class BiasFlag(BaseModel):
    type: str
    severity: str
    message: str


class BiasAnalysis(BaseModel):
    has_bias: bool
    bias_score: float
    bias_flags: List[BiasFlag] = []
    recommendations: List[str] = []


class ChatResponse(BaseModel):
    message_id: str
    session_id: str
    response: Dict
    metadata: Dict
    timestamp: datetime
    retrieved_documents: Optional[List[RetrievedDocument]] = []
    bias_analysis: Optional[BiasAnalysis] = None
    requires_human_review: bool = False
    safety_blocked: bool = False


class MessageResponse(BaseModel):
    text: str
    citations: List[Citation] = []
    confidence: float
    intent: str
    sources_used: int
    requires_human_review: bool = False
    safety_blocked: bool = False
