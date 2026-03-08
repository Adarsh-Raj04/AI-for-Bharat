from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.core.database import Base


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    intent = Column(String, nullable=True)  # summarization, comparison, etc.
    confidence = Column(Float, nullable=True)
    citations = Column(JSON, nullable=True)  # Store citations as JSON
    timeline = Column(JSON, nullable=True, default=list)
    source_id = Column(String, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    compare_docs = Column(
        JSON, nullable=True
    )  # Store compare_docs as JSON, e.g. [{"source_id": "id_a", "title": "title_a"}, {"source_id": "id_b", "title": "title_b"}]

    # Relationships
    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<Message {self.id} - {self.role}>"
