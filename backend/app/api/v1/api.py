from fastapi import APIRouter
from app.api.v1.endpoints import chat, auth, sessions, analytics, ingest

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
