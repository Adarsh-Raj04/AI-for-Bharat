from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.rate_limit import limiter
from app.api.v1.api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# Set up rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "message": "MedResearch AI API",
        "version": "1.0.0",
        "docs": f"{settings.API_V1_PREFIX}/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "medresearch-api"
    }


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "status": "success",
        "message": "API is working correctly!",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "api": settings.API_V1_PREFIX
        }
    }


@app.get("/test-mock-ai")
async def test_mock_ai(query: str = "Summarize PMID 33301246"):
    """
    Test the mock AI response generator without authentication.
    Try different queries to see different response types:
    - 'Summarize PMID 33301246' - Summarization intent
    - 'Compare drug A vs drug B' - Comparison intent
    - 'FDA regulatory requirements' - Regulatory intent
    - 'What is cancer?' - General Q&A intent
    """
    from app.api.v1.endpoints.chat import generate_mock_response
    
    mock_response = generate_mock_response(query)
    
    return {
        "query": query,
        "response": mock_response,
        "note": "This is a mock response. Real AI integration coming soon!"
    }
