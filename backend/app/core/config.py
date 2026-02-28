from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # Auth0
    AUTH0_DOMAIN: str
    AUTH0_API_AUDIENCE: str
    AUTH0_ISSUER: str
    AUTH0_ALGORITHMS: str = "RS256"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "MedResearch AI"
    
    # Security
    SECRET_KEY: str
    
    # Development/Hackathon Settings
    SKIP_AUTH: bool = False  # Set to False for production with full Auth0
    
    # OpenAI Configuration (Recommended - Simple Setup)
    USE_OPENAI: bool = False
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"  # or gpt-4o-mini, gpt-4-turbo
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # or text-embedding-3-large
    
    # AWS Configuration
    USE_AWS_BEDROCK: bool = False
    USE_AWS_EMBEDDINGS: bool = False
    USE_AWS_OPENSEARCH: bool = False
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    
    # AWS Bedrock
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    BEDROCK_EMBEDDING_MODEL: str = "amazon.titan-embed-text-v1"
    
    # AWS OpenSearch Serverless
    OPENSEARCH_ENDPOINT: str = ""
    OPENSEARCH_INDEX: str = "medresearch-docs"
    
    # RAG Pipeline Settings (External Services)
    ANTHROPIC_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "medresearch-ai"
    REDIS_URL: str = "redis://localhost:6379/0"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLAUDE_MODEL: str = "claude-sonnet-4-5-20250929"
    TOP_K_RESULTS: int = 10
    ENABLE_RAG: bool = False
    
    # Observability - LangSmith (optional)
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = ""


    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
