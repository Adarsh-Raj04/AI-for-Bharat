# Backend - FastAPI Server

MedResearch AI backend with RAG pipeline for medical research queries.

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Edit `.env` file:

```env
# Required
USE_OPENAI=True
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_INDEX_NAME=medresearch-ai
DATABASE_URL=postgresql://user:pass@host:port/db
ENABLE_RAG=True

# Optional
REDIS_URL=  # Leave empty to disable caching
```

### 3. Initialize Database

```bash
alembic upgrade head
```

### 4. Test Setup

```bash
python test_openai_setup.py
```

### 5. Start Server

```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

## Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/    # API routes
│   ├── core/                # Config, auth, database
│   ├── models/              # Database models
│   ├── schemas/             # Request/Response schemas
│   ├── services/            # RAG pipeline
│   └── main.py              # FastAPI app
├── alembic/                 # Database migrations
├── .env                     # Configuration
├── requirements.txt         # Dependencies
└── test_openai_setup.py     # Setup verification
```

## API Endpoints

- `POST /api/v1/chat/query` - Submit query to RAG pipeline
- `GET /api/v1/sessions` - List chat sessions
- `GET /api/v1/auth/me` - Get current user

Full docs: http://localhost:8000/docs

## How It Works

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for complete explanation of:
- RAG pipeline flow
- Each component's role
- Query processing steps
- Caching strategy
- Error handling
- Performance optimization

## Troubleshooting

### "Dummy response" issue
1. Check `ENABLE_RAG=True` in `.env`
2. Check Pinecone API key is set
3. Verify Pinecone index exists
4. Run `python test_openai_setup.py`

### "Redis authentication failed"
Leave `REDIS_URL` empty in `.env` - Redis is optional.

### "No documents retrieved"
Ingest data first using the data-pipeline.

### "python-jose error"
```bash
pip uninstall python-jose -y
pip install "python-jose[cryptography]==3.3.0"
```

## Development

### Run Tests
```bash
python test_openai_setup.py
```

### Create Migration
```bash
alembic revision --autogenerate -m "description"
```

### Apply Migration
```bash
alembic upgrade head
```

## Documentation

- `README.md` - This file (quick start)
- `HOW_IT_WORKS.md` - Complete technical explanation
- `../SETUP.md` - Full project setup guide
