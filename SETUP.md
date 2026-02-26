# MedResearch AI - Setup Guide

Complete technical setup guide for the MedResearch AI project.

## Project Structure

```
medresearch-ai/
├── frontend/           # React UI (Port 5173)
├── backend/            # FastAPI Server (Port 8000)
├── data-pipeline/      # Data Ingestion Scripts
├── design.md           # System design document
├── requirements.md     # Functional requirements
└── README.md           # Project overview
```

---

## Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **PostgreSQL** (you have one running)
- **OpenAI API Key** (you have one)
- **Pinecone Account** (for vector database)
- **Redis** (optional - for caching)

---

## Part 1: Backend Setup

### 1.1 Create Virtual Environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Configure Environment

Edit `backend/.env`:

```env
# Database
DATABASE_URL=postgresql://user:password@host:port/database

# OpenAI (REQUIRED)
USE_OPENAI=True
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone (REQUIRED)
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=medresearch-ai

# Enable RAG
ENABLE_RAG=True

# Redis (OPTIONAL - leave empty to disable)
REDIS_URL=

# Auth0 (already configured)
AUTH0_DOMAIN=dev-nvyez2teursdttot.us.auth0.com
AUTH0_API_AUDIENCE=https://medresearch-ai-api
SKIP_AUTH=False
```

### 1.4 Create Pinecone Index

Open Python:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-pinecone-key-here")

# Create index with 1536 dimensions (for text-embedding-3-small)
pc.create_index(
    name="medresearch-ai",
    dimension=1536,
    metric="cosine"
)

print("✅ Pinecone index created!")
```

### 1.5 Initialize Database

```bash
alembic upgrade head
```

### 1.6 Test Setup

```bash
python test_openai_setup.py
```

This will verify:
- ✅ Environment variables are set
- ✅ OpenAI API is working
- ✅ Pinecone index exists
- ✅ RAG pipeline initializes
- ✅ Test query works

### 1.7 Start Backend

```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

---

## Part 2: Data Pipeline Setup

### 2.1 Create Virtual Environment

```bash
cd data-pipeline
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Configure Environment

Create `data-pipeline/.env`:

```env
# OpenAI
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=medresearch-ai

# PubMed (optional)
PUBMED_EMAIL=your-email@example.com
```

### 2.4 Ingest Sample Data

**Option 1: Quick Demo (30 documents)**

```bash
quick_ingest.bat
# Choose option 4 (All Sources)
```

**Option 2: Manual Ingestion**

```bash
# PubMed (50 papers about aspirin)
python ingest/pubmed_ingest.py --query "aspirin" --max-results 50

# ClinicalTrials.gov (50 diabetes trials)
python ingest/clinicaltrials_ingest.py --query "diabetes" --max-results 50

# medRxiv (50 recent preprints)
python ingest/biorxiv_ingest.py --server medrxiv --days 30 --max-results 50
```

### 2.5 Verify Data in Pinecone

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("medresearch-ai")
stats = index.describe_index_stats()

print(f"Total vectors: {stats['total_vector_count']}")
print(f"Dimension: {stats['dimension']}")
```

---

## Part 3: Frontend Setup

### 3.1 Install Dependencies

```bash
cd frontend
npm install
```

### 3.2 Configure Environment

Edit `frontend/.env`:

```env
VITE_AUTH0_DOMAIN=dev-nvyez2teursdttot.us.auth0.com
VITE_AUTH0_CLIENT_ID=aDlj1k3MmDRhFpBKh54LECsqv3dQAjnr
VITE_API_URL=http://localhost:8000
```

### 3.3 Start Frontend

```bash
npm run dev
```

Visit: http://localhost:5173

---

## Part 4: Running the Full Stack

### Terminal 1: Backend

```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

### Terminal 2: Frontend

```bash
cd frontend
npm run dev
```

### Terminal 3: Ingest Data (as needed)

```bash
cd data-pipeline
venv\Scripts\activate
python ingest/pubmed_ingest.py --query "your query" --max-results 50
```

---

## Configuration Details

### OpenAI Models

**Generation Models:**
- `gpt-4o` - Best quality, $2.50/$10 per 1M tokens
- `gpt-4o-mini` - Fast and cheap, $0.15/$0.60 per 1M tokens (recommended for dev)
- `gpt-4-turbo` - High quality, $10/$30 per 1M tokens

**Embedding Models:**
- `text-embedding-3-small` - 1536 dimensions, $0.02 per 1M tokens (recommended)
- `text-embedding-3-large` - 3072 dimensions, $0.13 per 1M tokens

### Pinecone Index

**Dimensions:**
- text-embedding-3-small: **1536 dimensions**
- text-embedding-3-large: **3072 dimensions**

**Important:** Index dimension must match your embedding model!

### Redis (Optional)

Redis is optional for caching. If you don't have Redis:

1. Leave `REDIS_URL` empty in `.env`
2. The system will work without caching
3. Caching reduces API costs and improves speed

**Redis URL Formats:**
```env
# Local Redis (no password)
REDIS_URL=redis://localhost:6379/0

# Redis with password
REDIS_URL=redis://:your-password@host:port/0

# Redis Cloud with username and password
REDIS_URL=redis://username:password@host:port/0

# Disable caching (leave empty)
REDIS_URL=
```

---

## API Endpoints

### Chat Endpoints

- `POST /api/v1/chat/query` - Submit query (uses RAG pipeline)
- `POST /api/v1/chat/stream` - Stream response
- `POST /api/v1/chat/disclaimer/accept` - Accept disclaimer

### Session Endpoints

- `GET /api/v1/sessions` - List user sessions
- `POST /api/v1/sessions` - Create new session
- `GET /api/v1/sessions/{id}` - Get session details
- `GET /api/v1/sessions/{id}/messages` - Get session messages

### Auth Endpoints

- `GET /api/v1/auth/me` - Get current user
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user

**Full API Documentation:** http://localhost:8000/docs

---

## Testing

### Test Backend Setup

```bash
cd backend
python test_openai_setup.py
```

### Test API Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{"message": "What are the side effects of aspirin?"}'
```

### Test Data Ingestion

```bash
cd data-pipeline
python embed/embed_documents.py
```

---

## Troubleshooting

### Issue: "Dummy response" instead of real AI response

**Solution:**
1. Check `ENABLE_RAG=True` in `backend/.env`
2. Check `PINECONE_API_KEY` is set correctly
3. Verify Pinecone index exists and has data
4. Use `/api/v1/chat/query` endpoint (not `/chat`)
5. Run `python test_openai_setup.py` to diagnose

### Issue: "No documents retrieved"

**Solution:**
1. Ingest data first: `cd data-pipeline && quick_ingest.bat`
2. Verify Pinecone has vectors:
   ```python
   from pinecone import Pinecone
   pc = Pinecone(api_key="your-key")
   index = pc.Index("medresearch-ai")
   print(index.describe_index_stats())
   ```

### Issue: "Pinecone dimension mismatch"

**Solution:**
- text-embedding-3-small requires 1536 dimensions
- text-embedding-3-large requires 3072 dimensions
- Delete and recreate index with correct dimension

### Issue: "Redis authentication failed"

**Solution:**
1. If you have Redis with password:
   ```env
   REDIS_URL=redis://:your-password@host:port/0
   ```
2. If you don't have Redis or want to disable caching:
   ```env
   REDIS_URL=
   ```
3. The system works fine without Redis (caching is optional)

### Issue: "OpenAI API error"

**Solution:**
1. Check API key is valid at https://platform.openai.com/api-keys
2. Ensure you have credits in your account
3. Try `gpt-4o-mini` instead of `gpt-4o` for testing

### Issue: "Database connection failed"

**Solution:**
1. Check PostgreSQL is running
2. Verify `DATABASE_URL` in `.env` is correct
3. Run `alembic upgrade head` to initialize database

---

## Cost Estimation

### For 100 Demo Queries

**Using gpt-4o-mini + text-embedding-3-small:**
- Embeddings: ~$0.001
- Generation: ~$0.12
- Pinecone: ~$0.10/month
- **Total: ~$0.23**

**Using gpt-4o + text-embedding-3-small:**
- Embeddings: ~$0.001
- Generation: ~$0.50
- Pinecone: ~$0.10/month
- **Total: ~$0.61**

---

## Data Sources

### PubMed
- **API:** E-utilities
- **Rate Limit:** 3 requests/sec (10 with API key)
- **Data:** Research papers, abstracts, citations

### ClinicalTrials.gov
- **API:** API v2
- **Rate Limit:** 1 request/sec
- **Data:** Clinical trials, phases, outcomes

### bioRxiv/medRxiv
- **API:** bioRxiv API
- **Rate Limit:** 1 request/sec
- **Data:** Preprints, recent research

---

## Production Deployment

### Environment Variables

Set these in production:

```env
# Security
SKIP_AUTH=False
SECRET_KEY=generate-new-secret-key

# Database
DATABASE_URL=your-production-database-url

# API Keys
OPENAI_API_KEY=your-production-key
PINECONE_API_KEY=your-production-key

# CORS
BACKEND_CORS_ORIGINS=["https://your-domain.com"]
```

### Recommended Setup

1. **Backend:** Deploy to AWS ECS, Heroku, or Render
2. **Frontend:** Deploy to Vercel, Netlify, or AWS S3+CloudFront
3. **Database:** Use managed PostgreSQL (AWS RDS, Render, etc.)
4. **Redis:** Use managed Redis (AWS ElastiCache, Redis Cloud)
5. **Monitoring:** Add Sentry, DataDog, or CloudWatch

---

## Quick Reference

### Start Backend
```bash
cd backend && venv\Scripts\activate && uvicorn app.main:app --reload
```

### Start Frontend
```bash
cd frontend && npm run dev
```

### Ingest Data
```bash
cd data-pipeline && quick_ingest.bat
```

### Test Setup
```bash
cd backend && python test_openai_setup.py
```

### Check Pinecone
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
index = pc.Index("medresearch-ai")
print(index.describe_index_stats())
```

---

## Support

- **OpenAI:** https://platform.openai.com/docs
- **Pinecone:** https://docs.pinecone.io/
- **FastAPI:** https://fastapi.tiangolo.com/
- **React:** https://react.dev/

---

## Summary

1. ✅ Setup backend with OpenAI and Pinecone
2. ✅ Create Pinecone index (1536 dimensions)
3. ✅ Ingest sample data (30-50 documents)
4. ✅ Test RAG pipeline
5. ✅ Start frontend
6. ✅ Test end-to-end

**Redis is optional** - leave `REDIS_URL` empty if you don't have it or it requires authentication you don't have.

The system will work perfectly without caching!
