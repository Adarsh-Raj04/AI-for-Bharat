# Backend - How It Works

Complete explanation of the MedResearch AI backend architecture and workflow.

---

## Directory Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/    # API route handlers
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── chat.py          # Chat/query endpoints
│   │   └── sessions.py      # Session management
│   │
│   ├── core/                # Core functionality
│   │   ├── auth.py          # JWT authentication logic
│   │   ├── config.py        # Configuration settings
│   │   ├── database.py      # Database connection
│   │   └── rate_limit.py    # Rate limiting
│   │
│   ├── models/              # Database models (SQLAlchemy)
│   │   ├── user.py          # User model
│   │   ├── session.py       # Chat session model
│   │   ├── message.py       # Message model
│   │   └── disclaimer.py    # Disclaimer acceptance
│   │
│   ├── schemas/             # Request/Response schemas (Pydantic)
│   │   ├── auth.py          # Auth schemas
│   │   ├── chat.py          # Chat schemas
│   │   └── session.py       # Session schemas
│   │
│   ├── services/            # Business logic (RAG pipeline)
│   │   ├── openai_service.py           # OpenAI GPT integration
│   │   ├── openai_embedding_service.py # OpenAI embeddings
│   │   ├── pinecone_service.py         # Vector search
│   │   ├── cache_service.py            # Redis caching
│   │   ├── reranker_service.py         # Document re-ranking
│   │   ├── intent_classifier.py        # Query classification
│   │   ├── bias_detector.py            # Source bias analysis
│   │   ├── safety_guardrails.py        # Medical advice blocking
│   │   ├── observability.py            # LangSmith logging
│   │   └── rag_pipeline.py             # Main RAG orchestrator
│   │
│   └── main.py              # FastAPI application entry point
│
├── alembic/                 # Database migrations
│   ├── versions/            # Migration scripts
│   └── env.py               # Alembic configuration
│
├── .env                     # Environment variables (your config)
├── alembic.ini              # Alembic settings
├── requirements.txt         # Python dependencies
└── test_openai_setup.py     # Setup verification script
```

---

## How a Query Flows Through the System

### Step 1: User Sends Query

**Frontend → Backend**

```
POST /api/v1/chat/query
{
  "message": "What are the side effects of aspirin?",
  "session_id": "optional-session-id"
}
```

**Handler:** `app/api/v1/endpoints/chat.py` → `query()` function

---

### Step 2: Authentication & Rate Limiting

**Authentication** (`app/core/auth.py`):
- Validates JWT token from Auth0
- Extracts user information
- Creates/retrieves user from database

**Rate Limiting** (`app/core/rate_limit.py`):
- Limits to 20 requests per minute per user
- Prevents abuse

---

### Step 3: Session Management

**Database Operations** (`app/models/`):
- Gets or creates chat session
- Saves user message to database
- Links message to session and user

---

### Step 4: RAG Pipeline Processing

**Main Orchestrator:** `app/services/rag_pipeline.py` → `process_query()`

The RAG pipeline runs through these steps:

#### 4.1 Safety Check (`safety_guardrails.py`)

**Purpose:** Block unsafe medical advice queries

**Checks for:**
- Diagnostic requests: "Do I have...", "Am I..."
- Treatment recommendations: "Should I take...", "What should I use..."
- Personal medical advice: "I have...", "My doctor said..."

**If unsafe:**
```python
return {
    "text": "I cannot provide personal medical advice...",
    "safety_blocked": True,
    "block_reason": "diagnostic_request"
}
```

**If safe:** Continue to next step

---

#### 4.2 Intent Classification (`intent_classifier.py`)

**Purpose:** Understand what type of query this is

**Intent Types:**
- `SUMMARIZATION` - "Summarize this paper..."
- `COMPARISON` - "Compare drug A vs drug B..."
- `FACTUAL_LOOKUP` - "What are the side effects..."
- `REGULATORY_COMPLIANCE` - "What are FDA requirements..."
- `ADVERSE_EVENTS` - "What are the risks..."
- `CLINICAL_TRIAL` - "Show me trials for..."
- `DRUG_INTERACTION` - "Can I take X with Y..."
- `GENERAL_QA` - General questions

**Output:**
```python
{
    "intent": "FACTUAL_LOOKUP",
    "confidence": 0.92,
    "routing_config": {
        "top_k": 10,
        "rerank_top_k": 5,
        "max_tokens": 2000,
        "temperature": 0.3
    }
}
```

Each intent has optimized retrieval settings.

---

#### 4.3 Query Embedding (`openai_embedding_service.py`)

**Purpose:** Convert text query to vector for similarity search

**Process:**
1. Check Redis cache for existing embedding
2. If not cached, call OpenAI API:
   ```python
   response = openai_client.embeddings.create(
       model="text-embedding-3-small",
       input="What are the side effects of aspirin?"
   )
   embedding = response.data[0].embedding  # 1536 dimensions
   ```
3. Cache embedding in Redis for future use

**Output:** `[0.123, -0.456, 0.789, ...]` (1536 numbers)

---

#### 4.4 Vector Search (`pinecone_service.py`)

**Purpose:** Find relevant research papers

**Process:**
1. Search Pinecone index with query embedding
2. Hybrid search (vector + keyword)
3. Retrieve top K documents (based on intent)

**Query to Pinecone:**
```python
results = index.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True
)
```

**Retrieved Documents:**
```python
[
    {
        "id": "PMID:12345_chunk_0",
        "score": 0.89,
        "metadata": {
            "title": "Aspirin Safety Study",
            "text": "Aspirin may cause stomach bleeding...",
            "source_type": "pubmed",
            "doi": "10.1234/example",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"
        }
    },
    # ... more documents
]
```

---

#### 4.5 Re-ranking (`reranker_service.py`)

**Purpose:** Improve relevance by re-scoring documents

**Process:**
1. Takes top 10 documents from Pinecone
2. Calculates more sophisticated relevance scores
3. Considers:
   - Semantic similarity
   - Keyword overlap
   - Document quality indicators
4. Returns top 5 most relevant

**Output:** Sorted list of 5 best documents

---

#### 4.6 Bias Detection (`bias_detector.py`)

**Purpose:** Analyze source diversity and potential bias

**Checks:**
1. **Industry Sponsorship:** >70% industry-funded = high bias
2. **Recency Bias:** All papers from same year
3. **Source Diversity:** Shannon entropy of sources
4. **Geographic Bias:** >80% from single region

**Output:**
```python
{
    "has_bias": True,
    "bias_score": 0.65,
    "bias_flags": [
        {
            "type": "industry_sponsorship",
            "severity": "medium",
            "description": "60% of sources are industry-sponsored"
        }
    ],
    "recommendations": [
        "Consider additional independent research sources"
    ]
}
```

---

#### 4.7 Response Generation (`openai_service.py`)

**Purpose:** Generate evidence-based answer using GPT

**Process:**

1. **Build Prompt:**
   ```python
   system_prompt = """You are MedResearch AI, an expert medical research assistant.
   Answer using ONLY the provided context documents.
   Cite sources using [1], [2] notation."""
   
   user_prompt = f"""
   CONTEXT DOCUMENTS:
   [1] PMID:12345 - Aspirin Safety Study
   Content: Aspirin may cause stomach bleeding...
   
   [2] NCT:67890 - Aspirin Clinical Trial
   Content: Common side effects include headache...
   
   USER QUESTION:
   What are the side effects of aspirin?
   
   Please provide your answer:
   """
   ```

2. **Call OpenAI API:**
   ```python
   response = openai_client.chat.completions.create(
       model="gpt-4o",
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": user_prompt}
       ],
       max_tokens=2000,
       temperature=0.3
   )
   ```

3. **Extract Response:**
   ```python
   text = response.choices[0].message.content
   tokens_used = response.usage.total_tokens
   ```

**Generated Response:**
```
Based on the research papers, aspirin has several documented side effects:

**Common Side Effects [1]:**
- Stomach bleeding and ulcers
- Heartburn and indigestion
- Nausea

**Less Common Side Effects [2]:**
- Headache (12% of patients)
- Dizziness
- Allergic reactions

**Serious Side Effects [1]:**
- Gastrointestinal bleeding (requires immediate medical attention)
- Increased bleeding risk during surgery

Always consult your healthcare provider before starting aspirin therapy [1,2].
```

---

#### 4.8 Confidence Scoring

**Purpose:** Calculate reliability of the answer

**Formula:**
```python
confidence = (
    avg_relevance_score * 0.5 +      # 50% weight on document relevance
    intent_confidence * 0.3 +         # 30% weight on intent classification
    (1.0 - bias_penalty) * 0.2        # 20% weight on bias (inverted)
)
```

**Example:**
- Average relevance: 0.85
- Intent confidence: 0.92
- Bias score: 0.30 (low bias)

```python
confidence = (0.85 * 0.5) + (0.92 * 0.3) + (0.70 * 0.2)
           = 0.425 + 0.276 + 0.140
           = 0.841  # 84.1% confidence
```

**If confidence < 0.7:** Flag for human review

---

#### 4.9 Build Citations

**Purpose:** Provide source references

**Output:**
```python
[
    {
        "number": 1,
        "source_id": "PMID:12345",
        "source_type": "pubmed",
        "title": "Aspirin Safety Study",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        "relevance_score": 0.89
    },
    {
        "number": 2,
        "source_id": "NCT:67890",
        "source_type": "clinical_trial",
        "title": "Aspirin Clinical Trial",
        "url": "https://clinicaltrials.gov/study/NCT67890",
        "relevance_score": 0.82
    }
]
```

---

#### 4.10 Observability Logging (`observability.py`)

**Purpose:** Track everything for debugging and monitoring

**Logs to LangSmith:**
1. Query received
2. Safety check result
3. Intent classification
4. Documents retrieved
5. Bias analysis
6. Response generated
7. Confidence score
8. Total processing time

---

### Step 5: Save Response to Database

**Database Operations:**
- Save assistant message to database
- Link to session and user
- Store metadata (confidence, intent, tokens used)
- Update session timestamp

---

### Step 6: Return Response to Frontend

**Response Format:**
```json
{
  "message_id": "uuid",
  "session_id": "uuid",
  "response": {
    "text": "Based on the research papers, aspirin has...",
    "citations": [
      {
        "number": 1,
        "source_id": "PMID:12345",
        "title": "Aspirin Safety Study",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"
      }
    ],
    "confidence": 0.84,
    "intent": "factual_lookup",
    "sources_used": 5,
    "requires_human_review": false,
    "safety_blocked": false
  },
  "metadata": {
    "processing_time_ms": 2500,
    "tokens_used": 1234,
    "model": "gpt-4o"
  },
  "retrieved_documents": [
    {
      "id": "PMID:12345_chunk_0",
      "title": "Aspirin Safety Study",
      "relevance_score": 0.89,
      "doi": "10.1234/example"
    }
  ],
  "bias_analysis": {
    "has_bias": false,
    "bias_score": 0.30,
    "bias_flags": []
  }
}
```

---

## Key Components Explained

### Configuration (`app/core/config.py`)

**Environment Variables:**
```python
class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # OpenAI
    USE_OPENAI: bool = True
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "medresearch-ai"
    
    # Redis (optional)
    REDIS_URL: str = ""
    
    # RAG
    ENABLE_RAG: bool = True
    TOP_K_RESULTS: int = 10
```

Loaded from `.env` file.

---

### Database Models (`app/models/`)

**User Model:**
```python
class User(Base):
    id: UUID
    auth0_id: str
    email: str
    name: str
    created_at: datetime
```

**Session Model:**
```python
class Session(Base):
    id: UUID
    user_id: UUID
    session_name: str
    total_messages: int
    created_at: datetime
    updated_at: datetime
```

**Message Model:**
```python
class Message(Base):
    id: UUID
    session_id: UUID
    role: str  # "user" or "assistant"
    content: str
    intent: str
    confidence: float
    citations: JSON
    tokens_used: int
    created_at: datetime
```

---

### Caching Strategy (`app/services/cache_service.py`)

**What's Cached:**
1. **Query Embeddings** - TTL: 1 hour
   - Key: `embedding:{md5(query)}`
   - Saves OpenAI API calls

2. **Query Results** - TTL: 30 minutes
   - Key: `results:{md5(query)}`
   - Saves entire RAG pipeline execution

**Cache Hit:**
- Instant response
- No API calls
- No cost

**Cache Miss:**
- Full RAG pipeline execution
- Result cached for next time

**If Redis unavailable:**
- System works without caching
- Slightly slower
- Higher API costs

---

## Performance Optimization

### 1. Caching
- Embeddings cached (1 hour)
- Results cached (30 minutes)
- Reduces API calls by ~60%

### 2. Batch Processing
- Pinecone queries in batches
- OpenAI embeddings in batches
- Reduces latency

### 3. Async Operations
- Database queries are async
- Multiple operations in parallel
- Faster response times

### 4. Connection Pooling
- Database connection pool
- Redis connection pool
- Reuses connections

---

## Error Handling

### Graceful Degradation

**Redis Fails:**
- System continues without caching
- Logs warning
- No user impact

**Pinecone Fails:**
- Returns fallback response
- Suggests trying again
- Logs error

**OpenAI Fails:**
- Retries with exponential backoff
- Falls back to cached response if available
- Returns error message if all fails

**Database Fails:**
- Returns error to user
- Logs critical error
- Requires manual intervention

---

## Security

### 1. Authentication
- JWT tokens from Auth0
- Token validation on every request
- User identity verified

### 2. Rate Limiting
- 20 requests per minute per user
- Prevents abuse
- Protects API costs

### 3. Input Validation
- Pydantic schemas validate all inputs
- SQL injection prevention (SQLAlchemy ORM)
- XSS prevention (sanitized outputs)

### 4. Safety Guardrails
- Blocks medical advice queries
- Prevents harmful responses
- Logs all blocked queries

---

## Monitoring & Observability

### LangSmith Integration

**Tracks:**
- Every query and response
- Processing time for each step
- API calls and costs
- Errors and failures
- User patterns

**Dashboard Shows:**
- Total queries processed
- Average response time
- Success/failure rate
- Cost per query
- Most common intents

---

## Cost Breakdown

### Per Query (with gpt-4o-mini)

**Embedding:**
- 100 tokens × $0.02 / 1M = $0.000002

**Generation:**
- 2000 tokens × $0.60 / 1M = $0.0012

**Pinecone:**
- 1 query × $0.001 = $0.001

**Total:** ~$0.0022 per query

### With Caching (60% hit rate)

**Average:** ~$0.0009 per query

---

## Development Workflow

### 1. Start Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### 2. Test Endpoint
```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the side effects of aspirin?"}'
```

### 3. Check Logs
- Console shows all operations
- LangSmith dashboard for detailed traces
- Database for stored messages

### 4. Debug
- Use `test_openai_setup.py` to verify setup
- Check `.env` configuration
- Review logs for errors

---

## Summary

The backend is a sophisticated RAG (Retrieval-Augmented Generation) system that:

1. ✅ Authenticates users with Auth0
2. ✅ Blocks unsafe medical advice queries
3. ✅ Classifies query intent
4. ✅ Converts queries to vectors
5. ✅ Searches Pinecone for relevant papers
6. ✅ Re-ranks results for relevance
7. ✅ Detects bias in sources
8. ✅ Generates evidence-based answers with GPT
9. ✅ Calculates confidence scores
10. ✅ Provides source citations
11. ✅ Caches results for performance
12. ✅ Logs everything for monitoring

**Result:** Fast, accurate, safe, and cost-effective medical research assistant!
