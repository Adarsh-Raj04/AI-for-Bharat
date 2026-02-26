# Chat Flow Diagrams

## 1. `/query` Endpoint Flow (Production RAG Only)

```
┌─────────────┐
│   Frontend  │
│  (ChatPage) │
└──────┬──────┘
       │ POST /api/v1/chat/query
       │ { message: "Summarize PMID 123", session_id: "abc" }
       │ Authorization: Bearer <token>
       ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend API                          │
│                                                         │
│  1. Authenticate User (JWT validation)                 │
│     ├─ Verify token with Auth0                         │
│     └─ Extract user info from token                    │
│                                                         │
│  2. Get/Create User in DB                              │
│     ├─ Query: SELECT * FROM users WHERE auth0_id=...   │
│     └─ If not exists: INSERT INTO users...             │
│                                                         │
│  3. Get/Create Session                                 │
│     ├─ If session_id provided: SELECT * FROM sessions  │
│     └─ Else: INSERT INTO sessions (new session)        │
│                                                         │
│  4. Save User Message                                  │
│     └─ INSERT INTO messages (role='user', content=...) │
│                                                         │
│  5. Call RAG Pipeline ⚡                                │
│     └─ rag_pipeline.process_query(message)             │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG Pipeline                           │
│                                                         │
│  1. Intent Classification                              │
│     └─ Detect: summarization, comparison, regulatory   │
│                                                         │
│  2. Vector Search (Pinecone)                           │
│     └─ Search for relevant documents                   │
│                                                         │
│  3. Document Retrieval                                 │
│     └─ Get top-k papers/trials/FDA docs                │
│                                                         │
│  4. Reranking                                          │
│     └─ Improve relevance of results                    │
│                                                         │
│  5. LLM Generation (Claude/GPT)                        │
│     └─ Generate response with citations                │
│                                                         │
│  6. Bias Detection & Safety Checks                     │
│     └─ Ensure medical safety compliance                │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Returns:
                   │ {
                   │   text: "AI response...",
                   │   citations: [...],
                   │   confidence: 0.92,
                   │   intent: "summarization"
                   │ }
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend API                          │
│                                                         │
│  6. Save AI Response                                   │
│     └─ INSERT INTO messages (role='assistant'...)      │
│                                                         │
│  7. Update Session                                     │
│     └─ UPDATE sessions SET total_messages += 2         │
│                                                         │
│  8. Return Response                                    │
│     └─ JSON with text, citations, metadata             │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────┐
│   Frontend  │
│  Display    │
│  Response   │
└─────────────┘
```

---

## 2. `/chat` Endpoint Flow (RAG with Mock Fallback)

```
┌─────────────┐
│   Frontend  │
└──────┬──────┘
       │ POST /api/v1/chat/chat
       ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend API                          │
│                                                         │
│  Steps 1-4: Same as /query                             │
│                                                         │
│  5. Smart Response Generation                          │
│     ┌─────────────────────────────────┐                │
│     │ if ENABLE_RAG == True:          │                │
│     │   ┌─────────────────────────┐   │                │
│     │   │ Try RAG Pipeline        │   │                │
│     │   │   ├─ Success → Use RAG  │   │                │
│     │   │   └─ Error → Use Mock   │   │                │
│     │   └─────────────────────────┘   │                │
│     │ else:                           │                │
│     │   └─ Use Mock Response          │                │
│     └─────────────────────────────────┘                │
│                                                         │
└─────────────────────────────────────────────────────────┘
                   │
                   ├─ RAG Path (if enabled & working)
                   │  └─ Same as /query
                   │
                   └─ Mock Path (if disabled or error)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Mock Response Generator                    │
│                                                         │
│  Detect Intent from Keywords:                          │
│  ├─ "summarize" or "pmid" → Summarization              │
│  ├─ "compare" → Comparison                             │
│  ├─ "fda" or "regulatory" → Regulatory                 │
│  └─ Default → General Q&A                              │
│                                                         │
│  Return Pre-formatted Response:                        │
│  {                                                      │
│    text: "Mock summary with fake data...",             │
│    citations: [fake citations],                        │
│    confidence: 0.92,                                   │
│    intent: "summarization"                             │
│  }                                                      │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
              (Continue with steps 6-8)
```

---

## 3. `/chat/stream` Endpoint Flow (Streaming Response)

```
┌─────────────┐
│   Frontend  │
└──────┬──────┘
       │ POST /api/v1/chat/stream
       ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend API                          │
│                                                         │
│  Steps 1-4: Same as /query                             │
│                                                         │
│  5. Start Streaming Response                           │
│     └─ Return StreamingResponse (SSE)                  │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Server-Sent Events (SSE)
                   │
                   ├─ Event 1: Citations
                   │  data: {"type": "citations", "data": [...]}
                   │
                   ├─ Event 2: Text Chunk 1
                   │  data: {"type": "text", "data": "This is "}
                   │
                   ├─ Event 3: Text Chunk 2
                   │  data: {"type": "text", "data": "a streaming "}
                   │
                   ├─ Event 4: Text Chunk 3
                   │  data: {"type": "text", "data": "response."}
                   │
                   ├─ Event 5: Metadata
                   │  data: {"type": "metadata", "data": {...}}
                   │
                   └─ Event 6: Done
                      data: {"type": "done", "data": {"session_id": "..."}}
                   │
                   ▼
┌─────────────┐
│   Frontend  │
│  Displays   │
│  text as it │
│  arrives    │
└─────────────┘
```

---

## Database Schema Relationships

```
┌──────────────────────┐
│       users          │
│──────────────────────│
│ id (PK)              │
│ auth0_id (unique)    │
│ email (unique)       │
│ name                 │
│ created_at           │
│ updated_at           │
└──────────┬───────────┘
           │
           │ 1:many
           │
           ▼
┌──────────────────────┐
│      sessions        │
│──────────────────────│
│ id (PK)              │
│ user_id (FK)         │◄─────┐
│ session_name         │      │
│ total_messages       │      │
│ created_at           │      │
│ updated_at           │      │
└──────────┬───────────┘      │
           │                  │
           │ 1:many           │
           │                  │
           ▼                  │
┌──────────────────────┐      │
│      messages        │      │
│──────────────────────│      │
│ id (PK)              │      │
│ session_id (FK)      │──────┘
│ role (user/assistant)│
│ content              │
│ intent               │
│ confidence           │
│ citations (JSON)     │
│ tokens_used          │
│ created_at           │
└──────────────────────┘
```

---

## Request/Response Examples

### Request (All Endpoints)
```json
POST /api/v1/chat/chat
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...

{
  "message": "Summarize PMID 33301246",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "stream": false
}
```

### Response (`/query` and `/chat`)
```json
{
  "message_id": 123,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "response": {
    "text": "This study evaluated the efficacy of...",
    "citations": [
      {
        "number": 1,
        "source_id": "PMID:33301246",
        "source_type": "pubmed",
        "title": "mRNA Vaccine Efficacy Study",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33301246/",
        "relevance_score": 0.95
      }
    ],
    "confidence": 0.92,
    "intent": "summarization",
    "sources_used": 5,
    "requires_human_review": false,
    "safety_blocked": false
  },
  "metadata": {
    "processing_time_ms": 1234,
    "tokens_used": 567,
    "model": "claude-sonnet-4"
  },
  "timestamp": "2026-02-27T10:30:00Z",
  "retrieved_documents": [...],
  "bias_analysis": {...},
  "requires_human_review": false,
  "safety_blocked": false
}
```

### Response (`/chat/stream`)
```
data: {"type": "citations", "data": [{"number": 1, "title": "..."}]}

data: {"type": "text", "data": "This study "}

data: {"type": "text", "data": "evaluated the "}

data: {"type": "text", "data": "efficacy of..."}

data: {"type": "metadata", "data": {"confidence": 0.92, "intent": "summarization"}}

data: {"type": "done", "data": {"session_id": "550e8400-e29b-41d4-a716-446655440000"}}
```

---

## Decision Tree: Which Endpoint to Use?

```
Do you need streaming response (ChatGPT-like typing)?
│
├─ YES → Use /chat/stream
│         └─ Implement EventSource in frontend
│
└─ NO → Do you have RAG fully configured?
        │
        ├─ YES → Use /query
        │         └─ Production-ready, no fallback
        │
        └─ NO/UNSURE → Use /chat
                       └─ Works with or without RAG
```

---

## Current Frontend Usage

```javascript
// ChatPage.jsx currently uses /chat endpoint
const response = await api.post('/chat/chat', {
  session_id: sessionId,
  message: input,
  stream: false
})

// Could be upgraded to streaming:
const eventSource = new EventSource(
  `/api/v1/chat/stream?message=${input}&session_id=${sessionId}`
)
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data)
  // Handle different event types
}
```

---

## Summary

| Endpoint | Purpose | RAG | Mock | Streaming | Frontend Uses |
|----------|---------|-----|------|-----------|---------------|
| `/query` | Production | ✅ Required | ❌ No | ❌ No | ❌ No |
| `/chat` | Flexible | ✅ Optional | ✅ Yes | ❌ No | ✅ Yes |
| `/chat/stream` | Real-time UX | ✅ Optional | ✅ Yes | ✅ Yes | ❌ No |
