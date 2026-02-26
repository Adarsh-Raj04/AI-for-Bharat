# Chat Endpoints Explained

## Overview

The `chat.py` file contains three endpoints that handle user queries and AI responses. They all do similar things but with different behaviors based on configuration and response format.

---

## 1. `/query` Endpoint - Production RAG Only

```python
@router.post("/query", response_model=ChatResponse)
```

### Purpose
This is the **production endpoint** that ALWAYS uses the RAG (Retrieval-Augmented Generation) pipeline. No fallback to mock responses.

### What It Does

1. **User Management**
   - Gets or creates the user in the database based on Auth0 ID
   - Links the user to their chat session

2. **Session Management**
   - If `session_id` is provided, retrieves existing session
   - If no `session_id`, creates a new session with timestamp name
   - Validates that the session belongs to the authenticated user

3. **Message Storage**
   - Saves the user's message to the database
   - Links it to the session

4. **RAG Pipeline Processing**
   - Calls `rag_pipeline.process_query()` to:
     - Search Pinecone vector database for relevant documents
     - Retrieve research papers, clinical trials, FDA docs
     - Use AI (Claude/GPT) to generate response with citations
     - Detect intent (summarization, comparison, regulatory, etc.)
     - Calculate confidence score
     - Run bias detection and safety checks

5. **Response Storage**
   - Saves AI response to database with:
     - Generated text
     - Citations (sources used)
     - Intent classification
     - Confidence score
     - Token usage
   - Updates session metadata (message count, timestamp)

6. **Return Response**
   - Returns structured JSON with:
     - AI-generated text
     - Citations with source links
     - Confidence score
     - Intent classification
     - Processing metadata
     - Bias analysis
     - Safety flags

### When to Use
- **Production environment** with RAG fully configured
- When you want to ensure real AI responses (no mocks)
- When ENABLE_RAG=True and all services are running

### Error Handling
- Throws 500 error if RAG pipeline fails
- No fallback to mock responses

---

## 2. `/chat` Endpoint - RAG with Mock Fallback

```python
@router.post("/chat", response_model=ChatResponse)
```

### Purpose
This is the **flexible endpoint** that uses RAG if available, but falls back to mock responses for development/testing.

### What It Does

Same as `/query` endpoint for steps 1-3, then:

4. **Smart Response Generation**
   ```python
   if settings.ENABLE_RAG:
       # Try to use RAG pipeline
       try:
           ai_response = rag_pipeline.process_query(...)
       except Exception as e:
           # Fallback to mock on error
           ai_response = generate_mock_response(...)
   else:
       # Use mock if RAG disabled
       ai_response = generate_mock_response(...)
   ```

5. **Mock Response Generator**
   - Detects query intent from keywords:
     - "summarize" or "pmid" → Summarization intent
     - "compare" → Comparison intent
     - "fda" or "regulatory" → Compliance/regulatory intent
     - Default → General Q&A
   - Returns pre-formatted responses with fake citations
   - Useful for frontend development without backend setup

6-7. Same as `/query` endpoint

### When to Use
- **Development environment** where RAG might not be fully set up
- **Testing** the frontend without needing Pinecone/Claude/etc.
- **Graceful degradation** - works even if RAG services are down
- This is what the frontend currently uses

### Mock Response Examples

**Summarization Query:**
```
User: "Summarize PMID 12345678"
Mock: Returns formatted summary with fake study details and citations
```

**Comparison Query:**
```
User: "Compare pembrolizumab vs nivolumab"
Mock: Returns comparison table with fake efficacy data
```

**Regulatory Query:**
```
User: "FDA accelerated approval requirements"
Mock: Returns FDA guidance summary with fake citations
```

**General Query:**
```
User: "What is immunotherapy?"
Mock: Returns explanation that this is a demo response
```

---

## 3. `/chat/stream` Endpoint - Streaming Responses

```python
@router.post("/chat/stream")
```

### Purpose
This endpoint returns responses as a **stream** (Server-Sent Events) instead of waiting for the complete response. Like ChatGPT's typing effect.

### What It Does

Same as `/chat` for steps 1-3, then:

4. **Streaming Response**
   - Returns `StreamingResponse` with `text/event-stream` media type
   - Sends data in chunks as it's generated
   - Client sees text appearing word-by-word

5. **Event Generator**
   ```python
   async def event_generator():
       # Send citations first
       yield {"type": "citations", "data": [...]}
       
       # Stream text in chunks
       for chunk in text:
           yield {"type": "text", "data": chunk}
       
       # Send metadata
       yield {"type": "metadata", "data": {...}}
       
       # Signal completion
       yield {"type": "done", "data": {...}}
   ```

6. **Event Types**
   - `citations` - List of sources used
   - `text` - Chunks of the response text
   - `metadata` - Confidence, intent, sources count
   - `done` - Signals stream completion with session_id
   - `error` - If something goes wrong

### When to Use
- **Better UX** - Users see response appearing in real-time
- **Long responses** - Don't wait for entire response to complete
- **ChatGPT-like experience** - Streaming text effect
- **Currently NOT used** by the frontend (could be implemented)

### Response Format (SSE)
```
data: {"type": "citations", "data": [{"number": 1, "title": "..."}]}

data: {"type": "text", "data": "This is a "}

data: {"type": "text", "data": "streaming "}

data: {"type": "text", "data": "response."}

data: {"type": "metadata", "data": {"confidence": 0.92}}

data: {"type": "done", "data": {"session_id": "123"}}
```

---

## Comparison Table

| Feature | `/query` | `/chat` | `/chat/stream` |
|---------|----------|---------|----------------|
| **RAG Pipeline** | Always | If enabled | If enabled |
| **Mock Fallback** | ❌ No | ✅ Yes | ✅ Yes |
| **Response Type** | Complete JSON | Complete JSON | Streaming SSE |
| **Error Handling** | Throws 500 | Falls back to mock | Falls back to mock |
| **Use Case** | Production | Development/Testing | Real-time UX |
| **Frontend Uses** | ❌ No | ✅ Yes | ❌ No (could be) |

---

## Common Flow (All Endpoints)

```
1. User sends message
   ↓
2. Authenticate user (JWT token)
   ↓
3. Get/create user in database
   ↓
4. Get/create chat session
   ↓
5. Save user message to DB
   ↓
6. Generate AI response
   - /query: RAG only
   - /chat: RAG or mock
   - /chat/stream: RAG or mock (streamed)
   ↓
7. Save AI response to DB
   ↓
8. Update session metadata
   ↓
9. Return response to frontend
```

---

## Database Operations

### Tables Used

1. **users** - Stores user profiles
   - auth0_id, email, name
   - Created on first message

2. **sessions** - Chat sessions
   - user_id, session_name, total_messages
   - One user can have multiple sessions

3. **messages** - Individual messages
   - session_id, role (user/assistant), content
   - Stores both user queries and AI responses
   - Includes citations, intent, confidence

### Relationships
```
User (1) ──→ (many) Sessions
Session (1) ──→ (many) Messages
```

---

## RAG Pipeline Integration

When `ENABLE_RAG=True`, the endpoints call:

```python
rag_pipeline = get_rag_pipeline()
ai_response = rag_pipeline.process_query(
    query=chat_request.message,
    user_id=str(user.id),
    session_id=str(session.id)
)
```

### RAG Pipeline Does:
1. **Intent Classification** - Determines query type
2. **Vector Search** - Searches Pinecone for relevant docs
3. **Document Retrieval** - Gets top-k most relevant papers/trials
4. **Reranking** - Improves relevance of retrieved docs
5. **Context Building** - Combines retrieved docs into context
6. **LLM Generation** - Uses Claude/GPT to generate response
7. **Citation Extraction** - Links response to source documents
8. **Bias Detection** - Checks for biased language
9. **Safety Checks** - Ensures medical safety guidelines

### RAG Response Structure:
```python
{
    "text": "AI-generated response...",
    "citations": [
        {
            "number": 1,
            "source_id": "PMID:12345678",
            "source_type": "pubmed",
            "title": "Study Title",
            "url": "https://...",
            "relevance_score": 0.95
        }
    ],
    "confidence": 0.92,
    "intent": "summarization",
    "sources_used": 5,
    "requires_human_review": False,
    "safety_blocked": False,
    "processing_time_ms": 1234,
    "tokens_used": 567,
    "model": "claude-sonnet-4",
    "retrieved_documents": [...],
    "bias_analysis": {...}
}
```

---

## Rate Limiting

All endpoints have rate limiting:
```python
@limiter.limit("20/minute")
```

- Each user can make 20 requests per minute
- Based on IP address or user ID
- Returns 429 error if exceeded

---

## Frontend Integration

### Current Usage (ChatPage.jsx)

```javascript
const response = await api.post('/chat/chat', 
    {
        session_id: sessionId,
        message: input,
        stream: false
    },
    {
        headers: { Authorization: `Bearer ${token}` }
    }
)
```

### Could Use Streaming:

```javascript
const eventSource = new EventSource('/api/v1/chat/stream')
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data)
    if (data.type === 'text') {
        // Append text chunk to UI
    }
}
```

---

## Configuration

### Environment Variables

```env
ENABLE_RAG=True          # Use real RAG pipeline
SKIP_AUTH=False          # Require authentication
RATE_LIMIT_PER_MINUTE=20 # Rate limit threshold
```

### When RAG is Disabled
- `/query` - Will fail (no fallback)
- `/chat` - Uses mock responses
- `/chat/stream` - Streams mock responses

---

## Summary

- **`/query`** = Production-only, RAG required, no fallback
- **`/chat`** = Flexible, RAG preferred, mock fallback (currently used)
- **`/chat/stream`** = Same as `/chat` but with streaming response

All three endpoints:
- Require authentication
- Manage users and sessions
- Store messages in database
- Return structured responses with citations
- Have rate limiting
- Support session continuity
