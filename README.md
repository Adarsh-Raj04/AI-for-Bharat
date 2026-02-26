# MedResearch AI

Conversational AI Research Assistant for Life Sciences - Built for AWS AI for Bharat Hackathon

## Overview

MedResearch AI helps researchers and scientists in healthcare and life sciences efficiently summarize clinical trials, research papers, and regulatory documents using RAG (Retrieval-Augmented Generation) with publicly available data sources.

## Features

✅ **AI-Powered Chat** - Ask questions about medical research  
✅ **RAG Pipeline** - Retrieves relevant papers and generates evidence-based answers  
✅ **Multiple Data Sources** - PubMed, ClinicalTrials.gov, bioRxiv/medRxiv  
✅ **Safety Guardrails** - Blocks unsafe medical advice queries  
✅ **Source Citations** - Every answer includes DOI links and references  
✅ **Bias Detection** - Analyzes source diversity and industry sponsorship  
✅ **Confidence Scoring** - Flags low-confidence responses for review  

## Tech Stack

- **Frontend:** React + Vite + Tailwind CSS
- **Backend:** FastAPI + PostgreSQL
- **AI:** OpenAI GPT-4o + text-embedding-3
- **Vector DB:** Pinecone
- **Cache:** Redis (optional)
- **Auth:** Auth0

## Quick Start

### 1. Backend Setup (2 minutes)

```bash
cd backend
setup_openai.bat
# Edit .env with your OpenAI and Pinecone keys
alembic upgrade head
uvicorn app.main:app --reload
```

### 2. Ingest Data (2 minutes)

```bash
cd data-pipeline
quick_ingest.bat
# Choose option 4 for quick demo
```

### 3. Frontend Setup (1 minute)

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 and start chatting!

## Documentation

- **[SETUP.md](SETUP.md)** - Complete technical setup guide
- **[design.md](design.md)** - System architecture and design
- **[requirements.md](requirements.md)** - Functional requirements

## Project Structure

```
medresearch-ai/
├── frontend/           # React UI (Port 5173)
├── backend/            # FastAPI Server (Port 8000)
├── data-pipeline/      # Data Ingestion Scripts
├── design.md           # System design
├── requirements.md     # Requirements
├── SETUP.md            # Setup guide
└── README.md           # This file
```

## API Endpoints

- `POST /api/v1/chat/query` - Submit query to RAG pipeline
- `GET /api/v1/sessions` - List chat sessions
- `GET /api/v1/auth/me` - Get current user

Full API docs: http://localhost:8000/docs

## Configuration

### Backend (.env)

```env
USE_OPENAI=True
OPENAI_API_KEY=sk-your-key
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=medresearch-ai
ENABLE_RAG=True
REDIS_URL=  # Optional - leave empty to disable caching
```

### Data Pipeline (.env)

```env
OPENAI_API_KEY=sk-your-key
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=medresearch-ai
```

## Data Ingestion

```bash
cd data-pipeline

# PubMed
python ingest/pubmed_ingest.py --query "aspirin" --max-results 50

# ClinicalTrials.gov
python ingest/clinicaltrials_ingest.py --query "diabetes" --max-results 50

# bioRxiv/medRxiv
python ingest/biorxiv_ingest.py --server medrxiv --days 30 --max-results 50
```

## Cost Estimate

For 100 queries with gpt-4o-mini:
- Embeddings: ~$0.001
- Generation: ~$0.12
- Pinecone: ~$0.10/month
- **Total: ~$0.23**

## Troubleshooting

### Getting dummy responses?

1. Check `ENABLE_RAG=True` in `backend/.env`
2. Check Pinecone API key is set
3. Verify Pinecone index exists and has data
4. Use `/api/v1/chat/query` endpoint
5. Run `cd backend && python test_openai_setup.py`

### Redis authentication failed?

Redis is optional. Leave `REDIS_URL` empty in `.env` to disable caching.

### No documents retrieved?

Ingest data first: `cd data-pipeline && quick_ingest.bat`

See [SETUP.md](SETUP.md) for detailed troubleshooting.

## Team

MedResearch AI Team - AWS AI for Bharat Hackathon

## License

MIT License

## Disclaimer

⚠️ This tool is for research purposes only. Not medical advice. Always verify information with qualified healthcare professionals.
