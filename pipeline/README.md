# Data Pipeline - Ingestion & Embedding

This directory contains scripts for fetching research papers, chunking them, embedding them, and uploading to Pinecone.

## Structure

```
data-pipeline/
├── ingest/                    # Data ingestion scripts
│   ├── pubmed_ingest.py       # Fetch from PubMed
│   ├── clinicaltrials_ingest.py  # Fetch from ClinicalTrials.gov
│   └── biorxiv_ingest.py      # Fetch from bioRxiv/medRxiv
├── embed/                     # Embedding scripts
│   └── embed_documents.py     # Embed and upload to Pinecone
├── utils/                     # Shared utilities
│   ├── pubmed_fetcher.py
│   ├── clinicaltrials_fetcher.py
│   ├── biorxiv_fetcher.py
│   └── document_chunker.py
├── config/                    # Configuration files
├── .env                       # Environment variables
└── requirements.txt           # Python dependencies
```

## Setup

### 1. Create Virtual Environment

```bash
cd data-pipeline
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
copy .env.example .env  # Windows
# cp .env.example .env  # Mac/Linux
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=medresearch-ai
PUBMED_EMAIL=your-email@example.com
```

### 4. Create Pinecone Index

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
pc.create_index(
    name="medresearch-ai",
    dimension=1536,  # for text-embedding-3-small
    metric="cosine"
)
```

## Usage

### Ingest from PubMed

```bash
python ingest/pubmed_ingest.py --query "aspirin cardiovascular" --max-results 100
```

Options:
- `--query`: Search query (required)
- `--max-results`: Maximum number of papers (default: 100)
- `--email`: Your email for PubMed API (default: from .env)

### Ingest from ClinicalTrials.gov

```bash
python ingest/clinicaltrials_ingest.py --query "diabetes" --max-results 50 --phase PHASE3
```

Options:
- `--query`: Search query (required)
- `--max-results`: Maximum number of trials (default: 100)
- `--phase`: Trial phase filter (optional, e.g., PHASE3)

### Ingest from bioRxiv/medRxiv

```bash
python ingest/biorxiv_ingest.py --server medrxiv --days 30 --max-results 50
```

Options:
- `--server`: "biorxiv" or "medrxiv" (default: medrxiv)
- `--days`: Number of days back to fetch (default: 30)
- `--max-results`: Maximum number of preprints (default: 100)

## How It Works

### 1. Fetch Data

Each ingestion script fetches data from its respective source:
- **PubMed**: Uses E-utilities API to search and fetch papers
- **ClinicalTrials.gov**: Uses API v2 to search trials
- **bioRxiv/medRxiv**: Uses API to fetch recent preprints

### 2. Chunk Documents

Documents are chunked into smaller pieces for better embedding:
- **Chunk Size**: 512 tokens (approximate)
- **Overlap**: 50 tokens between chunks
- **Method**: Semantic chunking (splits on paragraphs/sentences)

### 3. Generate Embeddings

Uses OpenAI's embedding models:
- **Model**: text-embedding-3-small (1536 dimensions)
- **Alternative**: text-embedding-3-large (3072 dimensions)
- **Batch Processing**: 100 documents at a time

### 4. Upload to Pinecone

Vectors are uploaded with metadata:
- **ID**: Unique identifier (e.g., PMID:12345_chunk_0)
- **Vector**: Embedding values
- **Metadata**: Title, source, URL, dates, etc.

## Examples

### Example 1: Ingest Aspirin Research

```bash
# Fetch 50 PubMed papers about aspirin
python ingest/pubmed_ingest.py --query "aspirin" --max-results 50

# Fetch 20 clinical trials about aspirin
python ingest/clinicaltrials_ingest.py --query "aspirin" --max-results 20
```

### Example 2: Ingest Diabetes Research

```bash
# Fetch recent diabetes papers
python ingest/pubmed_ingest.py --query "diabetes treatment" --max-results 100

# Fetch Phase 3 diabetes trials
python ingest/clinicaltrials_ingest.py --query "diabetes" --max-results 50 --phase PHASE3

# Fetch recent diabetes preprints
python ingest/biorxiv_ingest.py --server medrxiv --days 60 --max-results 30
```

### Example 3: Ingest Cancer Research

```bash
# Fetch cancer immunotherapy papers
python ingest/pubmed_ingest.py --query "cancer immunotherapy" --max-results 100

# Fetch cancer trials
python ingest/clinicaltrials_ingest.py --query "cancer" --max-results 100
```

## Testing

Test the embedding functionality:

```bash
cd embed
python embed_documents.py
```

This will:
1. Create a test document
2. Generate embedding
3. Upload to Pinecone
4. Verify success

## Monitoring

Check Pinecone index stats:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("medresearch-ai")
stats = index.describe_index_stats()

print(f"Total vectors: {stats['total_vector_count']}")
print(f"Dimension: {stats['dimension']}")
```

## Cost Estimation

### OpenAI Embeddings

- **text-embedding-3-small**: $0.02 per 1M tokens
- **Average document**: ~500 tokens
- **100 documents**: ~50K tokens = $0.001

### Pinecone

- **Serverless**: Pay per request and storage
- **Pod-based**: Fixed monthly cost
- **100 documents**: ~300 vectors = minimal cost

### Example: 1000 Documents

- Embeddings: ~$0.01
- Pinecone: ~$0.10/month
- **Total**: ~$0.11

## Troubleshooting

### "OPENAI_API_KEY not set"

Add your OpenAI API key to `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
```

### "PINECONE_API_KEY not set"

Add your Pinecone API key to `.env`:
```env
PINECONE_API_KEY=your-key-here
```

### "Index not found"

Create the Pinecone index first:
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
pc.create_index(name="medresearch-ai", dimension=1536, metric="cosine")
```

### "Dimension mismatch"

Ensure your Pinecone index dimension matches your embedding model:
- text-embedding-3-small: 1536 dimensions
- text-embedding-3-large: 3072 dimensions

### "Rate limit exceeded"

- PubMed: Add API key to increase limit
- OpenAI: Wait or upgrade tier
- ClinicalTrials.gov: Reduce batch size

## Best Practices

1. **Start Small**: Test with 10-20 documents first
2. **Use Specific Queries**: More specific = better results
3. **Monitor Costs**: Check OpenAI usage dashboard
4. **Batch Processing**: Process in batches to avoid timeouts
5. **Error Handling**: Scripts continue on errors, check logs

## Next Steps

1. ✅ Set up environment and API keys
2. ✅ Create Pinecone index
3. ✅ Test with small dataset (10-20 documents)
4. ✅ Ingest larger datasets
5. ✅ Verify data in Pinecone
6. ✅ Test RAG pipeline with backend

## Support

- OpenAI: https://platform.openai.com/docs
- Pinecone: https://docs.pinecone.io/
- PubMed API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- ClinicalTrials.gov API: https://clinicaltrials.gov/data-api/api
