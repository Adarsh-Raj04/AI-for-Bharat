"""
PubMed Data Ingestion Script
Fetches research papers from PubMed, chunks them, embeds them, and uploads to Pinecone
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dotenv import load_dotenv
import logging
from typing import List, Dict
from utils.pubmed_fetcher import PubMedFetcher
from utils.document_chunker import DocumentChunker
from embed.embed_documents import embed_and_upload

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_pubmed(query: str, max_results: int = 100, email: str = "your-email@example.com"):
    """
    Ingest PubMed papers
    
    Args:
        query: Search query
        max_results: Maximum number of papers to fetch
        email: Your email for PubMed API
    """
    logger.info(f"Starting PubMed ingestion for query: '{query}'")
    logger.info(f"Max results: {max_results}")
    
    # Step 1: Fetch papers from PubMed
    logger.info("Step 1: Fetching papers from PubMed...")
    fetcher = PubMedFetcher(email=email)
    papers = fetcher.fetch_by_query(query, max_results=max_results)
    
    if not papers:
        logger.error("No papers fetched. Exiting.")
        return
    
    logger.info(f"Fetched {len(papers)} papers")
    
    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    
    all_chunks = []
    for paper in papers:
        chunks = chunker.chunk_document(paper)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(papers)} papers")
    
    # Step 3: Embed and upload to Pinecone
    logger.info("Step 3: Embedding and uploading to Pinecone...")
    success_count = embed_and_upload(all_chunks, batch_size=100)
    
    logger.info(f"Successfully uploaded {success_count}/{len(all_chunks)} chunks")
    logger.info("PubMed ingestion complete!")


def main():
    parser = argparse.ArgumentParser(description="Ingest PubMed research papers")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--max-results", type=int, default=100, help="Maximum number of results")
    parser.add_argument("--email", type=str, default="your-email@example.com", help="Your email for PubMed API")
    
    args = parser.parse_args()
    
    ingest_pubmed(
        query=args.query,
        max_results=args.max_results,
        email=args.email
    )


if __name__ == "__main__":
    main()
