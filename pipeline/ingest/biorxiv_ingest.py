"""
bioRxiv/medRxiv Data Ingestion Script
Fetches preprints, chunks them, embeds them, and uploads to Pinecone
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dotenv import load_dotenv
import logging
from utils.biorxiv_fetcher import BioRxivFetcher
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


def ingest_biorxiv(server: str = "medrxiv", days: int = 30, max_results: int = 100):
    """
    Ingest bioRxiv/medRxiv preprints
    
    Args:
        server: "biorxiv" or "medrxiv"
        days: Number of days back to fetch
        max_results: Maximum number of preprints to fetch
    """
    logger.info(f"Starting {server} ingestion")
    logger.info(f"Fetching last {days} days, Max results: {max_results}")
    
    # Step 1: Fetch preprints
    logger.info(f"Step 1: Fetching preprints from {server}...")
    fetcher = BioRxivFetcher(server=server)
    preprints = fetcher.fetch_recent(days=days, max_results=max_results)
    
    if not preprints:
        logger.error("No preprints fetched. Exiting.")
        return
    
    logger.info(f"Fetched {len(preprints)} preprints")
    
    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    
    all_chunks = []
    for preprint in preprints:
        chunks = chunker.chunk_document(preprint)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(preprints)} preprints")
    
    # Step 3: Embed and upload to Pinecone
    logger.info("Step 3: Embedding and uploading to Pinecone...")
    success_count = embed_and_upload(all_chunks, batch_size=100)
    
    logger.info(f"Successfully uploaded {success_count}/{len(all_chunks)} chunks")
    logger.info(f"{server} ingestion complete!")


def main():
    parser = argparse.ArgumentParser(description="Ingest bioRxiv/medRxiv preprints")
    parser.add_argument("--server", type=str, default="medrxiv", choices=["biorxiv", "medrxiv"], help="Server to fetch from")
    parser.add_argument("--days", type=int, default=30, help="Number of days back to fetch")
    parser.add_argument("--max-results", type=int, default=100, help="Maximum number of results")
    
    args = parser.parse_args()
    
    ingest_biorxiv(
        server=args.server,
        days=args.days,
        max_results=args.max_results
    )


if __name__ == "__main__":
    main()
