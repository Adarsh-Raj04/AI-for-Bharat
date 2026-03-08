"""
ClinicalTrials.gov Data Ingestion Script
Fetches clinical trials, chunks them, embeds them, and uploads to Pinecone
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dotenv import load_dotenv
import logging
from utils.clinicaltrials_fetcher import ClinicalTrialsFetcher
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


def ingest_clinical_trials(query: str, max_results: int = 100, phase: str = None):
    """
    Ingest clinical trials
    
    Args:
        query: Search query
        max_results: Maximum number of trials to fetch
        phase: Trial phase filter (e.g., "PHASE3")
    """
    logger.info(f"Starting ClinicalTrials.gov ingestion for query: '{query}'")
    logger.info(f"Max results: {max_results}, Phase: {phase or 'All'}")
    
    # Step 1: Fetch trials
    logger.info("Step 1: Fetching clinical trials...")
    fetcher = ClinicalTrialsFetcher()
    trials = fetcher.search(query, max_results=max_results, phase=phase)
    
    if not trials:
        logger.error("No trials fetched. Exiting.")
        return
    
    logger.info(f"Fetched {len(trials)} trials")
    
    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    
    all_chunks = []
    for trial in trials:
        chunks = chunker.chunk_document(trial)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(trials)} trials")
    
    # Step 3: Embed and upload to Pinecone
    logger.info("Step 3: Embedding and uploading to Pinecone...")
    success_count = embed_and_upload(all_chunks, batch_size=100)
    
    logger.info(f"Successfully uploaded {success_count}/{len(all_chunks)} chunks")
    logger.info("ClinicalTrials.gov ingestion complete!")


def main():
    parser = argparse.ArgumentParser(description="Ingest ClinicalTrials.gov data")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--max-results", type=int, default=100, help="Maximum number of results")
    parser.add_argument("--phase", type=str, default=None, help="Trial phase (e.g., PHASE3)")
    
    args = parser.parse_args()
    
    ingest_clinical_trials(
        query=args.query,
        max_results=args.max_results,
        phase=args.phase
    )


if __name__ == "__main__":
    main()
