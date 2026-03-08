"""
Unified Data Ingestion Script
Fetches from multiple sources, chunks, embeds, and uploads to Pinecone
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from dotenv import load_dotenv
import logging
from typing import List, Dict
from utils.pubmed_fetcher import PubMedFetcher
from utils.biorxiv_fetcher import BioRxivFetcher
from utils.clinicaltrials_fetcher import ClinicalTrialsFetcher
from utils.document_chunker import DocumentChunker
from embed.embed_documents import embed_and_upload
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_from_source(
    source: str,
    query: str = None,
    max_results: int = 100,
    **kwargs
) -> tuple[int, int]:
    """
    Ingest data from a specific source
    
    Args:
        source: "pubmed", "clinicaltrials", "biorxiv", or "medrxiv"
        query: Search query (required for pubmed and clinicaltrials)
        max_results: Maximum number of documents to fetch
        **kwargs: Additional source-specific parameters
        
    Returns:
        Tuple of (documents_fetched, chunks_uploaded)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting ingestion from: {source.upper()}")
    logger.info(f"{'='*80}\n")
    
    # Step 1: Fetch documents
    logger.info("📥 Step 1: Fetching documents...")
    documents = []
    
    try:
        if source == "pubmed":
            if not query:
                raise ValueError("Query required for PubMed")
            fetcher = PubMedFetcher(email=kwargs.get("email", "your-email@example.com"))
            documents = fetcher.fetch_by_query(query, max_results=max_results)
            
        elif source == "clinicaltrials":
            if not query:
                raise ValueError("Query required for ClinicalTrials")
            fetcher = ClinicalTrialsFetcher()
            documents = fetcher.search(
                query,
                max_results=max_results,
                phase=kwargs.get("phase")
            )
            
        elif source in ["biorxiv", "medrxiv"]:
            fetcher = BioRxivFetcher(server=source)
            documents = fetcher.fetch_recent(
                days=kwargs.get("days", 30),
                max_results=max_results
            )
            
        else:
            raise ValueError(f"Unknown source: {source}")
        
        if not documents:
            logger.warning(f"⚠️  No documents fetched from {source}")
            return 0, 0
        
        logger.info(f"✅ Fetched {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch from {source}: {e}")
        return 0, 0
    
    # Step 2: Chunk documents
    logger.info("\n📄 Step 2: Chunking documents...")
    try:
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        all_chunks = []
        
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("⚠️  No chunks created")
            return len(documents), 0
        
        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        logger.info(f"   Average: {len(all_chunks)/len(documents):.1f} chunks per document")
        
    except Exception as e:
        logger.error(f"❌ Failed to chunk documents: {e}")
        return len(documents), 0
    
    # Step 3: Embed and upload
    logger.info("\n🚀 Step 3: Embedding and uploading to Pinecone...")
    try:
        success_count = embed_and_upload(all_chunks, batch_size=100)
        
        logger.info(f"✅ Successfully uploaded {success_count}/{len(all_chunks)} chunks")
        logger.info(f"   Success rate: {success_count/len(all_chunks)*100:.1f}%")
        
        return len(documents), success_count
        
    except Exception as e:
        logger.error(f"❌ Failed to embed and upload: {e}")
        return len(documents), 0


def ingest_all_sources(
    pubmed_query: str = None,
    clinical_query: str = None,
    max_results_per_source: int = 50,
    include_preprints: bool = True
):
    """
    Ingest from all available sources
    
    Args:
        pubmed_query: Query for PubMed
        clinical_query: Query for ClinicalTrials (defaults to pubmed_query)
        max_results_per_source: Max results per source
        include_preprints: Whether to include bioRxiv/medRxiv
    """
    logger.info("\n" + "="*80)
    logger.info("🌐 MULTI-SOURCE INGESTION")
    logger.info("="*80)
    
    total_docs = 0
    total_chunks = 0
    results = {}
    
    # PubMed
    if pubmed_query:
        docs, chunks = ingest_from_source(
            "pubmed",
            query=pubmed_query,
            max_results=max_results_per_source
        )
        results["PubMed"] = {"documents": docs, "chunks": chunks}
        total_docs += docs
        total_chunks += chunks
    
    # ClinicalTrials
    if clinical_query or pubmed_query:
        docs, chunks = ingest_from_source(
            "clinicaltrials",
            query=clinical_query or pubmed_query,
            max_results=max_results_per_source
        )
        results["ClinicalTrials"] = {"documents": docs, "chunks": chunks}
        total_docs += docs
        total_chunks += chunks
    
    # Preprints
    if include_preprints:
        for source in ["medrxiv", "biorxiv"]:
            docs, chunks = ingest_from_source(
                source,
                max_results=max_results_per_source,
                days=30
            )
            results[source.title()] = {"documents": docs, "chunks": chunks}
            total_docs += docs
            total_chunks += chunks
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 INGESTION SUMMARY")
    logger.info("="*80)
    
    for source, stats in results.items():
        logger.info(f"{source:15} → {stats['documents']:3} docs, {stats['chunks']:4} chunks")
    
    logger.info("-"*80)
    logger.info(f"{'TOTAL':15} → {total_docs:3} docs, {total_chunks:4} chunks")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest medical research data from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest from PubMed only
  python ingest_all.py --source pubmed --query "diabetes treatment" --max-results 100
  
  # Ingest from ClinicalTrials only
  python ingest_all.py --source clinicaltrials --query "cancer immunotherapy" --max-results 50
  
  # Ingest from medRxiv only
  python ingest_all.py --source medrxiv --days 30 --max-results 50
  
  # Ingest from all sources
  python ingest_all.py --all --query "aspirin cardiovascular" --max-results 50
        """
    )
    
    # Source selection
    parser.add_argument(
        "--source",
        type=str,
        choices=["pubmed", "clinicaltrials", "biorxiv", "medrxiv"],
        help="Single source to ingest from"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest from all sources"
    )
    
    # Query parameters
    parser.add_argument(
        "--query",
        type=str,
        help="Search query (required for pubmed and clinicaltrials)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of results per source (default: 100)"
    )
    
    # Source-specific parameters
    parser.add_argument(
        "--phase",
        type=str,
        help="Clinical trial phase filter (e.g., PHASE3)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days back for bioRxiv/medRxiv (default: 30)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="your-email@example.com",
        help="Email for PubMed API"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.source and not args.all:
        parser.error("Either --source or --all must be specified")
    
    if args.source in ["pubmed", "clinicaltrials"] and not args.query:
        parser.error(f"--query is required for {args.source}")
    
    # Execute ingestion
    start_time = datetime.now()
    
    if args.all:
        ingest_all_sources(
            pubmed_query=args.query,
            max_results_per_source=args.max_results,
            include_preprints=True
        )
    else:
        ingest_from_source(
            source=args.source,
            query=args.query,
            max_results=args.max_results,
            phase=args.phase,
            days=args.days,
            email=args.email
        )
    
    # Execution time
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"⏱️  Total execution time: {duration:.1f} seconds")


if __name__ == "__main__":
    main()
