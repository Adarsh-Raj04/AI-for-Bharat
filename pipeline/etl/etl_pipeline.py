"""
ETL Pipeline - Main orchestrator for data ingestion
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from app.etl.pubmed_fetcher import PubMedFetcher
from app.etl.clinicaltrials_fetcher import ClinicalTrialsFetcher
from app.etl.biorxiv_fetcher import BioRxivFetcher
from app.etl.document_chunker import DocumentChunker
from app.services.embedding_service import get_embedding_service
from app.services.pinecone_service import get_pinecone_service

logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    Main ETL pipeline for ingesting medical research data
    Fetches → Chunks → Embeds → Upserts to Pinecone
    """
    
    def __init__(
        self,
        email: str = "your-email@example.com",
        pubmed_api_key: Optional[str] = None
    ):
        # Initialize fetchers
        self.pubmed_fetcher = PubMedFetcher(email=email, api_key=pubmed_api_key)
        self.clinicaltrials_fetcher = ClinicalTrialsFetcher()
        self.medrxiv_fetcher = BioRxivFetcher(server="medrxiv")
        self.biorxiv_fetcher = BioRxivFetcher(server="biorxiv")
        
        # Initialize chunker
        self.chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        
        # Initialize services
        self.embedding_service = get_embedding_service()
        self.pinecone_service = get_pinecone_service()
    
    def ingest_pubmed(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Ingest PubMed papers
        
        Args:
            query: Search query
            max_results: Maximum number of papers
            start_date: Start date (YYYY/MM/DD)
            end_date: End date (YYYY/MM/DD)
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting PubMed ingestion: {query}")
        
        # Fetch papers
        papers = self.pubmed_fetcher.fetch_by_query(
            query=query,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date
        )
        
        if not papers:
            logger.warning("No papers fetched from PubMed")
            return {"source": "pubmed", "fetched": 0, "chunked": 0, "embedded": 0, "upserted": 0}
        
        # Process papers
        stats = self._process_documents(papers)
        stats["source"] = "pubmed"
        
        logger.info(f"PubMed ingestion complete: {stats}")
        return stats
    
    def ingest_clinical_trials(
        self,
        query: str,
        max_results: int = 100,
        status: Optional[str] = None,
        phase: Optional[str] = None
    ) -> Dict:
        """
        Ingest clinical trials
        
        Args:
            query: Search query
            max_results: Maximum number of trials
            status: Trial status filter
            phase: Trial phase filter
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting ClinicalTrials.gov ingestion: {query}")
        
        # Fetch trials
        trials = self.clinicaltrials_fetcher.search(
            query=query,
            max_results=max_results,
            status=status,
            phase=phase
        )
        
        if not trials:
            logger.warning("No trials fetched from ClinicalTrials.gov")
            return {"source": "clinical_trials", "fetched": 0, "chunked": 0, "embedded": 0, "upserted": 0}
        
        # Process trials
        stats = self._process_documents(trials)
        stats["source"] = "clinical_trials"
        
        logger.info(f"ClinicalTrials.gov ingestion complete: {stats}")
        return stats
    
    def ingest_medrxiv(
        self,
        days: int = 30,
        max_results: int = 100
    ) -> Dict:
        """
        Ingest medRxiv preprints
        
        Args:
            days: Number of days back to fetch
            max_results: Maximum number of preprints
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting medRxiv ingestion: last {days} days")
        
        # Fetch preprints
        preprints = self.medrxiv_fetcher.fetch_recent(days=days, max_results=max_results)
        
        if not preprints:
            logger.warning("No preprints fetched from medRxiv")
            return {"source": "medrxiv", "fetched": 0, "chunked": 0, "embedded": 0, "upserted": 0}
        
        # Process preprints
        stats = self._process_documents(preprints)
        stats["source"] = "medrxiv"
        
        logger.info(f"medRxiv ingestion complete: {stats}")
        return stats
    
    def ingest_biorxiv(
        self,
        days: int = 30,
        max_results: int = 100
    ) -> Dict:
        """
        Ingest bioRxiv preprints
        
        Args:
            days: Number of days back to fetch
            max_results: Maximum number of preprints
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting bioRxiv ingestion: last {days} days")
        
        # Fetch preprints
        preprints = self.biorxiv_fetcher.fetch_recent(days=days, max_results=max_results)
        
        if not preprints:
            logger.warning("No preprints fetched from bioRxiv")
            return {"source": "biorxiv", "fetched": 0, "chunked": 0, "embedded": 0, "upserted": 0}
        
        # Process preprints
        stats = self._process_documents(preprints)
        stats["source"] = "biorxiv"
        
        logger.info(f"bioRxiv ingestion complete: {stats}")
        return stats
    
    def ingest_all(
        self,
        query: str,
        max_results_per_source: int = 50
    ) -> Dict:
        """
        Ingest from all sources
        
        Args:
            query: Search query
            max_results_per_source: Max results per source
            
        Returns:
            Combined ingestion statistics
        """
        logger.info(f"Starting full ingestion for query: {query}")
        
        all_stats = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": {}
        }
        
        # PubMed
        try:
            pubmed_stats = self.ingest_pubmed(query, max_results=max_results_per_source)
            all_stats["sources"]["pubmed"] = pubmed_stats
        except Exception as e:
            logger.error(f"PubMed ingestion failed: {e}")
            all_stats["sources"]["pubmed"] = {"error": str(e)}
        
        # ClinicalTrials.gov
        try:
            ct_stats = self.ingest_clinical_trials(query, max_results=max_results_per_source)
            all_stats["sources"]["clinical_trials"] = ct_stats
        except Exception as e:
            logger.error(f"ClinicalTrials.gov ingestion failed: {e}")
            all_stats["sources"]["clinical_trials"] = {"error": str(e)}
        
        # medRxiv (recent papers, not query-based)
        try:
            medrxiv_stats = self.ingest_medrxiv(days=30, max_results=max_results_per_source)
            all_stats["sources"]["medrxiv"] = medrxiv_stats
        except Exception as e:
            logger.error(f"medRxiv ingestion failed: {e}")
            all_stats["sources"]["medrxiv"] = {"error": str(e)}
        
        # Calculate totals
        all_stats["totals"] = self._calculate_totals(all_stats["sources"])
        
        logger.info(f"Full ingestion complete: {all_stats['totals']}")
        return all_stats
    
    def _process_documents(self, documents: List[Dict]) -> Dict:
        """
        Process documents: chunk → embed → upsert
        
        Args:
            documents: List of documents to process
            
        Returns:
            Processing statistics
        """
        stats = {
            "fetched": len(documents),
            "chunked": 0,
            "embedded": 0,
            "upserted": 0,
            "errors": 0
        }
        
        all_chunks = []
        
        # Chunk documents
        for doc in documents:
            try:
                chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                stats["chunked"] += len(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('id')}: {e}")
                stats["errors"] += 1
        
        if not all_chunks:
            logger.warning("No chunks generated")
            return stats
        
        # Generate embeddings and prepare vectors
        vectors = []
        
        for chunk in all_chunks:
            try:
                # Generate embedding
                embedding = self.embedding_service.embed_query(chunk["text"])
                stats["embedded"] += 1
                
                # Prepare vector for Pinecone
                vector = {
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"][:1000],  # Pinecone metadata limit
                        "source_id": chunk["metadata"].get("source_id", ""),
                        "source_type": chunk["metadata"].get("source_type", ""),
                        "title": chunk["metadata"].get("title", "")[:500],
                        "url": chunk["metadata"].get("url", ""),
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                        "industry_sponsored": chunk["metadata"].get("industry_sponsored", False),
                        "publication_date": chunk["metadata"].get("publication_date", ""),
                        "authors": str(chunk["metadata"].get("authors", []))[:500],
                        "doi": chunk["metadata"].get("doi", ""),
                    }
                }
                
                vectors.append(vector)
                
            except Exception as e:
                logger.error(f"Failed to embed chunk {chunk.get('id')}: {e}")
                stats["errors"] += 1
        
        # Upsert to Pinecone in batches
        if vectors:
            try:
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    self.pinecone_service.upsert_documents(batch)
                    stats["upserted"] += len(batch)
                    
            except Exception as e:
                logger.error(f"Failed to upsert vectors: {e}")
                stats["errors"] += 1
        
        return stats
    
    def _calculate_totals(self, sources: Dict) -> Dict:
        """Calculate total statistics across all sources"""
        totals = {
            "fetched": 0,
            "chunked": 0,
            "embedded": 0,
            "upserted": 0,
            "errors": 0
        }
        
        for source_stats in sources.values():
            if isinstance(source_stats, dict) and "error" not in source_stats:
                for key in totals.keys():
                    totals[key] += source_stats.get(key, 0)
        
        return totals


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = ETLPipeline(email="your-email@example.com")
    
    # Ingest diabetes research
    stats = pipeline.ingest_all("diabetes treatment", max_results_per_source=10)
    
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"Total fetched: {stats['totals']['fetched']}")
    print(f"Total chunked: {stats['totals']['chunked']}")
    print(f"Total embedded: {stats['totals']['embedded']}")
    print(f"Total upserted: {stats['totals']['upserted']}")
    print(f"Total errors: {stats['totals']['errors']}")
    print("=" * 80)
