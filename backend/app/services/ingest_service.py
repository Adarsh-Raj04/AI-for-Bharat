"""
Ingest Service
Chunks text → embeds → upserts to Pinecone → returns summary.
Dimension: 1536 (OpenAI text-embedding-3-small / ada-002)
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks
EMBEDDING_DIM = 1536  # OpenAI embedding dimension


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _stable_id(source_id: str, chunk_index: int) -> str:
    """Generate a deterministic Pinecone vector ID matching existing pattern."""
    # Matches: PMID:41785024_chunk_0
    safe = re.sub(r"[^a-zA-Z0-9:._-]", "_", source_id)
    return f"{safe}_chunk_{chunk_index}"


class IngestService:
    """
    Ingests a fetched/parsed document into Pinecone and returns a summary.
    """

    def __init__(self, pinecone_index, embedding_model, summarization_chain):
        self.index = pinecone_index
        self.embedder = embedding_model  # LangChain embeddings object
        self.summarization_chain = summarization_chain

    def ingest_and_summarize(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full pipeline: chunk → embed → upsert → summarize.

        Args:
            doc: Output from URLFetcher.fetch() or DocumentParser.parse()

        Returns:
            { summary, source_id, chunks_indexed, already_existed, citation }
        """
        text = doc.get("text", "").strip()
        if not text:
            return {
                "error": doc.get("error")
                or "No text could be extracted from this source."
            }

        source_id = self._build_source_id(doc)
        title = doc.get("title") or source_id
        url = doc.get("url", "")
        source_type = doc.get("source_type", "web")

        # Check if already indexed
        already_existed, existing_chunks = self._check_exists(source_id)

        chunks_indexed = 0
        if not already_existed:
            chunks = _chunk_text(text)
            vectors = self._build_vectors(chunks, source_id, doc)
            if vectors:
                # Upsert in batches of 100 (Pinecone limit)
                for i in range(0, len(vectors), 100):
                    self.index.upsert(vectors=vectors[i : i + 100])
                chunks_indexed = len(vectors)
                logger.info(
                    "Upserted %d chunks for source_id=%s", chunks_indexed, source_id
                )
        else:
            logger.info(
                "Source %s already indexed (%d chunks), skipping upsert",
                source_id,
                existing_chunks,
            )

        # Build doc dict compatible with summarization_chain
        summary_doc = {
            "text": text,
            "title": title,
            "url": url,
            "source_type": source_type,
            "source_id": source_id,
            "pmid": doc.get("pmid"),
            "journal": doc.get("journal", ""),
            "publication_date": doc.get("publication_date", ""),
            "authors": doc.get("authors", ""),
            "metadata": doc,
        }

        summary = self.summarization_chain.summarize_single_document(summary_doc)

        citation = {
            "number": 1,
            "title": title,
            "url": url,
            "relevance_score": 1.0,
            "source_type": source_type,
            "source_id": source_id,
            "publication_date": doc.get("publication_date") or None,
            "times_cited": 0,
            "is_cited": True,
        }

        return {
            "summary": summary,
            "source_id": source_id,
            "chunks_indexed": chunks_indexed,
            "already_existed": already_existed,
            "citation": citation,
            "error": None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_source_id(self, doc: Dict[str, Any]) -> str:
        if doc.get("pmid"):
            return f"PMID:{doc['pmid']}"
        if doc.get("nct_id"):
            return f"NCT:{doc['nct_id']}"
        url = doc.get("url", "")
        if url:
            h = hashlib.md5(url.encode()).hexdigest()[:8]
            return f"URL:{h}"
        return f"DOC:{hashlib.md5(doc.get('title', '').encode()).hexdigest()[:8]}"

    def _check_exists(self, source_id: str):
        """Return (exists: bool, chunk_count: int)."""
        try:
            probe_id = _stable_id(source_id, 0)
            result = self.index.fetch(ids=[probe_id])
            if result.vectors and probe_id in result.vectors:
                # Count chunks via metadata filter
                dummy = [0.0] * EMBEDDING_DIM
                res = self.index.query(
                    vector=dummy,
                    top_k=100,
                    filter={"source_id": {"$eq": source_id}},
                    include_metadata=False,
                )
                return True, len(res.matches)
            return False, 0
        except Exception as e:
            logger.warning("Existence check failed for %s: %s", source_id, e)
            return False, 0

    def _build_vectors(
        self, chunks: List[str], source_id: str, doc: Dict[str, Any]
    ) -> List[Dict]:
        """Embed chunks and build Pinecone upsert payload."""
        try:
            embeddings = self.embedder.embed_documents(chunks)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return []

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append(
                {
                    "id": _stable_id(source_id, i),
                    "values": embedding,
                    "metadata": {
                        # Core fields — matches existing index metadata shape
                        "text": chunk,
                        "source_id": source_id,
                        "source_type": doc.get("source_type", "web"),
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                        "pmid": doc.get("pmid") or "",
                        "nct_id": doc.get("nct_id") or "",
                        "authors": doc.get("authors", ""),
                        "journal": doc.get("journal", ""),
                        "publication_date": doc.get("publication_date", ""),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "industry_sponsored": False,
                        "ingested_by_user": True,  # flag so we know it was user-submitted
                    },
                }
            )
        return vectors


# Singleton
_ingest_service: Optional[IngestService] = None


def get_ingest_service(
    pinecone_index, embedding_model, summarization_chain
) -> IngestService:
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService(
            pinecone_index, embedding_model, summarization_chain
        )
    return _ingest_service
