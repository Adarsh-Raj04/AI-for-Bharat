"""
Document Embedding and Upload to Pinecone
"""

import os
import time
from typing import List, Dict
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def embed_and_upload(chunks: List[Dict], batch_size: int = 100) -> int:
    """
    Embed documents and upload to Pinecone
    """

    if not chunks:
        logger.warning("No chunks provided for embedding.")
        return 0

    # -------------------------
    # OpenAI Initialization
    # -------------------------

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set")

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    openai_client = OpenAI(api_key=openai_api_key)

    # -------------------------
    # Pinecone Initialization
    # -------------------------

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not set")

    index_name = os.getenv("PINECONE_INDEX_NAME", "medresearch-ai")
    reset_index = os.getenv("PINECONE_RESET_INDEX", "false").lower() == "true"

    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    # Delete index if requested
    if reset_index and index_name in existing_indexes:
        logger.info(f"Deleting existing Pinecone index: {index_name}")
        pc.delete_index(index_name)
        existing_indexes.remove(index_name)

    # Create index if missing
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
        )

    index = pc.Index(index_name)

    success_count = 0
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches")

    # -------------------------
    # Batch Processing
    # -------------------------

    for batch_idx in tqdm(range(total_batches), desc="Embedding and uploading"):

        start = batch_idx * batch_size
        end = start + batch_size
        batch = chunks[start:end]

        try:

            # -------------------------
            # Filter bad chunks
            # -------------------------

            valid_batch = [
                c for c in batch
                if c.get("text") and len(c["text"].strip()) > 50
            ]

            if not valid_batch:
                logger.warning(f"Batch {batch_idx+1} contains no valid chunks")
                continue

            texts = [c["text"][:8000] for c in valid_batch]

            # -------------------------
            # Embedding with Retry
            # -------------------------

            for attempt in range(3):
                try:
                    response = openai_client.embeddings.create(
                        model=embedding_model,
                        input=texts
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    logger.warning(f"Embedding retry {attempt+1}")
                    time.sleep(2)

            embeddings = response.data

            # -------------------------
            # Prepare Pinecone vectors
            # -------------------------

            vectors = []

            for j, chunk in enumerate(valid_batch):

                text = chunk.get("text") or ""
                meta = chunk.get("metadata", {})

                metadata = {
                    "text": text[:1000],
                    "source_id": str(meta.get("source_id", "")),
                    "source_type": str(meta.get("source_type", "")),
                    "title": str(meta.get("title", ""))[:500],
                    "url": str(meta.get("url", "")),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "total_chunks": int(chunk.get("total_chunks", 1)),
                    "industry_sponsored": bool(meta.get("industry_sponsored", False)),
                }

                # Optional fields

                publication_date = meta.get("publication_date")
                if publication_date:
                    metadata["publication_date"] = str(publication_date)

                doi = meta.get("doi")
                if doi:
                    metadata["doi"] = str(doi)

                pmid = meta.get("pmid")
                if pmid:
                    metadata["pmid"] = str(pmid)

                nct_id = meta.get("nct_id")
                if nct_id:
                    metadata["nct_id"] = str(nct_id)

                journal = meta.get("journal")
                if journal:
                    metadata["journal"] = str(journal)[:200]

                phase = meta.get("phase")
                if phase:
                    metadata["phase"] = str(phase)

                status = meta.get("status")
                if status:
                    metadata["status"] = str(status)

                sponsor = meta.get("sponsor")
                if sponsor:
                    metadata["sponsor"] = str(sponsor)[:200]

                category = meta.get("category")
                if category:
                    metadata["category"] = str(category)

                # Handle lists

                authors = meta.get("authors")
                if authors and isinstance(authors, list):
                    metadata["authors"] = ", ".join([str(a) for a in authors[:5]])

                conditions = meta.get("conditions")
                if conditions and isinstance(conditions, list):
                    metadata["conditions"] = ", ".join([str(c) for c in conditions[:5]])

                vector = {
                    "id": str(chunk.get("id", f"chunk_{batch_idx}_{j}")),
                    "values": embeddings[j].embedding,
                    "metadata": metadata
                }

                vectors.append(vector)

            # -------------------------
            # Pinecone Upsert
            # -------------------------

            if vectors:
                index.upsert(vectors=vectors)
                success_count += len(vectors)

        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx + 1}: {e}")

    logger.info(f"Successfully uploaded {success_count}/{len(chunks)} chunks")

    return success_count