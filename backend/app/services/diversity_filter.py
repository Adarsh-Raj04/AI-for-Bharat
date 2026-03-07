"""
Diversity Filter
----------------
Prevents the retriever from returning multiple chunks from the same source,
ensuring the context window contains a breadth of evidence rather than
depth from a single paper.

Strategy: Max Source Diversity (MSD)
  - After reranking, iterate results in ranked order
  - Track how many chunks each source_id has contributed
  - Skip a chunk if its source has already hit the cap
  - Always keep at least `min_sources` distinct sources if available

Usage:
    from app.services.diversity_filter import apply_diversity_filter

    reranked_docs = apply_diversity_filter(
        documents=reranked_docs,
        max_chunks_per_source=2,
        target_k=5,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def apply_diversity_filter(
    documents: List[Union[Dict[str, Any], Any]],
    max_chunks_per_source: int = 2,
    target_k: int = 5,
    min_sources: int = 3,
) -> List[Union[Dict[str, Any], Any]]:
    """
    Apply source diversity filtering to a ranked list of documents.

    Args:
        documents:             Ranked list of dicts or LangChain Document objects.
        max_chunks_per_source: Max chunks allowed from any single source_id.
        target_k:              Desired number of results to return.
        min_sources:           Minimum distinct sources to target before tightening cap.

    Returns:
        Filtered list preserving ranked order, capped at target_k.
    """
    if not documents:
        return documents

    source_counts: Dict[str, int] = {}
    filtered: List = []
    skipped: List = []  # keep skipped docs as fallback if we run short

    for doc in documents:
        source_id = _get_source_id(doc)

        count = source_counts.get(source_id, 0)

        if count < max_chunks_per_source:
            filtered.append(doc)
            source_counts[source_id] = count + 1
            if len(filtered) >= target_k:
                break
        else:
            skipped.append(doc)

    # If we didn't hit target_k, backfill from skipped (still ranked order)
    if len(filtered) < target_k:
        remaining = target_k - len(filtered)
        filtered.extend(skipped[:remaining])

    distinct_sources = len(source_counts)
    logger.info(
        "Diversity filter: %d → %d docs, %d distinct sources (cap=%d/source)",
        len(documents),
        len(filtered),
        distinct_sources,
        max_chunks_per_source,
    )

    return filtered


def _get_source_id(doc: Union[Dict[str, Any], Any]) -> str:
    """Extract source_id from either a dict or a LangChain Document."""
    if isinstance(doc, dict):
        # Reranker output shape
        return (
            doc.get("source_id")
            or doc.get("metadata", {}).get("source_id")
            or doc.get("id", "unknown")
        )
    # LangChain Document
    metadata = getattr(doc, "metadata", {})
    return metadata.get("source_id") or metadata.get("id", "unknown")
