"""
Summarization Chain - Specialized chain for research paper summarization
Phase 2 Feature #15
"""

from typing import Dict, Any, List, Optional
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SummarizationChain:

    def __init__(self, llm):
        self.llm = llm
        self._initialize_prompts()

    def _initialize_prompts(self):
        self.single_doc_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical research expert. Summarize the research paper in a structured format.

INSTRUCTIONS:
1. Extract key information from the paper
2. Use the exact structure provided below
3. Be concise but comprehensive
4. Include specific numbers, statistics, and findings
5. Cite the source with [1] notation

REQUIRED STRUCTURE:

## Objective
[What was the research question or hypothesis?]

## Methods
- Study Design: [Type of study]
- Population: [Sample size and characteristics]
- Intervention/Exposure: [What was studied]
- Outcome Measures: [Primary and secondary endpoints]

## Key Results
[Main findings with statistics - use bullet points]
- [Finding 1 with numbers]
- [Finding 2 with numbers]
- [Finding 3 with numbers]

## Conclusions
[Main conclusions and clinical implications]

## Limitations
[Key limitations mentioned in the paper]

PAPER CONTENT:
{text}""",
                ),
                ("human", "Please summarize this research paper."),
            ]
        )

        self.map_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Extract the key points from this research paper section.

Focus on:
- Main findings and results
- Methodology highlights
- Conclusions
- Important statistics

Be concise and factual.

CONTENT:
{text}""",
                ),
                ("human", "Extract key points."),
            ]
        )

        self.reduce_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Synthesize the following summaries into a comprehensive research summary.

Use this structure:

## Overview
[Brief overview of the research]

## Key Findings
[Synthesized findings from all papers]

## Methodology
[Common methodological approaches]

## Conclusions
[Overall conclusions and implications]

SUMMARIES:
{text}""",
                ),
                ("human", "Create a comprehensive summary."),
            ]
        )

        self.pmid_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are summarizing a PubMed research paper.

REQUIRED STRUCTURE:

## Citation
**PMID:** {pmid}
**Title:** {title}
**Journal:** {journal}
**Year:** {year}

## Abstract Summary
[Concise summary of the abstract]

## Key Points
- [Point 1]
- [Point 2]
- [Point 3]

## Clinical Relevance
[Why this matters for clinical practice]

PAPER CONTENT:
{text}""",
                ),
                ("human", "Summarize this PubMed paper."),
            ]
        )

        logger.info("Initialized summarization prompts")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_field(self, doc: Dict[str, Any], *keys: str, default: str = "") -> str:
        """
        Pull a field from a reranker-flattened dict OR a nested metadata dict.
        Reranker output has metadata fields at the top level AND inside 'metadata'.
        """
        # Try top-level first (reranker flattens metadata)
        for key in keys:
            val = doc.get(key)
            if val:
                return str(val)
        # Fallback: nested metadata (LangChain Document shape)
        meta = doc.get("metadata", {})
        for key in keys:
            val = meta.get(key)
            if val:
                return str(val)
        return default

    def _get_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from either reranker dict or LangChain Document dict."""
        return (
            doc.get("text")
            or doc.get("page_content")
            or doc.get("metadata", {}).get("text", "")
        )

    # ── public API ────────────────────────────────────────────────────────────

    def summarize_single_document(
        self,
        document: Dict[str, Any],
        query: Optional[str] = None,
    ) -> str:
        try:
            text = self._get_text(document)
            source_type = self._extract_field(document, "source_type")
            pmid = self._extract_field(document, "pmid")

            if source_type == "pubmed" and pmid:
                year_raw = self._extract_field(document, "publication_date")
                year = year_raw[:4] if year_raw else "Unknown"

                chain = (
                    {
                        "text": lambda _: text,
                        "pmid": lambda _: pmid,
                        "title": lambda _: self._extract_field(
                            document, "title", default="Unknown"
                        ),
                        "journal": lambda _: self._extract_field(
                            document, "journal", default="Unknown"
                        ),
                        "year": lambda _: year,
                    }
                    | self.pmid_prompt
                    | self.llm
                    | StrOutputParser()
                )
            else:
                chain = (
                    {"text": lambda _: text}
                    | self.single_doc_prompt
                    | self.llm
                    | StrOutputParser()
                )

            summary = chain.invoke({})
            logger.info("Generated single document summary")
            return summary

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"Error generating summary: {str(e)}"

    def summarize_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> str:
        try:
            if len(documents) == 1:
                return self.summarize_single_document(documents[0], query)

            individual_summaries = []
            for i, doc in enumerate(documents[:5], 1):
                text = self._get_text(doc)
                title = self._extract_field(doc, "title", default=f"Document {i}")

                chain = (
                    {"text": lambda x, t=text: t}
                    | self.map_prompt
                    | self.llm
                    | StrOutputParser()
                )
                summary = chain.invoke({})
                individual_summaries.append(f"[{i}] {title}\n{summary}")

            combined_text = "\n\n".join(individual_summaries)
            chain = (
                {"text": lambda _: combined_text}
                | self.reduce_prompt
                | self.llm
                | StrOutputParser()
            )
            final_summary = chain.invoke({})
            logger.info(
                f"Generated multi-document summary from {len(documents)} documents"
            )
            return final_summary

        except Exception as e:
            logger.error(f"Multi-document summarization error: {e}")
            return f"Error generating summary: {str(e)}"

    def extract_pmid_from_query(self, query: str) -> Optional[str]:
        """Extract PMID from query if present."""
        patterns = [
            r"PMID:?\s*(\d{7,9})",  # PMID:41785024 or PMID 41785024
            r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,9})",  # URL form
            r"\b(\d{8,9})\b",  # bare 8-9 digit number
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def summarize_by_pmid(
        self,
        pmid: str,
        documents: List[Dict[str, Any]],
    ) -> str:
        """
        Summarize a specific paper by PMID.

        Searches both top-level fields (reranker-flattened) and nested metadata
        so it works regardless of which pipeline stage produced the docs.
        """
        target_doc = None
        for doc in documents:
            # Reranker flattens fields to top level
            doc_pmid = self._extract_field(doc, "pmid", "source_id")
            if doc_pmid in (pmid, f"PMID:{pmid}"):
                target_doc = doc
                break

        if not target_doc:
            logger.warning(
                "PMID:%s not found in %d retrieved docs. source_ids: %s",
                pmid,
                len(documents),
                [self._extract_field(d, "source_id") for d in documents],
            )
            return (
                f"The document PMID:{pmid} was retrieved but could not be matched "
                f"in the result set. This can happen when the reranker scores it below "
                f"the top-k cutoff. Try asking: 'summarize the paper about [topic]' "
                f"for better results, or check that PMID:{pmid} exists in the database."
            )

        return self.summarize_single_document(target_doc)


def get_summarization_chain(llm) -> SummarizationChain:
    return SummarizationChain(llm)
