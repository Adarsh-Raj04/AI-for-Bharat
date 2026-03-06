"""
Summarization Chain - Specialized chain for research paper summarization
Phase 2 Feature #15
"""

from typing import Dict, Any, List, Optional
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SummarizationChain:
    """
    Specialized chain for summarizing research papers.
    Supports both single-document and multi-document summarization.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize summarization prompts"""
        
        # Single document summarization
        self.single_doc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical research expert. Summarize the research paper in a structured format.

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
{text}"""),
            ("human", "Please summarize this research paper.")
        ])
        
        # Multiple document summarization (map-reduce)
        self.map_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the key points from this research paper section.

Focus on:
- Main findings and results
- Methodology highlights
- Conclusions
- Important statistics

Be concise and factual.

CONTENT:
{text}"""),
            ("human", "Extract key points.")
        ])
        
        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize the following summaries into a comprehensive research summary.

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
{text}"""),
            ("human", "Create a comprehensive summary.")
        ])
        
        # PMID-specific prompt
        self.pmid_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are summarizing a PubMed research paper.

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
{text}"""),
            ("human", "Summarize this PubMed paper.")
        ])
        
        logger.info("Initialized summarization prompts")
    
    def summarize_single_document(
        self,
        document: Dict[str, Any],
        query: Optional[str] = None
    ) -> str:
        """
        Summarize a single research document
        
        Args:
            document: Document dict with text and metadata
            query: Optional specific question to focus on
            
        Returns:
            Structured summary
        """
        try:
            text = document.get("text", "")
            metadata = document.get("metadata", {})
            
            # Check if it's a PMID document
            if metadata.get("source_type") == "pubmed" and metadata.get("pmid"):
                chain = (
                    {
                        "text": lambda _: text,
                        "pmid": lambda _: metadata.get("pmid", "Unknown"),
                        "title": lambda _: metadata.get("title", "Unknown"),
                        "journal": lambda _: metadata.get("journal", "Unknown"),
                        "year": lambda _: metadata.get("publication_date", "Unknown")[:4],
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
        query: Optional[str] = None
    ) -> str:
        """
        Summarize multiple research documents using map-reduce
        
        Args:
            documents: List of document dicts
            query: Optional specific question to focus on
            
        Returns:
            Synthesized summary
        """
        try:
            if len(documents) == 1:
                return self.summarize_single_document(documents[0], query)
            
            # Map phase: Summarize each document
            individual_summaries = []
            for i, doc in enumerate(documents[:5], 1):  # Limit to 5 docs
                text = doc.get("text", "")
                title = doc.get("title", f"Document {i}")
                
                chain = (
                    {"text": lambda x: text}
                    | self.map_prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                summary = chain.invoke({})
                individual_summaries.append(f"[{i}] {title}\n{summary}")
            
            # Reduce phase: Synthesize all summaries
            combined_text = "\n\n".join(individual_summaries)
            
            chain = (
                {"text": lambda _: combined_text}
                | self.reduce_prompt
                | self.llm
                | StrOutputParser()
            )
            
            final_summary = chain.invoke({})
            logger.info(f"Generated multi-document summary from {len(documents)} documents")
            return final_summary
            
        except Exception as e:
            logger.error(f"Multi-document summarization error: {e}")
            return f"Error generating summary: {str(e)}"
    
    def extract_pmid_from_query(self, query: str) -> Optional[str]:
        """Extract PMID from query if present"""
        # Match patterns like "PMID:12345678" or "PMID 12345678" or just "12345678"
        patterns = [
            r'PMID:?\s*(\d{7,8})',
            r'pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})',
            r'\b(\d{8})\b'  # 8-digit number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def summarize_by_pmid(
        self,
        pmid: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Summarize a specific paper by PMID
        
        Args:
            pmid: PubMed ID
            documents: Retrieved documents (should contain the PMID paper)
            
        Returns:
            Summary of the specific paper
        """
        # Find the document with matching PMID
        target_doc = None
        for doc in documents:
            metadata = doc.get("metadata", {})
            if metadata.get("pmid") == pmid or metadata.get("source_id") == f"PMID:{pmid}":
                target_doc = doc
                break
        
        if not target_doc:
            return f"Paper with PMID:{pmid} not found in retrieved documents. Please try a more specific search."
        
        return self.summarize_single_document(target_doc)


def get_summarization_chain(llm) -> SummarizationChain:
    """Create a summarization chain instance"""
    return SummarizationChain(llm)
