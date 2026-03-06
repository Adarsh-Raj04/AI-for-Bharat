"""
Citation Tracker - Enhanced citation extraction and validation
Phase 2 Feature #6
"""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class CitationTracker:
    """
    Enhanced citation tracking with validation and metadata enrichment.
    Extracts inline citations from responses and validates against sources.
    """
    
    def __init__(self):
        self.citation_pattern = r'\[(\d+)\]'
    
    def extract_inline_citations(self, text: str) -> List[int]:
        """
        Extract citation numbers from text
        
        Args:
            text: Response text with citations like [1], [2]
            
        Returns:
            List of citation numbers
        """
        matches = re.findall(self.citation_pattern, text)
        return [int(m) for m in matches]
    
    def validate_citations(
        self,
        response_text: str,
        available_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that all citations in response match available sources
        
        Args:
            response_text: Generated response text
            available_sources: List of source documents
            
        Returns:
            Validation results
        """
        cited_numbers = self.extract_inline_citations(response_text)
        available_numbers = [s.get('number', i+1) for i, s in enumerate(available_sources)]
        
        valid_citations = [n for n in cited_numbers if n in available_numbers]
        invalid_citations = [n for n in cited_numbers if n not in available_numbers]
        uncited_sources = [n for n in available_numbers if n not in cited_numbers]
        
        accuracy = len(valid_citations) / len(cited_numbers) if cited_numbers else 1.0
        
        return {
            'total_citations': len(cited_numbers),
            'valid_citations': len(valid_citations),
            'invalid_citations': invalid_citations,
            'uncited_sources': uncited_sources,
            'accuracy': accuracy,
            'is_valid': len(invalid_citations) == 0
        }
    
    def enrich_citations(
        self,
        citations: List[Dict[str, Any]],
        response_text: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich citations with usage information
        
        Args:
            citations: List of citation dicts
            response_text: Response text
            
        Returns:
            Enriched citations
        """
        cited_numbers = self.extract_inline_citations(response_text)
        citation_counts = {}
        for num in cited_numbers:
            citation_counts[num] = citation_counts.get(num, 0) + 1
        
        enriched = []
        for citation in citations:
            num = citation.get('number', 0)
            enriched_citation = citation.copy()
            enriched_citation['times_cited'] = citation_counts.get(num, 0)
            enriched_citation['is_cited'] = num in cited_numbers
            enriched.append(enriched_citation)
        
        return enriched
    
    def extract_citation_context(
        self,
        response_text: str,
        citation_number: int,
        context_chars: int = 100
    ) -> List[str]:
        """
        Extract text context around a citation
        
        Args:
            response_text: Response text
            citation_number: Citation number to find
            context_chars: Characters of context to extract
            
        Returns:
            List of context strings
        """
        pattern = rf'\[{citation_number}\]'
        contexts = []
        
        for match in re.finditer(pattern, response_text):
            start = max(0, match.start() - context_chars)
            end = min(len(response_text), match.end() + context_chars)
            context = response_text[start:end].strip()
            contexts.append(context)
        
        return contexts
    
    def generate_citation_report(
        self,
        response_text: str,
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive citation report
        
        Args:
            response_text: Response text
            citations: List of citations
            
        Returns:
            Citation report
        """
        validation = self.validate_citations(response_text, citations)
        enriched = self.enrich_citations(citations, response_text)
        
        # Calculate statistics
        total_sources = len(citations)
        cited_sources = sum(1 for c in enriched if c['is_cited'])
        citation_rate = cited_sources / total_sources if total_sources > 0 else 0.0
        
        # Find most cited source
        most_cited = max(enriched, key=lambda x: x['times_cited']) if enriched else None
        
        return {
            'validation': validation,
            'enriched_citations': enriched,
            'statistics': {
                'total_sources': total_sources,
                'cited_sources': cited_sources,
                'citation_rate': citation_rate,
                'total_inline_citations': validation['total_citations'],
                'most_cited': {
                    'number': most_cited['number'],
                    'title': most_cited['title'],
                    'times_cited': most_cited['times_cited']
                } if most_cited and most_cited['times_cited'] > 0 else None
            }
        }
    
    def format_citation_apa(self, citation: Dict[str, Any]) -> str:
        """
        Format citation in APA style
        
        Args:
            citation: Citation dict
            
        Returns:
            APA formatted citation string
        """
        title = citation.get('title', 'Untitled')
        source_type = citation.get('source_type', 'unknown')
        source_id = citation.get('source_id', '')
        url = citation.get('url', '')
        
        if source_type == 'pubmed':
            return f"{title}. PubMed. {source_id}. {url}"
        elif source_type == 'clinical_trial':
            return f"{title}. ClinicalTrials.gov. {source_id}. {url}"
        else:
            return f"{title}. {source_type.upper()}. {source_id}. {url}"
    
    def format_citation_mla(self, citation: Dict[str, Any]) -> str:
        """
        Format citation in MLA style
        
        Args:
            citation: Citation dict
            
        Returns:
            MLA formatted citation string
        """
        title = citation.get('title', 'Untitled')
        source_type = citation.get('source_type', 'unknown')
        url = citation.get('url', '')
        
        return f'"{title}." {source_type.capitalize()}. Web. {url}'
    
    def format_citations(
        self,
        citations: List[Dict[str, Any]],
        style: str = 'apa'
    ) -> List[str]:
        """
        Format all citations in specified style
        
        Args:
            citations: List of citations
            style: Citation style ('apa', 'mla', 'chicago')
            
        Returns:
            List of formatted citation strings
        """
        if style.lower() == 'apa':
            return [self.format_citation_apa(c) for c in citations]
        elif style.lower() == 'mla':
            return [self.format_citation_mla(c) for c in citations]
        else:
            # Default to simple format
            return [
                f"[{c.get('number')}] {c.get('title')} ({c.get('source_type')})"
                for c in citations
            ]


def get_citation_tracker() -> CitationTracker:
    """Create a citation tracker instance"""
    return CitationTracker()
