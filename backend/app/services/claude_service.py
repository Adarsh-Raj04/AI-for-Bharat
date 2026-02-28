"""
Claude Service - Anthropic Claude API integration with streaming
"""
from anthropic import Anthropic
from typing import List, Dict, Any, Iterator
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ClaudeService:
    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.CLAUDE_MODEL
    
    def build_prompt(
        self,
        query: str,
        context_documents: List[Dict[str, Any]]
    ) -> str:
        """
        Build augmented prompt with retrieved context
        
        Args:
            query: User's query
            context_documents: Retrieved and reranked documents
            
        Returns:
            Formatted prompt with context
        """
        # Build context section
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            source_type = doc.get("source_type", "unknown")
            source_id = doc.get("source_id", "")
            title = doc.get("title", "Untitled")
            text = doc.get("text", "")
            
            context_parts.append(
                f"[{i}] {source_type.upper()}: {source_id}\n"
                f"Title: {title}\n"
                f"Content: {text}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        # Build full prompt
        prompt = f"""You are MedResearch AI, an expert medical research assistant. Your role is to provide accurate, evidence-based answers to medical and pharmaceutical research questions.

CONTEXT DOCUMENTS:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question using ONLY the information from the context documents provided above
2. Cite sources using [number] notation (e.g., [1], [2]) corresponding to the context documents
3. If the context doesn't contain enough information, acknowledge this limitation
4. Use clear, professional medical language
5. Format your response with markdown for readability (headers, lists, tables where appropriate)
6. Include relevant statistics, findings, or data points from the sources
7. Be concise but comprehensive

Please provide your answer:"""
        
        return prompt
    
    def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate response using Claude (non-streaming)
        
        Args:
            query: User's query
            context_documents: Retrieved context
            max_tokens: Maximum tokens in response
            
        Returns:
            Response dict with text and metadata
        """
        try:
            prompt = self.build_prompt(query, context_documents)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            text = response.content[0].text
            
            return {
                "text": text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def generate_response_stream(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> Iterator[str]:
        """
        Generate streaming response using Claude
        
        Args:
            query: User's query
            context_documents: Retrieved context
            max_tokens: Maximum tokens in response
            
        Yields:
            Text chunks as they're generated
        """
        try:
            prompt = self.build_prompt(query, context_documents)
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise


# Singleton instance
_claude_service = None


def get_claude_service() -> ClaudeService:
    """Get or create the Claude service singleton"""
    global _claude_service
    if _claude_service is None:
        _claude_service = ClaudeService()
    return _claude_service
