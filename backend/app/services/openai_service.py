"""
OpenAI Service - OpenAI API integration for GPT models
"""
from openai import OpenAI
from typing import List, Dict, Any, Iterator
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class OpenAIService:
    """
    OpenAI service for GPT models
    Drop-in replacement for ClaudeService
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    def build_prompt(
        self,
        query: str,
        context_documents: List[Dict[str, Any]]
    ) -> str:
        """
        Build augmented prompt with retrieved context
        """
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
        
        system_prompt = """You are MedResearch AI, an expert medical research assistant. Your role is to provide accurate, evidence-based answers to medical and pharmaceutical research questions.

INSTRUCTIONS:
1. Answer questions using ONLY the information from the context documents provided
2. Cite sources using [number] notation (e.g., [1], [2]) corresponding to the context documents
3. If the context doesn't contain enough information, acknowledge this limitation
4. Use clear, professional medical language
5. Format your response with markdown for readability (headers, lists, tables where appropriate)
6. Include relevant statistics, findings, or data points from the sources
7. Be concise but comprehensive"""

        user_prompt = f"""CONTEXT DOCUMENTS:
{context_text}

USER QUESTION:
{query}

Please provide your answer:"""
        
        return system_prompt, user_prompt
    
    def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI GPT
        
        Args:
            query: User's query
            context_documents: Retrieved context
            max_tokens: Maximum tokens in response
            
        Returns:
            Response dict with text and metadata
        """
        try:
            system_prompt, user_prompt = self.build_prompt(query, context_documents)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            text = response.choices[0].message.content
            
            return {
                "text": text,
                "tokens_used": response.usage.total_tokens,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_response_stream(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> Iterator[str]:
        """
        Generate streaming response using OpenAI GPT
        
        Args:
            query: User's query
            context_documents: Retrieved context
            max_tokens: Maximum tokens in response
            
        Yields:
            Text chunks as they're generated
        """
        try:
            system_prompt, user_prompt = self.build_prompt(query, context_documents)
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


# Singleton instance
_openai_service = None


def get_openai_service() -> OpenAIService:
    """Get or create the OpenAI service singleton"""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service
