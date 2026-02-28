"""
Bedrock Service - AWS Bedrock integration for Claude
"""
import boto3
import json
from typing import List, Dict, Any, Iterator
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class BedrockService:
    """
    AWS Bedrock service for Claude model
    Drop-in replacement for ClaudeService
    """
    
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=settings.AWS_REGION
        )
        self.model_id = settings.BEDROCK_MODEL_ID
    
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
        Generate response using AWS Bedrock Claude
        
        Args:
            query: User's query
            context_documents: Retrieved context
            max_tokens: Maximum tokens in response
            
        Returns:
            Response dict with text and metadata
        """
        try:
            prompt = self.build_prompt(query, context_documents)
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            text = response_body['content'][0]['text']
            
            return {
                "text": text,
                "tokens_used": response_body['usage']['input_tokens'] + response_body['usage']['output_tokens'],
                "model": self.model_id
            }
            
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            raise
    
    def generate_response_stream(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> Iterator[str]:
        """
        Generate streaming response using AWS Bedrock Claude
        
        Args:
            query: User's query
            context_documents: Retrieved context
            max_tokens: Maximum tokens in response
            
        Yields:
            Text chunks as they're generated
        """
        try:
            prompt = self.build_prompt(query, context_documents)
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            })
            
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=body
            )
            
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'])
                if chunk['type'] == 'content_block_delta':
                    if 'delta' in chunk and 'text' in chunk['delta']:
                        yield chunk['delta']['text']
                        
        except Exception as e:
            logger.error(f"Bedrock streaming error: {e}")
            raise


# Singleton instance
_bedrock_service = None


def get_bedrock_service() -> BedrockService:
    """Get or create the Bedrock service singleton"""
    global _bedrock_service
    if _bedrock_service is None:
        _bedrock_service = BedrockService()
    return _bedrock_service
