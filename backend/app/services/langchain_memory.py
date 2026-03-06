"""
LangChain Memory - Enhanced conversational memory with summarization
Phase 2 Feature #3 (Simplified - uses existing Redis/DB)
"""

from typing import List, Dict, Any, Optional
import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class ConversationalMemory:
    """
    Enhanced conversational memory using LangChain patterns.
    Uses existing Redis/DB infrastructure with smart summarization.
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.max_messages = 20
        self.summarize_threshold = 10
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize memory-related prompts"""
        
        # Conversation summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize the following conversation, preserving key medical entities, topics discussed, and important context.

Focus on:
- Medical terms, drug names, conditions discussed
- Key questions asked and answers provided
- Important findings or recommendations
- Any ongoing topics or unresolved questions

Be concise but preserve all medically relevant information.

CONVERSATION:
{conversation}"""),
            ("human", "Provide a concise summary of this conversation.")
        ])
    
    def format_messages_for_langchain(
        self,
        chat_history: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """
        Convert chat history to LangChain message format
        
        Args:
            chat_history: List of message dicts with 'role' and 'content'
            
        Returns:
            List of LangChain BaseMessage objects
        """
        messages = []
        for msg in chat_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
        
        return messages
    
    def should_summarize(self, chat_history: List[Dict]) -> bool:
        """Check if conversation should be summarized"""
        return len(chat_history) >= self.summarize_threshold
    
    def summarize_conversation(
        self,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Summarize a long conversation to preserve context
        
        Args:
            chat_history: Full conversation history
            
        Returns:
            Summarized conversation text
        """
        if not self.llm:
            # Fallback: just truncate
            return self._truncate_history(chat_history)
        
        try:
            # Format conversation for summarization
            conversation_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in chat_history
            ])
            
            # Build summarization chain
            chain = (
                {"conversation": lambda _: conversation_text}
                | self.summarization_prompt
                | self.llm
                | StrOutputParser()
            )
            
            summary = chain.invoke({})
            logger.info(f"Summarized {len(chat_history)} messages")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return self._truncate_history(chat_history)
    
    def _truncate_history(self, chat_history: List[Dict]) -> str:
        """Fallback: simple truncation"""
        recent = chat_history[-5:]
        return "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:200]}"
            for msg in recent
        ])
    
    def manage_context_window(
        self,
        chat_history: List[Dict[str, str]],
        max_tokens: int = 4000
    ) -> List[Dict[str, str]]:
        """
        Manage conversation history to fit within context window
        
        Args:
            chat_history: Full conversation history
            max_tokens: Maximum tokens to preserve (approximate)
            
        Returns:
            Managed conversation history
        """
        if len(chat_history) <= self.max_messages:
            return chat_history
        
        # If we have LLM, summarize old messages
        if self.llm and len(chat_history) > self.summarize_threshold:
            # Keep recent messages, summarize old ones
            old_messages = chat_history[:-self.summarize_threshold]
            recent_messages = chat_history[-self.summarize_threshold:]
            
            summary = self.summarize_conversation(old_messages)
            
            # Create a summary message
            summary_msg = {
                'role': 'system',
                'content': f"Previous conversation summary: {summary}"
            }
            
            return [summary_msg] + recent_messages
        else:
            # Simple truncation
            return chat_history[-self.max_messages:]
    
    def extract_entities(
        self,
        chat_history: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """
        Extract key entities from conversation
        
        Args:
            chat_history: Conversation history
            
        Returns:
            Dict of entity types and values
        """
        import re
        
        entities = {
            'drugs': set(),
            'conditions': set(),
            'studies': set(),
            'pmids': set()
        }
        
        # Simple regex-based extraction
        for msg in chat_history:
            content = msg.get('content', '')
            
            # Extract PMIDs
            pmids = re.findall(r'PMID:?\s*(\d{7,8})', content, re.IGNORECASE)
            entities['pmids'].update(pmids)
            
            # Extract NCT numbers (clinical trials)
            ncts = re.findall(r'NCT\d{8}', content, re.IGNORECASE)
            entities['studies'].update(ncts)
            
            # Common drug name patterns (simplified)
            drug_patterns = [
                r'\b(pembrolizumab|nivolumab|atezolizumab|durvalumab)\b',
                r'\b(chemotherapy|immunotherapy|targeted therapy)\b'
            ]
            for pattern in drug_patterns:
                drugs = re.findall(pattern, content, re.IGNORECASE)
                entities['drugs'].update([d.lower() for d in drugs])
        
        # Convert sets to lists
        return {k: list(v) for k, v in entities.items()}
    
    def get_conversation_context(
        self,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Get a context string summarizing the conversation
        
        Args:
            chat_history: Conversation history
            
        Returns:
            Context string
        """
        if not chat_history:
            return ""
        
        entities = self.extract_entities(chat_history)
        
        context_parts = []
        
        if entities['drugs']:
            context_parts.append(f"Drugs discussed: {', '.join(entities['drugs'])}")
        
        if entities['pmids']:
            context_parts.append(f"Papers referenced: {', '.join(entities['pmids'])}")
        
        if entities['studies']:
            context_parts.append(f"Clinical trials: {', '.join(entities['studies'])}")
        
        return " | ".join(context_parts) if context_parts else ""


def get_conversational_memory(llm=None) -> ConversationalMemory:
    """Create a conversational memory instance"""
    return ConversationalMemory(llm=llm)
