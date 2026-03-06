"""
Intent Router - Routes queries to specialized chains based on intent
Phase 2 Feature #9
"""

from typing import Dict, Any, List
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.services.intent_classifier import QueryIntent

logger = logging.getLogger(__name__)


class IntentRouter:
    """
    Routes queries to specialized chains based on classified intent.
    Each intent gets optimized prompts and processing logic.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize specialized prompts for each intent"""
        
        # Base system message
        base_system = (
            "You are MedResearch AI, an expert medical research assistant. "
            "Provide accurate, evidence-based answers using ONLY the context provided.\n\n"
        )
        
        # Summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system + 
             "TASK: Provide a structured summary of the research paper(s).\n\n"
             "FORMAT YOUR RESPONSE AS:\n"
             "## Objective\n"
             "[State the research objective]\n\n"
             "## Methods\n"
             "[Describe the methodology]\n\n"
             "## Results\n"
             "[Key findings with statistics]\n\n"
             "## Conclusions\n"
             "[Main conclusions and implications]\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        # Comparison prompt
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system +
             "TASK: Compare the specified items systematically.\n\n"
             "FORMAT YOUR RESPONSE AS:\n"
             "## Comparison Overview\n"
             "[Brief introduction]\n\n"
             "## Key Differences\n"
             "| Aspect | Item A | Item B |\n"
             "|--------|--------|--------|\n"
             "[Use tables for structured comparison]\n\n"
             "## Similarities\n"
             "[Common features or findings]\n\n"
             "## Recommendations\n"
             "[Evidence-based recommendations]\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        # Regulatory/Compliance prompt
        self.regulatory_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system +
             "TASK: Provide regulatory and compliance information.\n\n"
             "IMPORTANT:\n"
             "- Cite specific regulations, guidelines, or approval documents\n"
             "- Include dates and regulatory body names (FDA, EMA, etc.)\n"
             "- Distinguish between requirements, recommendations, and guidance\n"
             "- Note any regional differences\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        # Adverse Events prompt
        self.adverse_events_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system +
             "TASK: Provide safety and adverse event information.\n\n"
             "FORMAT YOUR RESPONSE AS:\n"
             "## Common Adverse Events\n"
             "[List with frequencies if available]\n\n"
             "## Serious Adverse Events\n"
             "[Grade 3+ events with frequencies]\n\n"
             "## Warnings and Precautions\n"
             "[Important safety information]\n\n"
             "## Management Strategies\n"
             "[How to manage adverse events]\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        # Clinical Trial prompt
        self.clinical_trial_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system +
             "TASK: Provide clinical trial information.\n\n"
             "FORMAT YOUR RESPONSE AS:\n"
             "## Trial Design\n"
             "[Phase, design, population]\n\n"
             "## Primary Endpoint\n"
             "[Main outcome measure]\n\n"
             "## Key Results\n"
             "[Efficacy and safety results with statistics]\n\n"
             "## Clinical Implications\n"
             "[What this means for practice]\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        # Factual Lookup prompt (concise)
        self.factual_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system +
             "TASK: Provide a clear, concise answer to the factual question.\n\n"
             "GUIDELINES:\n"
             "- Be direct and specific\n"
             "- Define technical terms\n"
             "- Use bullet points for clarity\n"
             "- Cite sources with [number] notation\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        # General QA prompt (default)
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", base_system +
             "INSTRUCTIONS:\n"
             "1. Answer using ONLY the context provided\n"
             "2. Cite sources with [number] notation\n"
             "3. Use clear, professional medical language\n"
             "4. Format with markdown for readability\n"
             "5. Be concise but comprehensive\n\n"
             "CONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        
        logger.info("Initialized specialized prompts for intent routing")
    
    def get_prompt_for_intent(self, intent: QueryIntent) -> ChatPromptTemplate:
        """Get the appropriate prompt template for an intent"""
        prompt_map = {
            QueryIntent.SUMMARIZATION: self.summarization_prompt,
            QueryIntent.COMPARISON: self.comparison_prompt,
            QueryIntent.REGULATORY_COMPLIANCE: self.regulatory_prompt,
            QueryIntent.ADVERSE_EVENTS: self.adverse_events_prompt,
            QueryIntent.CLINICAL_TRIAL: self.clinical_trial_prompt,
            QueryIntent.FACTUAL_LOOKUP: self.factual_prompt,
            QueryIntent.DRUG_INTERACTION: self.factual_prompt,  # Use factual for drug interactions
            QueryIntent.GENERAL_QA: self.general_prompt,
        }
        
        return prompt_map.get(intent, self.general_prompt)
    
    def build_routed_chain(self, intent: QueryIntent, context: str):
        """
        Build a chain with the appropriate prompt for the intent
        
        Args:
            intent: Classified query intent
            context: Formatted context documents
            
        Returns:
            Runnable chain
        """
        prompt = self.get_prompt_for_intent(intent)
        
        chain = (
            {
                "context": lambda _: context,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info(f"Built routed chain for intent: {intent.value}")
        return chain
    
    def get_intent_metadata(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get metadata about how an intent should be processed"""
        metadata = {
            QueryIntent.SUMMARIZATION: {
                "requires_structured_output": True,
                "preferred_sources": ["pubmed", "clinical_trial"],
                "min_documents": 1,
                "max_documents": 3,
            },
            QueryIntent.COMPARISON: {
                "requires_structured_output": True,
                "preferred_sources": ["pubmed", "clinical_trial", "fda"],
                "min_documents": 4,
                "max_documents": 8,
            },
            QueryIntent.REGULATORY_COMPLIANCE: {
                "requires_structured_output": False,
                "preferred_sources": ["fda", "ema", "regulatory"],
                "min_documents": 2,
                "max_documents": 6,
            },
            QueryIntent.ADVERSE_EVENTS: {
                "requires_structured_output": True,
                "preferred_sources": ["pubmed", "fda", "clinical_trial"],
                "min_documents": 3,
                "max_documents": 6,
            },
            QueryIntent.CLINICAL_TRIAL: {
                "requires_structured_output": True,
                "preferred_sources": ["clinical_trial", "pubmed"],
                "min_documents": 2,
                "max_documents": 5,
            },
            QueryIntent.FACTUAL_LOOKUP: {
                "requires_structured_output": False,
                "preferred_sources": None,
                "min_documents": 2,
                "max_documents": 4,
            },
            QueryIntent.DRUG_INTERACTION: {
                "requires_structured_output": False,
                "preferred_sources": ["pubmed", "fda", "drugbank"],
                "min_documents": 2,
                "max_documents": 4,
            },
            QueryIntent.GENERAL_QA: {
                "requires_structured_output": False,
                "preferred_sources": None,
                "min_documents": 3,
                "max_documents": 5,
            },
        }
        
        return metadata.get(intent, metadata[QueryIntent.GENERAL_QA])


def get_intent_router(llm) -> IntentRouter:
    """Create an intent router instance"""
    return IntentRouter(llm)
