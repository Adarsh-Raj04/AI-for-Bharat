"""
Intent Classifier - Categorizes incoming queries and routes them appropriately
"""
import re
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query intent types"""
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    FACTUAL_LOOKUP = "factual_lookup"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    ADVERSE_EVENTS = "adverse_events"
    CLINICAL_TRIAL = "clinical_trial"
    DRUG_INTERACTION = "drug_interaction"
    GENERAL_QA = "general_qa"


class IntentClassifier:
    """
    Classifies user queries into intent categories
    Uses rule-based + keyword matching for hackathon
    Production: Use fine-tuned classifier model
    """
    
    def __init__(self):
        self.intent_patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build regex patterns for each intent"""
        return {
            QueryIntent.SUMMARIZATION: [
                r'\b(summarize|summary|overview|abstract|tldr|key findings?)\b',
                r'\bpmid:?\s*\d+\b',
                r'\b(what (is|are) the (main|key) (findings?|results?|conclusions?))\b',
                r'\b(give me (a|an) (summary|overview))\b',
            ],
            QueryIntent.COMPARISON: [
                r'\b(compare|comparison|versus|vs\.?|difference between)\b',
                r'\b(which is (better|more effective|safer))\b',
                r'\b(how do(es)? .+ compare (to|with))\b',
                r'\b(advantages? (and|vs) disadvantages?)\b',
                r'\b(pros and cons)\b',
            ],
            QueryIntent.FACTUAL_LOOKUP: [
                r'\b(what is|define|definition of|meaning of)\b',
                r'\b(how (does|do|is|are))\b',
                r'\b(explain|describe)\b',
                r'\b(mechanism of action|moa)\b',
                r'\b(side effects?|adverse (events?|reactions?))\b',
            ],
            QueryIntent.REGULATORY_COMPLIANCE: [
                r'\b(fda|ema|regulatory|approval|guidance|compliance)\b',
                r'\b(clinical trial (phase|protocol|design))\b',
                r'\b(ich guidelines?|gcp|glp)\b',
                r'\b(label(l?ing)?|package insert)\b',
                r'\b(black box warning|contraindication)\b',
            ],
            QueryIntent.ADVERSE_EVENTS: [
                r'\b(adverse (event|reaction|effect)|side effect|safety)\b',
                r'\b(toxicity|toxic)\b',
                r'\b(contraindication|warning|precaution)\b',
                r'\b(drug safety|pharmacovigilance)\b',
            ],
            QueryIntent.CLINICAL_TRIAL: [
                r'\b(clinical trial|study|trial)\b',
                r'\bnct\d+\b',
                r'\b(phase [i1-3]{1,3}|phase [1-3])\b',
                r'\b(randomized|placebo|double-blind)\b',
                r'\b(efficacy|endpoint|outcome)\b',
            ],
            QueryIntent.DRUG_INTERACTION: [
                r'\b(drug interaction|interact(s|ion) with)\b',
                r'\b(can i take .+ with)\b',
                r'\b(combination (therapy|treatment))\b',
                r'\b(contraindicated with)\b',
            ],
        }
    
    def classify(self, query: str) -> Tuple[QueryIntent, float, Dict]:
        """
        Classify query intent
        
        Args:
            query: User's query text
            
        Returns:
            Tuple of (intent, confidence, metadata)
        """
        query_lower = query.lower()
        
        # Score each intent
        intent_scores = {}
        matched_patterns = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
                    matches.append(pattern)
            
            if score > 0:
                intent_scores[intent] = score
                matched_patterns[intent] = matches
        
        # Determine primary intent
        if not intent_scores:
            return QueryIntent.GENERAL_QA, 0.5, {"matched_patterns": []}
        
        # Get intent with highest score
        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]
        
        # Calculate confidence (normalize by number of patterns)
        total_patterns = len(self.intent_patterns[primary_intent])
        confidence = min(0.95, 0.6 + (max_score / total_patterns) * 0.35)
        
        metadata = {
            "matched_patterns": matched_patterns.get(primary_intent, []),
            "all_scores": {k.value: v for k, v in intent_scores.items()},
            "query_length": len(query.split()),
        }
        
        logger.info(f"Classified query as {primary_intent.value} with confidence {confidence:.2f}")
        
        return primary_intent, confidence, metadata
    
    def get_routing_config(self, intent: QueryIntent) -> Dict:
        """
        Get routing configuration for an intent
        
        Args:
            intent: Classified intent
            
        Returns:
            Configuration dict with retrieval and generation parameters
        """
        configs = {
            QueryIntent.SUMMARIZATION: {
                "top_k": 3,  # Fewer, more focused documents
                "rerank_top_k": 2,
                "max_tokens": 1500,
                "temperature": 0.3,  # More deterministic
                "source_types": ["pubmed", "clinical_trial"],
            },
            QueryIntent.COMPARISON: {
                "top_k": 12,  # More documents for comparison
                "rerank_top_k": 6,
                "max_tokens": 2500,
                "temperature": 0.4,
                "source_types": ["pubmed", "clinical_trial", "fda"],
            },
            QueryIntent.FACTUAL_LOOKUP: {
                "top_k": 8,
                "rerank_top_k": 4,
                "max_tokens": 1500,
                "temperature": 0.2,  # Very deterministic
                "source_types": ["pubmed", "fda", "textbook"],
            },
            QueryIntent.REGULATORY_COMPLIANCE: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.2,
                "source_types": ["fda", "ema", "regulatory"],
            },
            QueryIntent.ADVERSE_EVENTS: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.3,
                "source_types": ["pubmed", "fda", "clinical_trial"],
            },
            QueryIntent.CLINICAL_TRIAL: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.3,
                "source_types": ["clinical_trial", "pubmed"],
            },
            QueryIntent.DRUG_INTERACTION: {
                "top_k": 8,
                "rerank_top_k": 4,
                "max_tokens": 1500,
                "temperature": 0.2,
                "source_types": ["pubmed", "fda", "drugbank"],
            },
            QueryIntent.GENERAL_QA: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.5,
                "source_types": None,  # No filtering
            },
        }
        
        return configs.get(intent, configs[QueryIntent.GENERAL_QA])


# Singleton instance
_intent_classifier = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the intent classifier singleton"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
